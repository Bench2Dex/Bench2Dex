# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format, with optional mesh-name sanitization.

This script is based on ``convert_urdf.py`` and adds an automatic pre-processing
step to sanitize mesh filenames that can break USD path creation in the URDF importer.
The sanitize workflow follows:
1) Rename mesh files whose paths contain USD-invalid token characters.
2) Update ``mtllib`` lines inside OBJ files accordingly.
3) Generate ``*_sanitized.urdf`` by updating mesh references.
4) Convert sanitized URDF to USD.
"""

"""Launch Isaac Sim Simulator first."""

import hashlib
import os
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

from isaaclab.app import AppLauncher
from convert_urdf_cli import (
    build_arg_parser,
    build_urdf_converter_cfg,
    resolve_collision_approximation_strategy,
)

parser = build_arg_parser(add_app_launcher_args=AppLauncher.add_app_launcher_args)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib

import carb
import omni.kit.app
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import modify_mesh_collision_properties, schemas_cfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict


def _detect_needs_sanitize(urdf_text: str) -> bool:
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        # Fallback for malformed XML: keep legacy regex behavior.
        return bool(re.search(r"filename\s*=\s*['\"][^'\"]*-[^'\"]*['\"]", urdf_text))

    robot_name = root.attrib.get("name")
    if robot_name and _sanitize_usd_name(robot_name) != robot_name:
        return True

    for tag in ("link", "joint", "visual", "collision"):
        for elem in root.iter(tag):
            name = elem.attrib.get("name")
            if not name:
                continue
            if _sanitize_usd_name(name) != name:
                return True

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        posix_name = _to_posix(filename)
        stem = Path(posix_name).stem
        if _sanitize_usd_name(stem) != stem:
            return True
        if "-" in posix_name:
            return True
    return False


def _to_posix(path_str: str) -> str:
    return path_str.replace("\\", "/")


def _sanitize_rel_path(rel_path: str) -> str:
    posix_path = _to_posix(rel_path)
    parts = posix_path.split("/")
    sanitized_parts: list[str] = []
    for idx, part in enumerate(parts):
        if idx == len(parts) - 1:
            suffix = Path(part).suffix
            stem = Path(part).stem
            sanitized_stem = _sanitize_usd_name(stem)
            sanitized_parts.append(f"{sanitized_stem}{suffix}")
        else:
            sanitized_parts.append(_sanitize_usd_name(part))
    return "/".join(sanitized_parts)


def _sanitize_usd_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not sanitized:
        sanitized = "_"
    if not re.match(r"[A-Za-z_]", sanitized[0]):
        sanitized = f"_{sanitized}"
    return sanitized


def _dedupe_name(base_name: str, original_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name

    suffix = hashlib.sha1(original_name.encode("utf-8")).hexdigest()[:4]
    candidate = f"{base_name}_{suffix}"
    if candidate not in used_names:
        used_names.add(candidate)
        return candidate

    index = 1
    while True:
        candidate = f"{base_name}_{suffix}_{index}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        index += 1


def _sanitize_urdf_xml_names(urdf_text: str) -> str:
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError:
        return urdf_text

    used_by_tag: dict[str, set[str]] = {tag: set() for tag in ("link", "joint", "visual", "collision")}
    changed = False

    robot_name = root.attrib.get("name")
    if robot_name:
        sanitized_robot_name = _sanitize_usd_name(robot_name)
        if sanitized_robot_name != robot_name:
            root.set("name", sanitized_robot_name)
            changed = True

    for tag in ("link", "joint", "visual", "collision"):
        for elem in root.iter(tag):
            original_name = elem.attrib.get("name")
            if not original_name:
                continue
            base_name = _sanitize_usd_name(original_name)
            unique_name = _dedupe_name(base_name, original_name, used_by_tag[tag])
            if unique_name != original_name:
                elem.set("name", unique_name)
                changed = True

    if not changed:
        return urdf_text

    return ET.tostring(root, encoding="unicode")


def _sanitize_mesh_names(urdf_path: Path, inplace_mesh_update: bool) -> Path:
    asset_dir = urdf_path.parent
    # Build rename maps for mesh assets whose paths need USD-token sanitization
    # (excluding usd/urdf/json/yaml etc.; only files likely used by mesh refs).
    rename_map_rel: dict[str, str] = {}
    allowed_ext = {
        ".obj",
        ".mtl",
        ".stl",
        ".fbx",
        ".dae",
        ".gltf",
        ".glb",
        ".png",
        ".jpg",
        ".jpeg",
        ".tga",
        ".bmp",
        ".tif",
        ".tiff",
    }
    for src in asset_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = _to_posix(str(src.relative_to(asset_dir)))
        if src.suffix.lower() not in allowed_ext:
            continue
        dst_rel = _sanitize_rel_path(rel)
        if dst_rel != rel:
            rename_map_rel[rel] = dst_rel

    if not rename_map_rel:
        urdf_text = urdf_path.read_text(encoding="utf-8")
        urdf_text = _sanitize_urdf_xml_names(urdf_text)
        sanitized_urdf = urdf_path.with_name(f"{urdf_path.stem}_sanitized{urdf_path.suffix}")
        sanitized_urdf.write_text(urdf_text, encoding="utf-8")
        return sanitized_urdf

    # Apply rename/copy for all mapped files.
    # Deepest paths first avoids parent directory edge-cases.
    sorted_pairs = sorted(rename_map_rel.items(), key=lambda kv: len(kv[0]), reverse=True)

    files_to_rewrite: set[Path] = set()
    if inplace_mesh_update:
        for old_rel, new_rel in sorted_pairs:
            src = asset_dir / old_rel
            dst = asset_dir / new_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                src.rename(dst)
            files_to_rewrite.add(dst)
    else:
        # Non-destructive mode: keep original files and create sanitized copies.
        for old_rel, new_rel in sorted_pairs:
            src = asset_dir / old_rel
            dst = asset_dir / new_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            files_to_rewrite.add(dst)

    # Update OBJ/MTL textual references to renamed paths.  Every material file
    # must be considered: a validly named MTL can reference a texture whose
    # filename was sanitized.  Restrict replacement to reference directives;
    # applying every filename substitution to every OBJ vertex line turns a
    # 100+ MB mesh into an avoidable quadratic text rewrite.
    path_replace_pairs = sorted(rename_map_rel.items(), key=lambda kv: len(kv[0]), reverse=True)
    basename_replace_map = {
        Path(old).name: Path(new).name
        for old, new in rename_map_rel.items()
        if Path(old).name != Path(new).name
    }

    def _remap_reference(reference: str) -> str:
        normalized = _to_posix(reference)
        direct = normalized[2:] if normalized.startswith("./") else normalized
        for old_rel, new_rel in path_replace_pairs:
            if direct == old_rel:
                return ("./" if normalized.startswith("./") else "") + new_rel
        old_name = Path(normalized).name
        new_name = basename_replace_map.get(old_name)
        if new_name is None:
            return reference
        return f"{reference[:-len(old_name)]}{new_name}"

    def _rewrite_reference_line(line: str) -> str:
        content, newline = (line[:-1], "\n") if line.endswith("\n") else (line, "")
        stripped = content.lstrip()
        indent = content[:len(content) - len(stripped)]
        if stripped.lower().startswith("mtllib "):
            reference = stripped.split(maxsplit=1)[1].strip()
            return f"{indent}mtllib {_remap_reference(reference)}{newline}"
        parts = stripped.split()
        if len(parts) >= 2 and parts[0].lower() in {"map_kd", "map_ks", "map_d", "bump", "norm"}:
            reference = parts[-1]
            remapped = _remap_reference(reference)
            if remapped != reference:
                return f"{indent}{' '.join(parts[:-1])} {remapped}{newline}"
        return line

    rewrite_candidates = [
        path for path in asset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".obj", ".mtl"}
    ]
    for text_file in rewrite_candidates:
        text = text_file.read_text(encoding="utf-8", errors="ignore")
        updated = "".join(_rewrite_reference_line(line) for line in text.splitlines(keepends=True))
        if updated != text:
            text_file.write_text(updated, encoding="utf-8")

    # Create sanitized URDF with updated mesh names.
    urdf_text = urdf_path.read_text(encoding="utf-8")
    for old_rel, new_rel in path_replace_pairs:
        urdf_text = urdf_text.replace(old_rel, new_rel)
        urdf_text = urdf_text.replace(_to_posix(old_rel), _to_posix(new_rel))
        urdf_text = urdf_text.replace(old_rel.replace("/", "\\"), new_rel.replace("/", "\\"))

    urdf_text = _sanitize_urdf_xml_names(urdf_text)

    sanitized_urdf = urdf_path.with_name(f"{urdf_path.stem}_sanitized{urdf_path.suffix}")
    sanitized_urdf.write_text(urdf_text, encoding="utf-8")
    return sanitized_urdf


def _validate_mesh_references(urdf_path: Path):
    try:
        root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        raise RuntimeError(f"Sanitized URDF is not valid XML: {urdf_path}\n{exc}") from exc

    invalid_name_refs: list[str] = []
    missing_refs: list[str] = []
    base_dir = urdf_path.parent

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        posix_name = _to_posix(filename)
        stem = Path(posix_name).stem
        if _sanitize_usd_name(stem) != stem:
            invalid_name_refs.append(filename)

        if posix_name.startswith("package://"):
            continue

        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = base_dir / filename
        if not mesh_path.exists():
            missing_refs.append(filename)

    if invalid_name_refs:
        bad = "\n".join(f"- {item}" for item in sorted(set(invalid_name_refs)))
        raise RuntimeError(
            "Sanitized URDF still contains invalid mesh token names (for USD prim paths):\n"
            f"{bad}"
        )

    if missing_refs:
        bad = "\n".join(f"- {item}" for item in sorted(set(missing_refs)))
        raise RuntimeError(
            "Sanitized URDF references missing mesh files:\n"
            f"{bad}"
        )


def _validate_usd_output(usd_path: Path):
    config_dir = usd_path.parent / "configuration"
    expected = [
        usd_path,
        config_dir / f"{usd_path.stem}_base.usd",
        config_dir / f"{usd_path.stem}_physics.usd",
        config_dir / f"{usd_path.stem}_robot.usd",
        config_dir / f"{usd_path.stem}_sensor.usd",
    ]

    missing = [str(path) for path in expected if not path.exists()]
    tiny = [str(path) for path in expected if path.exists() and path.stat().st_size <= 600]

    if missing or tiny:
        lines = []
        if missing:
            lines.append("Missing files:")
            lines.extend(f"- {path}" for path in missing)
        if tiny:
            lines.append("Suspicious tiny USD files (likely failed import):")
            lines.extend(f"- {path}" for path in tiny)
        raise RuntimeError(
            "URDF conversion produced invalid/empty USD outputs.\n"
            + "\n".join(lines)
            + "\nCheck for 'Used null prim' or 'Ill-formed SdfPath' in importer logs."
        )


def _apply_collision_approximation_postprocess(usd_path: Path, approximation_name: str | None) -> int:
    if approximation_name is None:
        return 0
    if approximation_name != "sdf":
        raise ValueError(f"Unsupported postprocess collision approximation: {approximation_name}")

    # URDF importer writes collision-authoring opinions into the physics layer, while the top-level USD
    # references those collider prims. We must patch the physics layer directly.
    physics_usd_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    target_usd_path = physics_usd_path if physics_usd_path.exists() else usd_path

    stage = Usd.Stage.Open(target_usd_path.as_posix(), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD stage for postprocess: {target_usd_path}")

    modified = 0
    sdf_cfg = schemas_cfg.SDFMeshPropertiesCfg()
    for prim in stage.Traverse():
        if prim.IsInstanceProxy():
            continue
        prim_path = prim.GetPath().pathString
        if not prim_path.startswith("/colliders/"):
            continue
        # URDF-generated collider schemas are attached to the collider Xform (for example ".../World"),
        # not to the leaf Mesh prim itself, so target collider-authoring prims directly.
        if not (UsdPhysics.CollisionAPI(prim) or UsdPhysics.MeshCollisionAPI(prim)):
            continue
        modify_mesh_collision_properties(prim_path, sdf_cfg, stage=stage)
        modified += 1

    stage.GetRootLayer().Save()
    return modified


def _srgb_to_linear(value: float) -> float:
    """Convert an sRGB color component to the linear value expected by OmniPBR."""
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _normalize_urdf_omnipbr_materials(usd_path: Path) -> int:
    """Correct color-space and invalid emission authored by the URDF importer.

    OBJ/MTL diffuse colors are authored in display (sRGB) space.  The URDF
    importer writes them unchanged to OmniPBR's ``diffuse_color_constant``,
    which is a linear color input.  This makes mid-tone materials visibly
    washed out under RTX lighting. Texture maps are already treated as sRGB
    by OmniPBR and are deliberately left untouched. The importer also authors
    white emission at intensity 10000 on every material despite source MTLs
    having no ``Ke`` term; clear it explicitly.
    """
    base_usd_path = usd_path.parent / "configuration" / f"{usd_path.stem}_base.usd"
    if not base_usd_path.exists():
        return 0

    stage = Usd.Stage.Open(base_usd_path.as_posix(), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open generated USD material layer: {base_usd_path}")

    modified = 0
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Shader":
            continue
        mdl_asset = prim.GetAttribute("info:mdl:sourceAsset").Get()
        if mdl_asset is None or Path(mdl_asset.path).name not in {"OmniPBR.mdl", "OmniPBR_Opacity.mdl"}:
            continue
        color_attr = prim.GetAttribute("inputs:diffuse_color_constant")
        color = color_attr.Get()
        if color is not None:
            corrected = Gf.Vec3f(*(_srgb_to_linear(component) for component in color))
            if corrected != color:
                color_attr.Set(corrected)
                modified += 1

        emission_color_attr = prim.GetAttribute("inputs:emissive_color")
        emission_intensity_attr = prim.GetAttribute("inputs:emissive_intensity")
        emission_enabled_attr = prim.GetAttribute("inputs:enable_emission")
        emission_color = emission_color_attr.Get()
        emission_intensity = emission_intensity_attr.Get()
        if emission_color is not None and emission_color != Gf.Vec3f(0.0, 0.0, 0.0):
            emission_color_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
            modified += 1
        if emission_intensity is not None and float(emission_intensity) != 0.0:
            emission_intensity_attr.Set(0.0)
            modified += 1
        if emission_enabled_attr.Get() is not False:
            emission_enabled_attr.Set(False)
            modified += 1

    if modified:
        stage.GetRootLayer().Save()
    return modified


def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    import_urdf_path = Path(urdf_path)
    sanitize_workspace: Path | None = None
    urdf_text = import_urdf_path.read_text(encoding="utf-8")
    needs_sanitize = (not args_cli.disable_sanitize_invalid_names) and _detect_needs_sanitize(urdf_text)
    if needs_sanitize and not args_cli.sanitize_inplace:
        # Sanitize on a temporary copy to avoid mutating source assets.
        workspace_parent = Path(dest_path).parent
        workspace_parent.mkdir(parents=True, exist_ok=True)
        sanitize_workspace = workspace_parent / f".isaaclab_urdf_sanitize_{os.getpid()}"
        if sanitize_workspace.exists():
            shutil.rmtree(sanitize_workspace, ignore_errors=True)
        sanitize_workspace.mkdir(parents=True, exist_ok=True)
        workspace_asset_dir = sanitize_workspace / import_urdf_path.parent.name
        ignore_names = {sanitize_workspace.name, Path(dest_path).parent.name}
        ignore_prefix = ".isaaclab_urdf_sanitize_"

        def _copytree_ignore(_dir: str, names: list[str]) -> list[str]:
            return [name for name in names if (name in ignore_names) or name.startswith(ignore_prefix)]

        shutil.copytree(
            import_urdf_path.parent,
            workspace_asset_dir,
            dirs_exist_ok=True,
            ignore=_copytree_ignore,
        )
        import_urdf_path = workspace_asset_dir / import_urdf_path.name
        print(f"[INFO] Sanitization workspace: {sanitize_workspace}")

    if needs_sanitize:
        import_urdf_path = _sanitize_mesh_names(import_urdf_path, inplace_mesh_update=True)
        print(f"[INFO] Sanitized URDF generated: {import_urdf_path}")

    _validate_mesh_references(import_urdf_path)

    # Create Urdf converter config
    urdf_converter_cfg = build_urdf_converter_cfg(
        args_cli,
        asset_path=str(import_urdf_path),
        dest_path=dest_path,
        cfg_cls=UrdfConverterCfg,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {import_urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Urdf converter and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    collision_strategy = resolve_collision_approximation_strategy(args_cli.collision_approximation)
    modified_collision_meshes = _apply_collision_approximation_postprocess(
        Path(urdf_converter.usd_path), collision_strategy.postprocess_approximation
    )
    corrected_materials = _normalize_urdf_omnipbr_materials(Path(urdf_converter.usd_path))
    _validate_usd_output(Path(urdf_converter.usd_path))
    # print output
    print("URDF importer output:")
    print(f"Generated USD file: {urdf_converter.usd_path}")
    if collision_strategy.postprocess_approximation is not None:
        print(
            f"Applied postprocess collision approximation '{collision_strategy.postprocess_approximation}'"
            f" to {modified_collision_meshes} mesh collider prim(s)."
        )
    print(f"Normalized color space and cleared invalid emission on {corrected_materials} OmniPBR inputs.")
    print("-" * 80)
    print("-" * 80)

    # Determine if there is a GUI to update:
    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    # Preview only when explicitly requested.
    if args_cli.preview_stage and (local_gui or livestream_gui):
        sim_utils.open_stage(urdf_converter.usd_path)
        prim_selection = omni.usd.get_context().get_selection()
        # Remove stale selection entries from previous stages.
        with contextlib.suppress(Exception):
            prim_selection.clear_selected_prim_paths()

        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                if not args_cli.skip_selection_cleanup:
                    with contextlib.suppress(Exception):
                        selected = prim_selection.get_selected_prim_paths()
                        has_invalid = any(
                            (not isinstance(path, str)) or (not path) or (not path.startswith("/"))
                            for path in selected
                        )
                        if has_invalid:
                            prim_selection.clear_selected_prim_paths()
                app.update()

    if sanitize_workspace and (not args_cli.keep_sanitize_workdir):
        shutil.rmtree(sanitize_workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
