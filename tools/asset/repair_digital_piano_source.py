#!/usr/bin/env python3
"""Remove cumulative material geometry from the digital-piano OBJ/URDF source.

The source asset represents several material groups as cumulative meshes: the
same triangle is authored in two or more OBJ files with different materials.
That produces z-fighting after URDF-to-USD conversion.  This tool rewrites a
staged source tree in place so each visible triangle has one material owner and
each link has one material-free collision mesh.
"""

from __future__ import annotations

import argparse
import copy
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path


ROUND_DIGITS = 7
BLACK_KEY_MATERIAL = "material_0_24"
WHITE_KEY_MATERIAL = "material_2_24"


@dataclass(frozen=True)
class Corner:
    position: tuple[float, float, float]
    texcoord: tuple[float, float] | None = None


@dataclass(frozen=True)
class Triangle:
    corners: tuple[Corner, Corner, Corner]

    @property
    def geometry_key(self) -> tuple[tuple[float, float, float], ...]:
        vertices = (
            tuple(round(component, ROUND_DIGITS) for component in corner.position)
            for corner in self.corners
        )
        return tuple(sorted(vertices))


@dataclass
class ObjMesh:
    path: Path
    mtllib: str | None
    object_name: str
    material: str | None
    triangles: list[Triangle]


@dataclass
class RepairSummary:
    links_processed: int = 0
    cross_material_faces_removed: int = 0
    intra_mesh_faces_removed: int = 0
    visual_elements_removed: int = 0
    collision_elements_removed: int = 0
    collision_meshes_created: int = 0
    black_keys_merged: int = 0


def _parse_index(raw: str, size: int) -> int:
    value = int(raw)
    index = value - 1 if value > 0 else size + value
    if index < 0 or index >= size:
        raise ValueError(f"OBJ index {value} is outside collection of size {size}")
    return index


def _load_obj(path: Path) -> ObjMesh:
    positions: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    triangles: list[Triangle] = []
    mtllib: str | None = None
    material: str | None = None
    object_name = path.stem

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        directive = parts[0].lower()
        if directive == "mtllib" and len(parts) >= 2:
            mtllib = " ".join(parts[1:])
        elif directive == "o" and len(parts) >= 2:
            object_name = " ".join(parts[1:])
        elif directive == "usemtl" and len(parts) >= 2:
            candidate = " ".join(parts[1:])
            if material is not None and candidate != material:
                raise ValueError(f"{path}:{line_number}: multiple materials in one split OBJ")
            material = candidate
        elif directive == "v" and len(parts) >= 4:
            positions.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif directive == "vt" and len(parts) >= 3:
            texcoords.append((float(parts[1]), float(parts[2])))
        elif directive == "f":
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_number}: face has fewer than three vertices")
            corners: list[Corner] = []
            for token in parts[1:]:
                indices = token.split("/")
                position = positions[_parse_index(indices[0], len(positions))]
                texcoord = None
                if len(indices) >= 2 and indices[1]:
                    texcoord = texcoords[_parse_index(indices[1], len(texcoords))]
                corners.append(Corner(position=position, texcoord=texcoord))
            for offset in range(1, len(corners) - 1):
                triangles.append(Triangle((corners[0], corners[offset], corners[offset + 1])))

    if not positions or not triangles:
        raise ValueError(f"OBJ has no triangle geometry: {path}")
    return ObjMesh(path=path, mtllib=mtllib, object_name=object_name, material=material, triangles=triangles)


def _write_obj(mesh: ObjMesh, triangles: list[Triangle], *, material_free: bool = False) -> None:
    lines: list[str] = []
    if not material_free and mesh.mtllib:
        lines.append(f"mtllib {mesh.mtllib}")
    lines.append(f"o {mesh.object_name}")
    if not material_free and mesh.material:
        lines.append(f"usemtl {mesh.material}")

    vertex_index = 1
    texcoord_index = 1
    for triangle in triangles:
        face_tokens: list[str] = []
        pending_texcoords: list[tuple[float, float]] = []
        for corner in triangle.corners:
            lines.append("v {:.9g} {:.9g} {:.9g}".format(*corner.position))
            if not material_free and corner.texcoord is not None:
                pending_texcoords.append(corner.texcoord)
                face_tokens.append(f"{vertex_index}/{texcoord_index + len(pending_texcoords) - 1}")
            else:
                face_tokens.append(str(vertex_index))
            vertex_index += 1
        for texcoord in pending_texcoords:
            lines.append("vt {:.9g} {:.9g}".format(*texcoord))
        texcoord_index += len(pending_texcoords)
        lines.append("f " + " ".join(face_tokens))

    mesh.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unique_triangles(mesh: ObjMesh) -> tuple[dict[tuple, Triangle], int]:
    unique: dict[tuple, Triangle] = {}
    duplicates = 0
    for triangle in mesh.triangles:
        key = triangle.geometry_key
        if key in unique:
            duplicates += 1
            continue
        unique[key] = triangle
    return unique, duplicates


def _mesh_path(urdf_path: Path, element: ET.Element) -> Path:
    mesh = element.find("geometry/mesh")
    if mesh is None or not mesh.get("filename"):
        raise ValueError("Every visual/collision must contain a mesh filename")
    filename = mesh.get("filename").replace("\\", "/")
    if filename.startswith("package://"):
        raise ValueError(f"package:// mesh references are not supported: {filename}")
    path = Path(filename)
    if not path.is_absolute():
        path = urdf_path.parent / path
    path = path.resolve()
    if path.suffix.lower() != ".obj" or not path.is_file():
        raise ValueError(f"Expected an existing OBJ mesh: {path}")
    return path


def _transform_signature(element: ET.Element) -> tuple[str, str, str]:
    origin = element.find("origin")
    xyz = " ".join((origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split())
    rpy = " ".join((origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split())
    mesh = element.find("geometry/mesh")
    scale = " ".join((mesh.get("scale", "1 1 1") if mesh is not None else "1 1 1").split())
    return xyz, rpy, scale


def _preflight(root: ET.Element, urdf_path: Path) -> dict[int, ObjMesh]:
    loaded: dict[int, ObjMesh] = {}
    meshes_by_path: dict[Path, ObjMesh] = {}
    for link in root.findall("link"):
        collisions = link.findall("collision")
        if collisions:
            signatures = {_transform_signature(element) for element in collisions}
            if len(signatures) != 1:
                name = link.get("name", "<unnamed>")
                raise ValueError(f"{name} collision origins or scales do not match")
        for element in (*link.findall("visual"), *collisions):
            path = _mesh_path(urdf_path, element)
            if path not in meshes_by_path:
                meshes_by_path[path] = _load_obj(path)
            loaded[id(element)] = meshes_by_path[path]
    return loaded


def _is_black_key(
    link: ET.Element,
    visuals: list[ET.Element],
    unique_by_element: dict[int, dict[tuple, Triangle]],
    loaded: dict[int, ObjMesh],
    threshold: float,
) -> tuple[ET.Element, ET.Element] | None:
    if not link.get("name", "").endswith("_key"):
        return None
    black = next((element for element in visuals if loaded[id(element)].material == BLACK_KEY_MATERIAL), None)
    white = next((element for element in visuals if loaded[id(element)].material == WHITE_KEY_MATERIAL), None)
    if black is None or white is None:
        return None
    black_keys = set(unique_by_element[id(black)])
    white_keys = set(unique_by_element[id(white)])
    if not black_keys:
        return None
    overlap_ratio = len(black_keys & white_keys) / len(black_keys)
    return (black, white) if overlap_ratio >= threshold else None


def _replace_collisions(
    link: ET.Element,
    collisions: list[ET.Element],
    loaded: dict[int, ObjMesh],
    urdf_path: Path,
    summary: RepairSummary,
) -> None:
    if not collisions:
        return
    unique: dict[tuple, Triangle] = {}
    for collision in collisions:
        mesh_unique, duplicates = _unique_triangles(loaded[id(collision)])
        summary.intra_mesh_faces_removed += duplicates
        for key, triangle in mesh_unique.items():
            unique.setdefault(key, triangle)

    first_mesh = loaded[id(collisions[0])]
    output_path = first_mesh.path.parent / f"{link.get('name', 'link')}__collision_clean.obj"
    output_mesh = ObjMesh(
        path=output_path,
        mtllib=None,
        object_name=output_path.stem,
        material=None,
        triangles=list(unique.values()),
    )
    _write_obj(output_mesh, list(unique.values()), material_free=True)

    replacement = copy.deepcopy(collisions[0])
    replacement.find("geometry/mesh").set(
        "filename", "./" + output_path.relative_to(urdf_path.parent).as_posix()
    )
    first_index = list(link).index(collisions[0])
    for collision in collisions:
        link.remove(collision)
    link.insert(first_index, replacement)
    summary.collision_elements_removed += len(collisions) - 1
    summary.collision_meshes_created += 1


def clean_piano_source(
    urdf_path: str | Path,
    *,
    black_key_overlap_threshold: float = 0.95,
) -> RepairSummary:
    """Rewrite one staged piano URDF asset tree in place."""
    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if not 0.0 <= black_key_overlap_threshold <= 1.0:
        raise ValueError("black_key_overlap_threshold must be between 0 and 1")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    loaded = _preflight(root, urdf_path)
    summary = RepairSummary()

    for link in root.findall("link"):
        summary.links_processed += 1
        visuals = link.findall("visual")
        unique_by_element: dict[int, dict[tuple, Triangle]] = {}
        for visual in visuals:
            unique, duplicates = _unique_triangles(loaded[id(visual)])
            unique_by_element[id(visual)] = unique
            summary.intra_mesh_faces_removed += duplicates

        black_pair = _is_black_key(
            link,
            visuals,
            unique_by_element,
            loaded,
            black_key_overlap_threshold,
        )
        kept_by_element: dict[int, list[Triangle]] = {}
        if black_pair is not None:
            black, white = black_pair
            union = dict(unique_by_element[id(black)])
            white_unique = unique_by_element[id(white)]
            overlap = set(union) & set(white_unique)
            summary.cross_material_faces_removed += len(overlap)
            for key, triangle in white_unique.items():
                union.setdefault(key, triangle)
            kept_by_element[id(black)] = list(union.values())
            kept_by_element[id(white)] = []
            for visual in visuals:
                if visual not in black_pair:
                    kept_by_element[id(visual)] = list(unique_by_element[id(visual)].values())
            summary.black_keys_merged += 1
        else:
            owner: dict[tuple, int] = {}
            total_unique = 0
            for index, visual in enumerate(visuals):
                current = unique_by_element[id(visual)]
                total_unique += len(current)
                for key in current:
                    owner[key] = index
            summary.cross_material_faces_removed += total_unique - len(owner)
            for index, visual in enumerate(visuals):
                kept_by_element[id(visual)] = [
                    triangle
                    for key, triangle in unique_by_element[id(visual)].items()
                    if owner[key] == index
                ]

        for visual in list(visuals):
            kept = kept_by_element[id(visual)]
            if kept:
                _write_obj(loaded[id(visual)], kept)
            else:
                link.remove(visual)
                summary.visual_elements_removed += 1

        _replace_collisions(link, link.findall("collision"), loaded, urdf_path, summary)

    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    remaining = find_cross_material_overlaps(urdf_path)
    if remaining:
        raise RuntimeError(f"Cross-material overlap remains after repair: {remaining}")
    return summary


def find_cross_material_overlaps(urdf_path: str | Path) -> dict[str, int]:
    """Return exact coplanar triangle counts for different visual materials."""
    urdf_path = Path(urdf_path).expanduser().resolve()
    root = ET.parse(urdf_path).getroot()
    overlaps: dict[str, int] = {}
    for link in root.findall("link"):
        by_signature: dict[tuple[str, str, str], list[tuple[ObjMesh, set[tuple]]]] = {}
        for visual in link.findall("visual"):
            mesh = _load_obj(_mesh_path(urdf_path, visual))
            unique, _ = _unique_triangles(mesh)
            by_signature.setdefault(_transform_signature(visual), []).append((mesh, set(unique)))
        for entries in by_signature.values():
            for left_index, (left_mesh, left_keys) in enumerate(entries):
                for right_mesh, right_keys in entries[left_index + 1 :]:
                    if left_mesh.material == right_mesh.material:
                        continue
                    count = len(left_keys & right_keys)
                    if count:
                        name = link.get("name", "<unnamed>")
                        key = f"{name}:{left_mesh.material}|{right_mesh.material}"
                        overlaps[key] = count
    return overlaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path, help="Staged piano URDF to rewrite in place")
    parser.add_argument("--black-key-overlap-threshold", type=float, default=0.95)
    args = parser.parse_args()
    summary = clean_piano_source(
        args.urdf,
        black_key_overlap_threshold=args.black_key_overlap_threshold,
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
