"""Utilities for applying contact friction overrides to dexterous hands."""

from __future__ import annotations

from collections.abc import Sequence


DEFAULT_HAND_LINK_KEYWORDS = (
    "thumb",
    "index",
    "middle",
    "ring",
    "little",
    "finger",
    "palm",
    "plam",
)

GEOM_TYPES = {"Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone"}


def set_dexterous_hand_friction(
    robot_prim_path: str,
    *,
    static_friction: float = 2.0,
    dynamic_friction: float = 2.0,
    hand_link_keywords: Sequence[str] = DEFAULT_HAND_LINK_KEYWORDS,
    log_prefix: str = "hand_friction",
) -> None:
    """Bind a high-friction PhysX material to dexterous hand links and collision meshes."""
    try:
        import omni.usd  # type: ignore
        from pxr import Usd, UsdShade, UsdPhysics  # type: ignore
    except ImportError:
        print(f"[{log_prefix}] WARNING: pxr not available, skipping hand friction setup")
        return

    stage = omni.usd.get_context().get_stage()
    keywords = tuple(keyword.lower() for keyword in hand_link_keywords)

    mat_path = f"{robot_prim_path}/HandFrictionMaterial"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    material = UsdShade.Material(mat_prim)

    physics_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    physics_mat.CreateStaticFrictionAttr(static_friction)
    physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
    physics_mat.CreateRestitutionAttr(0.0)

    bound_xform = 0
    bound_mesh = 0
    sample_unbound_under_hand: list[str] = []
    traverse_predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)

    for prim in stage.Traverse(traverse_predicate):
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(robot_prim_path):
            continue

        path_lower = prim_path.lower()
        if "handfrictionmaterial" in path_lower:
            continue
        if not any(keyword in path_lower for keyword in keywords):
            continue

        type_name = prim.GetTypeName()
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        is_collision_branch = "/collision" in path_lower
        is_collision_prim = has_collision_api or (type_name in GEOM_TYPES and is_collision_branch)
        is_hand_link_xform = (
            type_name == "Xform"
            and "/visual" not in path_lower
            and "/collision" not in path_lower
            and "/joint" not in path_lower
        )

        if not (is_collision_prim or is_hand_link_xform):
            if len(sample_unbound_under_hand) < 12 and type_name in GEOM_TYPES:
                sample_unbound_under_hand.append(f"{type_name}@{prim_path}")
            continue

        if prim.IsInstanceProxy():
            if is_collision_prim:
                bound_mesh += 1
            continue

        UsdShade.MaterialBindingAPI.Apply(prim)
        UsdShade.MaterialBindingAPI(prim).Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        if is_collision_prim:
            bound_mesh += 1
        else:
            bound_xform += 1

    print(
        f"[{log_prefix}] hand friction: bound material (sf={static_friction}, df={dynamic_friction}) to "
        f"{bound_xform} hand-link Xforms + {bound_mesh} collision meshes"
    )
    if bound_mesh == 0 and sample_unbound_under_hand:
        print(
            f"[{log_prefix}] WARNING: no collision prims matched. Sample geom prims under hand links: "
            f"{sample_unbound_under_hand}"
        )