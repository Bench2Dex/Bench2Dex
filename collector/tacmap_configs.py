"""TacMap asset registry for supported tactile hands."""

from __future__ import annotations

from dataclasses import dataclass

from collector.tactile import ROBOT_KEY_TO_SITE_NAMES, _resolve_site_body_name


@dataclass(frozen=True)
class TacMapNpyGroup:
    """One TacMap surface map shared by one or more tactile sites."""

    name: str
    points_npy: str
    normals_npy: str
    site_names: tuple[str, ...]


@dataclass(frozen=True)
class TacMapHandCfg:
    """TacMap configuration for one robot hand embodiment."""

    dataset_key: str
    native_resolution: int
    correction_scale: float
    flip_normals: bool
    site_names: tuple[str, ...]
    npy_groups: tuple[TacMapNpyGroup, ...]
    site_body_overrides: dict[str, str] | None = None


def resolve_tacmap_site_body_name(robot_key: str, site_name: str, robot_prim_path: str | None = None) -> str:
    """Resolve the articulation body used by TacMap for one tactile site."""

    cfg = ROBOT_KEY_TO_TACMAP_CFG.get(str(robot_key))
    override = (cfg.site_body_overrides or {}).get(str(site_name)) if cfg is not None else None
    if override:
        return override
    return _resolve_site_body_name(robot_key, site_name, robot_prim_path)


def _hand_sites(robot_key: str, suffix: str | None = None) -> tuple[str, ...]:
    sites = ROBOT_KEY_TO_SITE_NAMES[robot_key]
    return tuple(
        site_name
        for site_name in sites
        if site_name not in ("right_palm", "left_palm")
        and (suffix is None or site_name.endswith(suffix))
    )


def _split_side_groups(
    *,
    robot_key: str,
    suffix: str | None,
    thumb_marker: str,
) -> tuple[tuple[str, ...], tuple[TacMapNpyGroup, ...]]:
    sites = _hand_sites(robot_key, suffix)

    right_thumb = tuple(site for site in sites if site.startswith("right_") and thumb_marker in site)
    right_fingers = tuple(site for site in sites if site.startswith("right_") and thumb_marker not in site)
    left_thumb = tuple(site for site in sites if site.startswith("left_") and thumb_marker in site)
    left_fingers = tuple(site for site in sites if site.startswith("left_") and thumb_marker not in site)

    groups = (
        TacMapNpyGroup(
            "RTH",
            "tactileSensor_map_RTH_point.npy",
            "tactileSensor_map_RTH_normal.npy",
            right_thumb,
        ),
        TacMapNpyGroup(
            "R4F",
            "tactileSensor_map_R4F_point.npy",
            "tactileSensor_map_R4F_normal.npy",
            right_fingers,
        ),
        TacMapNpyGroup(
            "LTH",
            "tactileSensor_map_LTH_point.npy",
            "tactileSensor_map_LTH_normal.npy",
            left_thumb,
        ),
        TacMapNpyGroup(
            "L4F",
            "tactileSensor_map_L4F_point.npy",
            "tactileSensor_map_L4F_normal.npy",
            left_fingers,
        ),
    )
    return sites, groups


def _per_finger_groups(
    *,
    robot_key: str,
    suffix: str,
    fingers: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, ...], tuple[TacMapNpyGroup, ...]]:
    sites = _hand_sites(robot_key, suffix)
    groups: list[TacMapNpyGroup] = []

    for side_prefix, site_prefix in (("R", "right"), ("L", "left")):
        for group_suffix, finger_name in fingers:
            group_name = f"{side_prefix}{group_suffix}"
            site_name = f"{site_prefix}_{finger_name}{suffix}"
            if site_name not in sites:
                raise ValueError(f"TacMap site {site_name!r} not found for robot key {robot_key!r}")
            groups.append(
                TacMapNpyGroup(
                    group_name,
                    f"tactileSensor_map_{group_name}_point.npy",
                    f"tactileSensor_map_{group_name}_normal.npy",
                    (site_name,),
                )
            )

    return sites, tuple(groups)


def _sharpa_cfg() -> TacMapHandCfg:
    sites = _hand_sites("multi_iiwa7_with_sharpa", "_elastomer")
    thumb_sites = tuple(site for site in sites if "_thumb_" in site)
    finger_sites = tuple(site for site in sites if "_thumb_" not in site)
    return TacMapHandCfg(
        dataset_key="kuka+sharpa",
        native_resolution=240,
        correction_scale=1e-3,
        flip_normals=True,
        site_names=sites,
        npy_groups=(
            TacMapNpyGroup(
                "TH",
                "tactileSensor_map_TH_point.npy",
                "tactileSensor_map_TH_normal.npy",
                thumb_sites,
            ),
            TacMapNpyGroup(
                "4F",
                "tactileSensor_map_4F_point.npy",
                "tactileSensor_map_4F_normal.npy",
                finger_sites,
            ),
        ),
    )


def _generic_cfg(
    *,
    robot_key: str,
    dataset_key: str,
    suffix: str | None,
    thumb_marker: str,
    native_resolution: int = 240,
    site_body_overrides: dict[str, str] | None = None,
) -> TacMapHandCfg:
    sites, groups = _split_side_groups(
        robot_key=robot_key,
        suffix=suffix,
        thumb_marker=thumb_marker,
    )
    return TacMapHandCfg(
        dataset_key=dataset_key,
        native_resolution=native_resolution,
        correction_scale=1e-3,
        flip_normals=False,
        site_names=sites,
        npy_groups=groups,
        site_body_overrides=site_body_overrides,
    )


def _per_finger_cfg(
    *,
    robot_key: str,
    dataset_key: str,
    suffix: str,
    fingers: tuple[tuple[str, str], ...],
    native_resolution: int = 240,
    site_body_overrides: dict[str, str] | None = None,
) -> TacMapHandCfg:
    sites, groups = _per_finger_groups(
        robot_key=robot_key,
        suffix=suffix,
        fingers=fingers,
    )
    return TacMapHandCfg(
        dataset_key=dataset_key,
        native_resolution=native_resolution,
        correction_scale=1e-3,
        flip_normals=False,
        site_names=sites,
        npy_groups=groups,
        site_body_overrides=site_body_overrides,
    )


def _schunk_site_body_overrides() -> dict[str, str]:
    suffix_to_body = {
        "Thumb_Flexion": "hand_c",
        "Index_Finger_Distal": "hand_t",
        "Middle_Finger_Distal": "hand_s",
        "Ring_Finger": "hand_r",
        "Pinky": "hand_q",
    }
    return {
        f"{side}_{site_suffix}": f"{side}_{body_suffix}"
        for side in ("right", "left")
        for site_suffix, body_suffix in suffix_to_body.items()
    }


def _rh56dfx_site_body_overrides() -> dict[str, str]:
    suffix_to_body = {
        "thumb_pad": "thumb_rubber_3",
        "index_pad": "index_rubber_2",
        "middle_pad": "middle_rubber_2",
        "ring_pad": "ring_rubber_2",
        "little_pad": "little_rubber_2",
    }
    return {
        f"{side}_{site_suffix}": f"{side}_{body_suffix}"
        for side in ("right", "left")
        for site_suffix, body_suffix in suffix_to_body.items()
    }


def _wuji_site_body_overrides() -> dict[str, str]:
    finger_ids = {
        "thumb": "finger1",
        "index": "finger2",
        "middle": "finger3",
        "ring": "finger4",
        "little": "finger5",
    }
    return {
        f"{side}_{finger}_J4": f"{side}_{finger_id}_tip_link"
        for side in ("right", "left")
        for finger, finger_id in finger_ids.items()
    }


def _allegro_site_body_overrides() -> dict[str, str]:
    link_ids = {
        "index": "3.0",
        "middle": "7.0",
        "ring": "11.0",
        "thumb": "15.0",
    }
    return {
        f"{side}_{finger}_3": f"{prefix}link_{link_id}_tip"
        for side, prefix in (("left", ""), ("right", "multi_"))
        for finger, link_id in link_ids.items()
    }


_STANDARD_FIVE_FINGER_GROUPS = (
    ("TH", "thumb"),
    ("IDX", "index"),
    ("MID", "middle"),
    ("RING", "ring"),
    ("LIT", "little"),
)

_PINKY_FIVE_FINGER_GROUPS = (
    ("TH", "thumb"),
    ("IDX", "index"),
    ("MID", "middle"),
    ("RING", "ring"),
    ("LIT", "pinky"),
)

_ROHAND_FIVE_FINGER_GROUPS = (
    ("TH", "th"),
    ("IDX", "if"),
    ("MID", "mf"),
    ("RING", "rf"),
    ("LIT", "lf"),
)


ROBOT_KEY_TO_TACMAP_CFG: dict[str, TacMapHandCfg] = {
    "multi_iiwa7_with_sharpa": _sharpa_cfg(),
    "multi_ur5_rh56dfx_with_flange": _per_finger_cfg(
        robot_key="multi_ur5_rh56dfx_with_flange",
        dataset_key="ur5+RH56DFX",
        suffix="_pad",
        fingers=_STANDARD_FIVE_FINGER_GROUPS,
        site_body_overrides=_rh56dfx_site_body_overrides(),
    ),
    "multi_ur5_rh5dg2_with_flange": _generic_cfg(
        robot_key="multi_ur5_rh5dg2_with_flange",
        dataset_key="ur5+RH5DG2",
        suffix="_pad",
        thumb_marker="thumb_",
    ),
    "multi_ur5_shadow_hand_with_flange": _generic_cfg(
        robot_key="multi_ur5_shadow_hand_with_flange",
        dataset_key="ur5+shadow_hand",
        suffix="J1",
        thumb_marker="TH",
    ),
    "multi_ur5_schunk_hand_with_flange": _generic_cfg(
        robot_key="multi_ur5_schunk_hand_with_flange",
        dataset_key="ur5+schunk_hand",
        suffix=None,
        thumb_marker="Thumb_",
        site_body_overrides=_schunk_site_body_overrides(),
    ),
    "multi_ur5_wuji_with_flange": _generic_cfg(
        robot_key="multi_ur5_wuji_with_flange",
        dataset_key="ur5+wuji",
        suffix="_J4",
        thumb_marker="thumb_",
        site_body_overrides=_wuji_site_body_overrides(),
    ),
    "multi_panda_with_allegro": _generic_cfg(
        robot_key="multi_panda_with_allegro",
        dataset_key="panda+allegro",
        suffix="_3",
        thumb_marker="thumb_",
        site_body_overrides=_allegro_site_body_overrides(),
    ),
    "multi_panda_with_orca": _generic_cfg(
        robot_key="multi_panda_with_orca",
        dataset_key="panda+orca",
        suffix=None,
        thumb_marker="thumb_",
    ),
    "multi_xarm7_with_ability": _per_finger_cfg(
        robot_key="multi_xarm7_with_ability",
        dataset_key="xarm+ability",
        suffix="_q2",
        fingers=_PINKY_FIVE_FINGER_GROUPS,
    ),
    "multi_xarm7_with_leap": _generic_cfg(
        robot_key="multi_xarm7_with_leap",
        dataset_key="xarm+leap",
        suffix="_2",
        thumb_marker="thumb_",
        native_resolution=240,
    ),
    "multi_jaka_zu7_dexhand021_with_flange": _generic_cfg(
        robot_key="multi_jaka_zu7_dexhand021_with_flange",
        dataset_key="jaka_zu7+dexhand021",
        suffix="_J4",
        thumb_marker="thumb_",
    ),
    "multi_rm_65_with_revo2": _per_finger_cfg(
        robot_key="multi_rm_65_with_revo2",
        dataset_key="rm_65+BrainCo",
        suffix="_touch",
        fingers=_PINKY_FIVE_FINGER_GROUPS,
    ),
    "multi_rm_75_with_rohand": _per_finger_cfg(
        robot_key="multi_rm_75_with_rohand",
        dataset_key="rm_75+rohand",
        suffix="_distal",
        fingers=_ROHAND_FIVE_FINGER_GROUPS,
    ),
}
