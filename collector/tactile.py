"""Tactile site helpers shared by TacMap collection and asset generation."""

from __future__ import annotations


SUPPORTED_TACTILE_ROBOT_KEYS = [
    "multi_ur5_rh56dfx_with_flange",
    "multi_ur5_shadow_hand_with_flange",
    "multi_ur5_schunk_hand_with_flange",
    "multi_ur5_wuji_with_flange",
    "multi_panda_with_allegro",
    "multi_panda_with_orca",
    "multi_xarm7_with_ability",
    "multi_xarm7_with_leap",
    "multi_iiwa7_with_sharpa",
    "multi_jaka_zu7_dexhand021_with_flange",
    "multi_ur5_rh5dg2_with_flange",
    "multi_rm_65_with_revo2",
    "multi_rm_75_with_rohand",
]

_DEFAULT_ROBOT_PRIM_PATH = "/World/Objects/GlobalRobot"


def _two_hand_sites(per_hand_sites: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{side}_{site}" for side in ("right", "left") for site in per_hand_sites)


RH56DFX_HAND_SITE_NAMES = _two_hand_sites(
    ("thumb_pad", "index_pad", "middle_pad", "ring_pad", "little_pad", "palm")
)

SHADOW_HAND_SITE_NAMES = _two_hand_sites(
    ("FFJ1", "MFJ1", "RFJ1", "LFJ1", "THJ1", "palm")
)

SCHUNK_HAND_SITE_NAMES = _two_hand_sites(
    (
        "Thumb_Flexion",
        "Index_Finger_Distal",
        "Middle_Finger_Distal",
        "Ring_Finger",
        "Pinky",
        "palm",
    )
)

WUJI_HAND_SITE_NAMES = _two_hand_sites(
    ("thumb_J4", "index_J4", "middle_J4", "ring_J4", "little_J4", "palm")
)

ALLEGRO_HAND_SITE_NAMES = _two_hand_sites(
    ("thumb_3", "index_3", "middle_3", "ring_3", "palm")
)

ORCA_HAND_SITE_NAMES = _two_hand_sites(
    ("thumb_dip", "index_pip", "middle_pip", "ring_pip", "pinky_pip", "palm")
)

ABILITY_HAND_SITE_NAMES = _two_hand_sites(
    ("index_q2", "middle_q2", "ring_q2", "pinky_q2", "thumb_q2", "palm")
)

LEAP_HAND_SITE_NAMES = _two_hand_sites(
    ("index_2", "middle_2", "ring_2", "thumb_2", "palm")
)

SHARPA_HAND_SITE_NAMES = _two_hand_sites(
    (
        "thumb_elastomer",
        "index_elastomer",
        "middle_elastomer",
        "ring_elastomer",
        "pinky_elastomer",
        "palm",
    )
)

DEXHAND021_SITE_NAMES = _two_hand_sites(
    ("thumb_J4", "index_J4", "middle_J4", "ring_J4", "little_J4", "palm")
)

RH5DG2_HAND_SITE_NAMES = _two_hand_sites(
    ("thumb_pad", "index_pad", "middle_pad", "ring_pad", "little_pad", "palm")
)

REVO2_HAND_SITE_NAMES = _two_hand_sites(
    ("thumb_touch", "index_touch", "middle_touch", "ring_touch", "pinky_touch", "palm")
)

ROHAND_HAND_SITE_NAMES = _two_hand_sites(
    ("th_distal", "if_distal", "mf_distal", "rf_distal", "lf_distal", "palm")
)

ROBOT_KEY_TO_SITE_NAMES = {
    "multi_ur5_rh56dfx_with_flange": RH56DFX_HAND_SITE_NAMES,
    "multi_ur5_shadow_hand_with_flange": SHADOW_HAND_SITE_NAMES,
    "multi_ur5_schunk_hand_with_flange": SCHUNK_HAND_SITE_NAMES,
    "multi_ur5_wuji_with_flange": WUJI_HAND_SITE_NAMES,
    "multi_panda_with_allegro": ALLEGRO_HAND_SITE_NAMES,
    "multi_panda_with_orca": ORCA_HAND_SITE_NAMES,
    "multi_xarm7_with_ability": ABILITY_HAND_SITE_NAMES,
    "multi_xarm7_with_leap": LEAP_HAND_SITE_NAMES,
    "multi_iiwa7_with_sharpa": SHARPA_HAND_SITE_NAMES,
    "multi_jaka_zu7_dexhand021_with_flange": DEXHAND021_SITE_NAMES,
    "multi_ur5_rh5dg2_with_flange": RH5DG2_HAND_SITE_NAMES,
    "multi_rm_65_with_revo2": REVO2_HAND_SITE_NAMES,
    "multi_rm_75_with_rohand": ROHAND_HAND_SITE_NAMES,
}


def _shadow_hand_aliases() -> dict[str, tuple[str, ...]]:
    joint_to_link = {
        "FFJ1": "ffdistal",
        "FFJ2": "ffmiddle",
        "FFJ3": "ffproximal",
        "FFJ4": "ffknuckle",
        "MFJ1": "mfdistal",
        "MFJ2": "mfmiddle",
        "MFJ3": "mfproximal",
        "MFJ4": "mfknuckle",
        "RFJ1": "rfdistal",
        "RFJ2": "rfmiddle",
        "RFJ3": "rfproximal",
        "RFJ4": "rfknuckle",
        "LFJ1": "lfdistal",
        "LFJ2": "lfmiddle",
        "LFJ3": "lfproximal",
        "LFJ4": "lfknuckle",
        "LFJ5": "lfmetacarpal",
        "THJ1": "thdistal",
        "THJ2": "thmiddle",
        "THJ3": "thhub",
        "THJ4": "thproximal",
        "THJ5": "thbase",
    }
    aliases: dict[str, tuple[str, ...]] = {
        "right_palm": ("palm",),
        "left_palm": ("l_palm",),
    }
    for joint_name, link_name in joint_to_link.items():
        aliases[f"right_{joint_name}"] = (link_name,)
        aliases[f"left_{joint_name}"] = (f"l_{link_name}",)
    return aliases


def _wuji_aliases() -> dict[str, tuple[str, ...]]:
    finger_ids = {
        "thumb": "finger1",
        "index": "finger2",
        "middle": "finger3",
        "ring": "finger4",
        "little": "finger5",
    }
    aliases = {
        f"{side}_{finger}_J{joint}": (f"{side}_{finger_id}_link{joint}",)
        for side in ("right", "left")
        for finger, finger_id in finger_ids.items()
        for joint in range(1, 5)
    }
    aliases["right_palm"] = ("right_palm_link",)
    aliases["left_palm"] = ("left_palm_link",)
    return aliases


def _allegro_aliases() -> dict[str, tuple[str, ...]]:
    link_ids = {
        "index": ("0.0", "1.0", "2.0", "3.0"),
        "middle": ("4.0", "5.0", "6.0", "7.0"),
        "ring": ("8.0", "9.0", "10.0", "11.0"),
        "thumb": ("12", "13.0", "14.0", "15.0"),
    }
    aliases: dict[str, tuple[str, ...]] = {}
    for side, prefix in (("left", ""), ("right", "multi_")):
        for finger, ids in link_ids.items():
            for joint, link_id in enumerate(ids):
                aliases[f"{side}_{finger}_{joint}"] = (f"{prefix}link_{link_id}",)
    aliases["right_palm"] = ("multi_base_link",)
    aliases["left_palm"] = ("base_link",)
    return aliases


def _orca_aliases() -> dict[str, tuple[str, ...]]:
    joint_to_visual_link = {
        "thumb_mcp": "thumb_mp",
        "thumb_abd": "thumb_pp",
        "thumb_pip": "thumb_ip",
        "thumb_dip": "thumb_dp",
        "index_abd": "index_mp",
        "index_mcp": "index_pp",
        "index_pip": "index_ip",
        "middle_abd": "middle_mp",
        "middle_mcp": "middle_pp",
        "middle_pip": "middle_ip",
        "ring_abd": "ring_mp",
        "ring_mcp": "ring_pp",
        "ring_pip": "ring_ip",
        "pinky_abd": "pinky_mp",
        "pinky_mcp": "pinky_pp",
        "pinky_pip": "pinky_ip",
    }
    aliases: dict[str, tuple[str, ...]] = {}
    for side, prefix in (("left", "left_"), ("right", "multi_right_")):
        for joint_name, visual_link in joint_to_visual_link.items():
            aliases[f"{side}_{joint_name}"] = (f"{prefix}{visual_link}",)
    aliases["right_palm"] = ("multi_right_palm",)
    aliases["left_palm"] = ("left_palm",)
    return aliases


def _ability_aliases() -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for side, prefix in (("left", ""), ("right", "multi_")):
        for finger in ("index", "middle", "ring", "pinky", "thumb"):
            aliases[f"{side}_{finger}_q1"] = (f"{prefix}{finger}_L1",)
            aliases[f"{side}_{finger}_q2"] = (f"{prefix}{finger}_L2",)
    aliases["right_palm"] = ("multi_base",)
    aliases["left_palm"] = ("base",)
    return aliases


def _leap_aliases() -> dict[str, tuple[str, ...]]:
    finger_links = {
        "index": ("mcp_joint", "fingertip"),
        "middle": ("mcp_joint_2", "fingertip_2"),
        "ring": ("mcp_joint_3", "fingertip_3"),
        "thumb": ("thumb_pip", "thumb_fingertip"),
    }
    aliases: dict[str, tuple[str, ...]] = {}
    for side, prefix in (("left", "leap_l_"), ("right", "multi_leap_r_")):
        for finger, names in finger_links.items():
            aliases[f"{side}_{finger}_1"] = (f"{prefix}{names[0]}",)
            aliases[f"{side}_{finger}_2"] = (f"{prefix}{names[1]}",)
    aliases["right_palm"] = ("multi_leap_r_palm_lower",)
    aliases["left_palm"] = ("leap_l_palm_lower_left",)
    return aliases


def _schunk_aliases() -> dict[str, tuple[str, ...]]:
    child_links = {
        "Thumb_Flexion": "hand_a",
        "Thumb_Opposition": "hand_z",
        "Index_Finger_Proximal": "hand_l",
        "Index_Finger_Distal": "hand_p",
        "Middle_Finger_Proximal": "hand_k",
        "Middle_Finger_Distal": "hand_o",
        "Ring_Finger": "hand_j",
        "Pinky": "hand_i",
    }
    aliases: dict[str, tuple[str, ...]] = {}
    for side in ("right", "left"):
        for site, child_link in child_links.items():
            aliases[f"{side}_{site}"] = (f"{side}_{child_link}",)
        aliases[f"{side}_palm"] = (f"{side}_hand_e1",)
    return aliases


def _sharpa_aliases() -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for side, prefix in (("left", "left_"), ("right", "multi_right_")):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for segment in ("elastomer", "DP"):
                aliases[f"{side}_{finger}_{segment}"] = (f"{prefix}{finger}_{segment}",)
    aliases["right_palm"] = ("multi_right_hand_C_MC",)
    aliases["left_palm"] = ("left_hand_C_MC",)
    return aliases


def _dexhand021_aliases() -> dict[str, tuple[str, ...]]:
    finger_ids = {
        "thumb": "1",
        "index": "2",
        "middle": "3",
        "ring": "4",
        "little": "5",
    }
    aliases: dict[str, tuple[str, ...]] = {}
    for side, prefix in (("right", "r"), ("left", "l")):
        for finger, finger_id in finger_ids.items():
            for joint in range(1, 5):
                body_names = [f"{prefix}_f_link{finger_id}_{joint}"]
                if joint == 4:
                    body_names.append(f"{prefix}_f_link{finger_id}_pad")
                aliases[f"{side}_{finger}_J{joint}"] = tuple(body_names)
    aliases["right_palm"] = ("r_p_link0",)
    aliases["left_palm"] = ("l_p_link0",)
    return aliases


def _rh5dg2_aliases() -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for side in ("right", "left"):
        for finger in ("thumb", "index", "middle", "ring"):
            aliases[f"{side}_{finger}_pad"] = (f"{side}_{finger}_force_sensor",)
        aliases[f"{side}_little_pad"] = (f"{side}_pinky_force_sensor",)
        aliases[f"{side}_little_dip"] = (f"{side}_pinky_dip",)
        aliases[f"{side}_little_pip"] = (f"{side}_pinky_pip",)
        aliases[f"{side}_palm"] = (f"{side}_plam_force_sensor", f"{side}_hand_base")
    return aliases


def _rh56dfx_aliases() -> dict[str, tuple[str, ...]]:
    return {
        "right_palm": ("plam",),
        "left_palm": ("l_plam",),
    }


def _revo2_aliases() -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for side in ("right", "left"):
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            aliases[f"{side}_{finger}_touch"] = (f"{side}_{finger}_touch_link",)
        aliases[f"{side}_palm"] = (f"{side}_base_link",)
    return aliases


def _rohand_aliases() -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for site_side, prefix in (("right", "right_"), ("left", "multi_left_")):
        for finger in ("th", "if", "mf", "rf", "lf"):
            aliases[f"{site_side}_{finger}_distal"] = (f"{prefix}{finger}_distal_link",)
        aliases[f"{site_side}_palm"] = (f"{prefix}base_link",)
    return aliases


_TACTILE_SITE_BODY_ALIASES = {
    "multi_ur5_rh56dfx_with_flange": _rh56dfx_aliases(),
    "multi_ur5_shadow_hand_with_flange": _shadow_hand_aliases(),
    "multi_ur5_schunk_hand_with_flange": _schunk_aliases(),
    "multi_ur5_wuji_with_flange": _wuji_aliases(),
    "multi_panda_with_allegro": _allegro_aliases(),
    "multi_panda_with_orca": _orca_aliases(),
    "multi_xarm7_with_ability": _ability_aliases(),
    "multi_xarm7_with_leap": _leap_aliases(),
    "multi_iiwa7_with_sharpa": _sharpa_aliases(),
    "multi_jaka_zu7_dexhand021_with_flange": _dexhand021_aliases(),
    "multi_ur5_rh5dg2_with_flange": _rh5dg2_aliases(),
    "multi_rm_65_with_revo2": _revo2_aliases(),
    "multi_rm_75_with_rohand": _rohand_aliases(),
}


def _sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))


def _site_body_candidates(robot_key: str, site_name: str) -> tuple[str, ...]:
    aliases = _TACTILE_SITE_BODY_ALIASES.get(str(robot_key), {}).get(str(site_name), ())
    if isinstance(aliases, str):
        aliases = (aliases,)
    candidates = (str(site_name), *tuple(str(alias) for alias in aliases))
    return tuple(dict.fromkeys(candidates))


def _stage_has_site_body_prim(robot_prim_path: str, body_name: str) -> bool | None:
    try:
        import omni.usd
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None
    prim_path = f"{str(robot_prim_path).rstrip('/')}/{_sanitize_name(body_name)}"
    return bool(stage.GetPrimAtPath(prim_path).IsValid())


def _resolve_site_body_name(robot_key: str, site_name: str, robot_prim_path: str | None = None) -> str:
    candidates = _site_body_candidates(robot_key, site_name)
    if robot_prim_path:
        saw_stage = False
        for body_name in candidates:
            exists = _stage_has_site_body_prim(robot_prim_path, body_name)
            if exists is None:
                continue
            saw_stage = True
            if exists:
                return body_name
        if saw_stage:
            return candidates[-1]
    return candidates[1] if len(candidates) > 1 else candidates[0]


def _body_name_in_articulation(body_name: str, body_names: list[str]) -> bool:
    safe_body_name = _sanitize_name(body_name)
    return body_name in body_names or safe_body_name in body_names


def attach_tactile_to_robot_state(
    robot_state: dict[str, object] | None,
    tactile_payload: dict[str, object],
) -> dict[str, object]:
    payload = {} if robot_state is None else dict(robot_state)
    payload["tactile"] = tactile_payload
    return payload
