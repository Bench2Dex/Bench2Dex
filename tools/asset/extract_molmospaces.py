#!/usr/bin/env python3
"""Extract MolmoSpaces scenes and objects from packed shards.

Extracts tar.zst archives from the MolmoSpaces shard format into a flat
directory structure that preserves the relative USD references between
scenes and objects.

Expected input layout:
    <root>/isaac/scenes/ithor/<date>/shards/00000.tar
    <root>/isaac/objects/thor/<date>/shards/00000.tar

Output layout:
    <root>/isaac/scenes/ithor/FloorPlan1_physics/scene.usda
    <root>/isaac/scenes/ithor/FloorPlan1_physics/Payload/...
    <root>/isaac/objects/thor/Apple_1_mesh/Apple_1_mesh.usda
    <root>/isaac/objects/thor/Apple_1_mesh/Textures/...

Scene USD files reference objects via ../../../../objects/thor/... which
resolves correctly with this layout.

Usage:
    python extract_molmospaces.py /path/to/molmo_isaac_data/isaac

    # Scenes only
    python extract_molmospaces.py /path/to/molmo_isaac_data/isaac --scenes-only

    # Objects only
    python extract_molmospaces.py /path/to/molmo_isaac_data/isaac --objects-only

    # Filter kitchen scenes only (FloorPlan 1-30)
    python extract_molmospaces.py /path/to/molmo_isaac_data/isaac --room-types kitchen
"""

import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile

import zstandard


ROOM_TYPE_RANGES = {
    "kitchen": (1, 30),
    "living": (201, 230),
    "bedroom": (301, 330),
    "bathroom": (401, 430),
}


def _find_shard(base_dir: str) -> str:
    pattern = os.path.join(base_dir, "*/shards/00000.tar")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No shard found at {pattern}")
    return matches[0]


def _extract_zst_tar(zst_data: bytes, out_dir: str) -> list[str]:
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(io.BytesIO(zst_data))
    out_real = os.path.realpath(out_dir)
    extracted = []
    with tarfile.open(fileobj=reader, mode="r|") as inner:
        for member in inner:
            resolved = os.path.realpath(os.path.join(out_dir, member.name))
            if not resolved.startswith(out_real + os.sep) and resolved != out_real:
                raise ValueError(f"Path traversal detected: {member.name}")
            inner.extract(member, out_dir)
            extracted.append(member.name)
    return extracted


def _scene_matches_room_types(name: str, room_types: list[str]) -> bool:
    m = re.search(r"FloorPlan(\d+)", name)
    if not m:
        return False
    num = int(m.group(1))
    for rt in room_types:
        lo, hi = ROOM_TYPE_RANGES[rt]
        if lo <= num <= hi:
            return True
    return False


def extract_scenes(isaac_root: str, room_types: list[str] | None = None) -> int:
    scenes_dir = os.path.join(isaac_root, "scenes", "ithor")
    shard_path = _find_shard(scenes_dir)
    print(f"Scenes shard: {shard_path}")

    with open(os.path.join(os.path.dirname(os.path.dirname(shard_path)),
                           "arrow_table.json")) as f:
        index = json.load(f)
    print(f"  {len(index)} scenes in index")

    count = 0
    with tarfile.open(shard_path) as shard:
        for member in shard.getmembers():
            if not member.name.endswith(".tar.zst"):
                continue
            if room_types and not _scene_matches_room_types(member.name, room_types):
                continue

            scene_name = member.name.replace("ithor_", "").replace(".tar.zst", "")
            dest = os.path.join(scenes_dir, scene_name)
            if os.path.isdir(dest):
                print(f"  skip {scene_name} (exists)")
                continue

            zst_data = shard.extractfile(member).read()
            files = _extract_zst_tar(zst_data, scenes_dir)
            count += 1
            print(f"  [{count}] {scene_name} ({len(files)} files)")

    return count


def extract_objects(isaac_root: str) -> int:
    objects_dir = os.path.join(isaac_root, "objects", "thor")
    shard_path = _find_shard(objects_dir)
    print(f"Objects shard: {shard_path}")

    with open(os.path.join(os.path.dirname(os.path.dirname(shard_path)),
                           "arrow_table.json")) as f:
        index = json.load(f)
    print(f"  {len(index)} object archives in index")

    count = 0
    with tarfile.open(shard_path) as shard:
        for member in shard.getmembers():
            if not member.name.endswith(".tar.zst"):
                continue

            zst_data = shard.extractfile(member).read()
            files = _extract_zst_tar(zst_data, objects_dir)
            count += 1
            cat_name = member.name.replace("thor_", "").replace(".tar.zst", "")
            n_dirs = len({f.split("/")[0] for f in files if "/" in f})
            print(f"  [{count}] {cat_name} ({n_dirs} variants, {len(files)} files)")

    return count


def main():
    parser = argparse.ArgumentParser(description="Extract MolmoSpaces assets")
    parser.add_argument("isaac_root", help="Path to <molmo_data>/isaac/")
    parser.add_argument("--scenes-only", action="store_true")
    parser.add_argument("--objects-only", action="store_true")
    parser.add_argument("--room-types", nargs="+",
                        choices=list(ROOM_TYPE_RANGES.keys()),
                        help="Filter scene room types")
    args = parser.parse_args()

    if not os.path.isdir(args.isaac_root):
        print(f"Error: {args.isaac_root} not found", file=sys.stderr)
        sys.exit(1)

    if not args.objects_only:
        n = extract_scenes(args.isaac_root, args.room_types)
        print(f"Extracted {n} scenes\n")

    if not args.scenes_only:
        n = extract_objects(args.isaac_root)
        print(f"Extracted {n} object archives")


if __name__ == "__main__":
    main()
