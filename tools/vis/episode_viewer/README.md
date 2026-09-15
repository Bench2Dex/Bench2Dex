# HDF5 Episode Viewer

Standalone Flask viewer for inspecting one HDF5 episode file or a directory of
episode files. The viewer does
not import Dex2Bench project modules; all HDF5 parsing, depth rendering,
occupancy projection, tactile summaries, and 3D box projection are implemented
inside this directory.

## Install

```bash
pip install -r tools/vis/episode_viewer/requirements.txt
```

## Run

```bash
python tools/vis/episode_viewer/app.py \
  --hdf5 /path/to/episode.hdf5 \
  --host 0.0.0.0 \
  --port 8080
```

You can also pass a directory. The toolbar shows a small browser for the
current directory. Open subdirectories one level at a time, select any `.hdf5`
file to view it, or use the dropdown to switch between `.hdf5` files in the
current directory:

```bash
python tools/vis/episode_viewer/app.py \
  --hdf5 /path/to/episode_dir \
  --host 0.0.0.0 \
  --port 8080
```

Deleting from the UI does not permanently delete data. Files and folders are
moved into:

```text
tools/vis/episode_viewer/deleted_items/
```

Moved items get timestamped names to avoid collisions. The browser does not
allow deleting the browse root or the `deleted_items` folder itself.

The expected HDF5 paths are discovered lazily. Missing modalities render as
placeholders instead of crashing.

The Occupancy tab reads `labels/occupancy_gt`, `labels/occupancy_tsdf`, or
`labels/occupancy` and overlays occupied voxels on each camera view.
