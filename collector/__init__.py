"""Data collection package for dex2scene."""

__all__ = ["CollectConfig", "DataCollector", "load_collect_config", "load_collect_scene_background"]


def __getattr__(name):
    if name in {"CollectConfig", "load_collect_config", "load_collect_scene_background"}:
        from .config import CollectConfig, load_collect_config, load_collect_scene_background

        exports = {
            "CollectConfig": CollectConfig,
            "load_collect_config": load_collect_config,
            "load_collect_scene_background": load_collect_scene_background,
        }
        return exports[name]
    if name == "DataCollector":
        from .data_collector import DataCollector

        return DataCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
