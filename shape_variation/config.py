"""Analysis config: YAML/JSON file merged over defaults, plus the spec from the profile_check config."""
import copy
import json
from pathlib import Path

import yaml

from profile_check.config import Config as SpecConfig
from profile_check.config import load_config as load_spec_config

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "shape_variation.yaml"

DEFAULTS = {
    "product": "",
    "spec_config": None,
    "exclude": [],
    "segmentation": {"method": "otsu", "blur_mm": 0.15, "cleanup_kernel_mm": 0.3, "max_pore_area_mm2": 8.0,
                     "work_px_per_mm": 20.0, "frame_margin_mm": 0.3},
    "window": {"depth_mm": "auto", "auto_margin_mm": 1.0},
    "shape_model": {"n_outer": 200, "n_cavity": 200, "variance_keep": 0.95, "allow_mirror": True,
                    "bootstrap": 200, "mode_sigma": 2.0},
    "thickness": {"raster_px_per_mm": 20.0, "grid_step_mm": 1.0, "end_trim_mm": 10.0},
    "monitoring": {"t2_alpha": 0.01, "spc_sigma": 3.0, "run_length": 8, "trend_length": 6},
    "process_windows": None,
    "cells": {"seed_smooth_mm": 0.2, "seed_min_separation_mm": 0.39, "wall_smooth_mm": 0.02,
              "min_area_mm2": 0.02, "merged_area_factor": 4.0, "map_bin_mm": 2.0,
              "work_px_per_mm": 40.0},
    "relaxation": {"min_images_per_part": 3, "settle_fraction": 0.95},
}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise ValueError(f"unknown config key '{path}{k}'")
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


class AnalysisConfig(dict):
    """Nested dict with the merged settings; `spec` is the profile_check config (limits + measurement params)."""
    spec: SpecConfig
    source: str


def load_config(path: str | Path | None = None) -> AnalysisConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    text = path.read_text()
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    cfg = AnalysisConfig(_merge(DEFAULTS, raw or {}))
    spec_path = cfg["spec_config"]
    cfg.spec = load_spec_config((path.parent / spec_path) if spec_path else None)
    cfg.source = str(path)
    return cfg
