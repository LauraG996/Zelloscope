import json

import pytest
import yaml

from profile_check.config import DEFAULT_CONFIG, config_from_dict, load_config


def test_default_config_has_nmc_spec():
    cfg = load_config()
    assert cfg.spec["inner_width_max"].min == 24.0 and cfg.spec["inner_width_max"].max == 27.0
    assert cfg.spec["base_thickness"].min == 11.0 and cfg.spec["base_thickness"].max == 13.0
    assert cfg.spec["length"].min == 76.0 and cfg.spec["length"].max == 80.0
    assert cfg.contour_tolerance_mm == 1.0


def test_json_config_equivalent(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(raw))
    assert load_config(p).spec == load_config().spec


def test_scale_is_not_allowed_in_config():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    with pytest.raises(ValueError, match="px_per_mm"):
        config_from_dict(dict(raw, px_per_mm=20.0))


def test_unknown_parameter_rejected():
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["segmentation"]["hue_minn"] = 3
    with pytest.raises(ValueError, match="hue_minn"):
        config_from_dict(raw)
