"""Load the product spec and algorithm parameters from a YAML or JSON config file."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "nmc_nomapack_u_0-24-77.yaml"


@dataclass
class SpecLimit:
    nominal: float
    min: float
    max: float

    def contains(self, value: float) -> bool:
        return self.min <= value <= self.max


@dataclass
class SegmentationParams:
    hue: tuple[int, int] = (90, 130)
    sat_min: int = 80
    val_min: int = 90
    cleanup_kernel_mm: float = 0.5
    bridge_close_mm: tuple[float, ...] = (1.0, 2.0, 4.0, 7.0, 12.0)
    bridge_work_px_per_mm: float = 5.0
    min_cavity_area_frac: float = 0.02


@dataclass
class MeasurementParams:
    contour_step_mm: float = 0.1
    base_thickness_band_mm: float = 1.0
    width_profile_step_mm: float = 5.0


@dataclass
class AlignmentParams:
    canvas_res_px_per_mm: float = 10.0
    coarse_res_px_per_mm: float = 2.5
    canvas_size_mm: float = 130.0
    cavity_weight: float = 0.5


@dataclass
class ReferenceParams:
    mean_iterations: int = 2
    mask_threshold: float = 0.5


@dataclass
class Config:
    product: str
    spec: dict[str, SpecLimit]
    contour_tolerance_mm: float
    report_only: list[str] = field(default_factory=list)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    measurement: MeasurementParams = field(default_factory=MeasurementParams)
    alignment: AlignmentParams = field(default_factory=AlignmentParams)
    reference: ReferenceParams = field(default_factory=ReferenceParams)
    source: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _section(cls, raw: dict | None):
    raw = dict(raw or {})
    unknown = set(raw) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys in config: {sorted(unknown)}")
    for k, v in raw.items():
        if isinstance(v, list):
            raw[k] = tuple(v)
    return cls(**raw)


def config_from_dict(raw: dict, source: str = "") -> Config:
    if "px_per_mm" in raw:
        raise ValueError("px_per_mm does not belong in the product config; pass --px-per-mm or --calibration")
    for key in ("product", "spec", "contour_tolerance_mm"):
        if key not in raw:
            raise ValueError(f"Config is missing required key '{key}'")
    spec = {name: SpecLimit(**lim) for name, lim in raw["spec"].items()}
    for name, lim in spec.items():
        if lim.min > lim.max:
            raise ValueError(f"Spec '{name}': min {lim.min} > max {lim.max}")
    return Config(
        product=raw["product"],
        spec=spec,
        contour_tolerance_mm=float(raw["contour_tolerance_mm"]),
        report_only=list(raw.get("report_only", [])),
        segmentation=_section(SegmentationParams, raw.get("segmentation")),
        measurement=_section(MeasurementParams, raw.get("measurement")),
        alignment=_section(AlignmentParams, raw.get("alignment")),
        reference=_section(ReferenceParams, raw.get("reference")),
        source=source,
    )


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    text = path.read_text()
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    return config_from_dict(raw, source=str(path))
