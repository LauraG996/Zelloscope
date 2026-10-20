"""Image -> canonical part, and part + reference -> PASS/FAIL result."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config
from .deviation import contour_deviation, worst_p99
from .geometry import align, canonicalize, resample
from .measure import measure, spec_check
from .segmentation import segment

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Part:
    name: str
    outer: np.ndarray        # canonical frame, mm
    cavity: np.ndarray       # canonical frame, mm
    bridge_mm: float = 0.0
    source: str = ""


@dataclass
class InspectionResult:
    passed: bool
    reasons: list[str]
    data: dict                                   # everything that goes into results.json
    outer: np.ndarray = field(repr=False)        # aligned part contours (mm, reference frame)
    cavity: np.ndarray = field(repr=False)
    deviations: dict = field(repr=False)         # raw signed distances per contour


def expand_images(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        else:
            out.append(p)
    return out


def part_from_image(img: np.ndarray, px_per_mm: float, cfg: Config, name: str = "", source: str = "") -> Part:
    if not px_per_mm or px_per_mm <= 0:
        raise ValueError("px_per_mm must be given explicitly (from a calibration) and be > 0")
    seg = segment(img, cfg.segmentation, px_per_mm)
    outer, cavity = canonicalize(seg.outer / px_per_mm, seg.cavity / px_per_mm)
    step = cfg.measurement.contour_step_mm
    return Part(name, resample(outer, step), resample(cavity, step), seg.bridge_mm, source)


def load_part(path: str | Path, px_per_mm: float, cfg: Config) -> Part:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return part_from_image(img, px_per_mm, cfg, name=path.stem, source=str(path))


def inspect_part(part: Part, ref_outer: np.ndarray, ref_cavity: np.ndarray, cfg: Config,
                 contour_tol: float | None = None) -> InspectionResult:
    tol = cfg.contour_tolerance_mm if contour_tol is None else contour_tol
    al = align(part.outer, part.cavity, ref_outer, ref_cavity, cfg.alignment)
    outer, cavity = al.apply(part.outer), al.apply(part.cavity)
    m = measure(outer, cavity, cfg.measurement)
    spec = spec_check(m, cfg)
    dev, raw = contour_deviation(outer, cavity, ref_outer, ref_cavity)

    reasons = [f"{k}={v['value']} outside [{v['min']}, {v['max']}]" for k, v in spec.items() if not v["ok"]]
    reasons += [f"{k} contour p99 deviation {s['p99_abs']} mm > {tol} mm" for k, s in dev.items()
                if s["p99_abs"] > tol]
    contour_ok = worst_p99(dev) <= tol
    passed = not reasons
    data = dict(image=part.source or part.name, PASS=passed, reasons=reasons, spec=spec,
                contour_ok=contour_ok, contour_tol_mm=tol, deviation=dev, measurements=m,
                report_only={k: m[k] for k in cfg.report_only if k in m},
                alignment=al.to_dict(), mouth_bridged=part.bridge_mm > 0, bridge_close_mm=part.bridge_mm)
    return InspectionResult(passed, reasons, data, outer, cavity, raw)
