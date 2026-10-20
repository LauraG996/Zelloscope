"""Build, save and load the golden reference shape from known-good parts."""
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .geometry import Canvas, align, canonicalize, resample
from .measure import measure
from .pipeline import Part, inspect_part

FORMAT_VERSION = 1


@dataclass
class Reference:
    outer: np.ndarray                    # mean outer contour, canonical frame, mm
    cavity: np.ndarray                   # mean cavity contour, canonical frame, mm
    parts: list[Part] = field(default_factory=list)   # the canonical sample shapes it was built from
    meta: dict = field(default_factory=dict)

    def masks(self, canvas: Canvas) -> tuple[np.ndarray, np.ndarray]:
        return canvas.render(self.outer), canvas.render(self.cavity)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas = Canvas(self.meta["canvas_res_px_per_mm"], self.meta["canvas_size_mm"])
        om, cm = self.masks(canvas)
        # Sample shapes are stored too, so the reference can be extended later without the original images.
        np.savez_compressed(
            path, format_version=FORMAT_VERSION, outer=self.outer, cavity=self.cavity,
            outer_mask=om, cavity_mask=cm,
            part_outer=np.concatenate([p.outer for p in self.parts]),
            part_outer_len=[len(p.outer) for p in self.parts],
            part_cavity=np.concatenate([p.cavity for p in self.parts]),
            part_cavity_len=[len(p.cavity) for p in self.parts],
            part_names=np.array([p.name for p in self.parts]),
            meta=json.dumps(self.meta))

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        with np.load(path, allow_pickle=False) as z:
            if int(z["format_version"]) != FORMAT_VERSION:
                raise ValueError(f"{path}: unsupported reference format {int(z['format_version'])}")
            split = lambda arr, lens: np.split(arr, np.cumsum(lens)[:-1])
            parts = [Part(str(n), o, c) for n, o, c in zip(
                z["part_names"], split(z["part_outer"], z["part_outer_len"]),
                split(z["part_cavity"], z["part_cavity_len"]))]
            return cls(z["outer"], z["cavity"], parts, json.loads(str(z["meta"])))


def mean_shape(parts: list[Part], cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """Align all to the first part, average the binary masks, re-align to the mean, average again, ..."""
    canvas = Canvas(cfg.alignment.canvas_res_px_per_mm, cfg.alignment.canvas_size_mm)
    step = cfg.measurement.contour_step_mm
    ref_o, ref_c = parts[0].outer, parts[0].cavity
    for _ in range(max(1, cfg.reference.mean_iterations)):
        acc_o = np.zeros((canvas.n, canvas.n), np.float32)
        acc_c = np.zeros_like(acc_o)
        for p in parts:
            al = align(p.outer, p.cavity, ref_o, ref_c, cfg.alignment)
            acc_o += canvas.render(al.apply(p.outer))
            acc_c += canvas.render(al.apply(p.cavity))
        thr = cfg.reference.mask_threshold
        ref_o = canvas.contour_mm(acc_o / len(parts) >= thr)
        ref_c = canvas.contour_mm(acc_c / len(parts) >= thr)
        # Re-centre on the mean shape's own principal axis so the reference frame does not inherit the
        # first part's tilt (measurements are taken in this frame).
        ref_o, ref_c = canonicalize(ref_o, ref_c)
        ref_o, ref_c = resample(ref_o, step), resample(ref_c, step)
    return ref_o, ref_c


def leave_one_out(parts: list[Part], cfg: Config, contour_tol: float | None = None) -> list[dict]:
    """Build from N-1 parts, inspect the left-out one."""
    out = []
    for i, p in enumerate(parts):
        ro, rc = mean_shape([q for j, q in enumerate(parts) if j != i], cfg)
        res = inspect_part(p, ro, rc, cfg, contour_tol)
        out.append(dict(name=p.name, PASS=res.passed, reasons=res.reasons, deviation=res.data["deviation"],
                        measurements={k: res.data["measurements"][k] for k in cfg.spec}))
    return out


def build_reference(parts: list[Part], cfg: Config, px_per_mm: float | None = None,
                    contour_tol: float | None = None, run_loo: bool = True) -> tuple[Reference, dict]:
    """Returns the reference and a report with per-sample results and leave-one-out results."""
    if not parts:
        raise ValueError("need at least one good part")
    ref_o, ref_c = mean_shape(parts, cfg)
    samples = []
    for p in parts:
        res = inspect_part(p, ref_o, ref_c, cfg, contour_tol)
        spec_ok = all(v["ok"] for v in res.data["spec"].values())
        samples.append(dict(name=p.name, spec_ok=spec_ok, contour_ok=res.data["contour_ok"],
                            measurements=res.data["measurements"], spec=res.data["spec"],
                            deviation=res.data["deviation"], mouth_bridged=res.data["mouth_bridged"]))
    loo = leave_one_out(parts, cfg, contour_tol) if run_loo and len(parts) >= 3 else []

    meta = dict(product=cfg.product, created=dt.datetime.now().isoformat(timespec="seconds"),
                n_parts=len(parts), part_names=[p.name for p in parts],
                px_per_mm_at_build=px_per_mm, config_source=cfg.source,
                canvas_res_px_per_mm=cfg.alignment.canvas_res_px_per_mm,
                canvas_size_mm=cfg.alignment.canvas_size_mm,
                reference_measurements=measure(ref_o, ref_c, cfg.measurement))
    report = dict(samples=samples, leave_one_out=loo, reference=meta)
    meta = dict(meta, samples=samples, leave_one_out=loo)
    return Reference(ref_o, ref_c, list(parts), meta), report
