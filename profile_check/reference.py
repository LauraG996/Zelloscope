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


def target_dimensions(cfg: Config, mode: str = "nominal") -> dict[str, float]:
    """Spec dimensions to shape the ideal reference to: the drawing's nominal values or the middle of each
    accept range (these differ when the tolerance is one-sided, e.g. width 24 -0/+3)."""
    if mode == "nominal":
        return {k: lim.nominal for k, lim in cfg.spec.items()}
    if mode == "mid":
        return {k: (lim.min + lim.max) / 2 for k, lim in cfg.spec.items()}
    raise ValueError(f"unknown target mode '{mode}' (nominal or mid)")


def _reshape(outer: np.ndarray, cavity: np.ndarray, m: dict, t: dict) -> tuple[np.ndarray, np.ndarray]:
    """One correction step towards the target inner width, base thickness and length (canonical frame,
    y down, base at the bottom).

    Width: only the cavity is scaled across the axis, so the outer outline is kept and the walls take up the
    difference. Thickness / length: a piecewise-linear stretch along the axis, the base zone (cavity bottom ->
    outer base) and the part above it separately, applied to both contours.
    """
    o, c = outer.copy(), cavity.copy()
    if "inner_width_max" in t:
        cl, cr = c[:, 0].min(), c[:, 0].max()
        xc = (cl + cr) / 2
        c[:, 0] = xc + (c[:, 0] - xc) * t["inner_width_max"] / m["inner_width_max"]

    y_bot, y_top, y_cb = o[:, 1].max(), o[:, 1].min(), c[:, 1].max()
    base = y_bot - y_cb
    k_t = t.get("base_thickness", m["base_thickness"]) / m["base_thickness"]
    new_base = base * k_t
    new_len = t.get("length", m["length"])
    k_l = (new_len - new_base) / (y_cb - y_top)

    def ymap(y):
        return np.where(y >= y_cb, y_bot - (y_bot - y) * k_t, y_bot - new_base - (y_cb - y) * k_l)

    o[:, 1], c[:, 1] = ymap(o[:, 1]), ymap(c[:, 1])
    return o, c


def ideal_shape(outer: np.ndarray, cavity: np.ndarray, targets: dict[str, float], cfg: Config,
                iterations: int = 5, tol_mm: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Reshape a (mean) reference so its spec measurements equal `targets`.

    The drawing only fixes these dimensions; everything else about the outline (outer width, leg lean,
    corner and tip shape) is kept from the input shape. Iterates because the measurement definitions
    (widest row, median over a band) do not map exactly onto the stretch.
    """
    unknown = set(targets) - {"inner_width_max", "base_thickness", "length"}
    if unknown:
        raise ValueError(f"don't know how to reshape for {sorted(unknown)}")
    o, c = outer, cavity
    for _ in range(iterations):
        m = measure(o, c, cfg.measurement)
        if all(abs(m[k] - v) <= tol_mm for k, v in targets.items()):
            break
        o, c = _reshape(o, c, m, targets)
        o, c = canonicalize(o, c)
    step = cfg.measurement.contour_step_mm
    return resample(o, step), resample(c, step)


def make_ideal(ref: Reference, cfg: Config, targets: dict[str, float], mode: str = "") -> Reference:
    o, c = ideal_shape(ref.outer, ref.cavity, targets, cfg)
    meta = dict(ref.meta, kind="ideal", ideal_targets=targets, ideal_mode=mode,
                source_reference_measurements=ref.meta.get("reference_measurements"),
                reference_measurements=measure(o, c, cfg.measurement),
                created=dt.datetime.now().isoformat(timespec="seconds"))
    return Reference(o, c, list(ref.parts), meta)
