"""Find images and their scale, timestamp and optional per-part metadata.

The scale is never guessed. Per image it comes from (in this order):
  1. the `ZelloscopeData` text chunk Zelloscope writes into its PNGs (base64 JSON with "p2mm" = mm per pixel);
  2. a `scale.yaml` (or `scale.json`) with `mm_per_px` in the image's folder.
Images with neither are returned with `mm_per_px=None` and a reason, and are skipped by the analysis.
"""
import base64
import csv
import datetime as dt
import json
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_ T](\d{2})[-:](\d{2})(?:[-:](\d{2}))?")


@dataclass
class ImageRecord:
    path: Path
    mm_per_px: float | None
    scale_source: str                      # "png-metadata", "<folder>/scale.yaml" or why it is missing
    timestamp: dt.datetime | None
    meta: dict = field(default_factory=dict)  # row from parts.csv, if any

    @property
    def name(self) -> str:
        return self.path.stem


def png_text_chunks(path: Path) -> dict[str, str]:
    """tEXt / iTXt / zTXt chunks of a PNG (empty for other formats or unreadable files)."""
    out = {}
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return out
            while True:
                head = f.read(8)
                if len(head) < 8:
                    break
                n, kind = struct.unpack(">I4s", head)
                if kind == b"IDAT":          # text chunks we care about come before the image data
                    f.seek(n + 4, 1)
                    continue
                data = f.read(n)
                f.seek(4, 1)                 # CRC
                if kind == b"tEXt":
                    k, _, v = data.partition(b"\0")
                    out[k.decode("latin1")] = v.decode("latin1")
                elif kind == b"zTXt":
                    k, _, v = data.partition(b"\0")
                    out[k.decode("latin1")] = zlib.decompress(v[1:]).decode("latin1")
                elif kind == b"iTXt":
                    k, _, rest = data.partition(b"\0")
                    compressed, rest = rest[0], rest[2:]
                    _, _, rest = rest.partition(b"\0")      # language tag
                    _, _, text = rest.partition(b"\0")      # translated keyword
                    out[k.decode("latin1")] = (zlib.decompress(text) if compressed else text).decode("utf-8")
                elif kind == b"IEND":
                    break
    except OSError:
        pass
    return out


def zelloscope_metadata(path: Path) -> dict:
    raw = png_text_chunks(path).get("ZelloscopeData")
    if not raw:
        return {}
    for decode in (lambda s: json.loads(base64.b64decode(s)), json.loads):
        try:
            return decode(raw)
        except (ValueError, TypeError):
            continue
    return {}


def _folder_scale(folder: Path) -> tuple[float | None, str]:
    for name in ("scale.yaml", "scale.yml", "scale.json"):
        p = folder / name
        if p.exists():
            data = json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())
            try:
                v = float(data["mm_per_px"])
            except (TypeError, KeyError, ValueError):
                return None, f"{p} has no valid mm_per_px"
            return (v, str(p)) if v > 0 else (None, f"{p}: mm_per_px must be > 0")
    return None, "no ZelloscopeData scale in the PNG and no scale.yaml in its folder"


def parse_timestamp(path: Path, zmeta: dict) -> dt.datetime | None:
    for key in ("timestamp", "time", "datetime"):
        if key in zmeta:
            try:
                return dt.datetime.fromisoformat(str(zmeta[key]))
            except ValueError:
                pass
    m = _TIME_RE.search(path.stem)
    if not m:
        return None
    d, hh, mm, ss = m.groups()
    try:
        return dt.datetime.fromisoformat(f"{d}T{hh}:{mm}:{ss or '00'}")
    except ValueError:          # digits that look like a time but aren't one
        return None


def _parts_csv(folder: Path) -> dict[str, dict]:
    p = folder / "parts.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {row["filename"]: {k: v for k, v in row.items() if k != "filename" and v not in ("", None)}
                for row in csv.DictReader(f)}


def discover(inputs, exclude: tuple[str, ...] = ()) -> list[ImageRecord]:
    """Image records for files and/or folders (not recursive), sorted by timestamp then name."""
    files = []
    for p in map(Path, inputs):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    files = [f for f in files if not any(re.search(pat, f.name) for pat in exclude)]

    records, folder_cache = [], {}
    for f in files:
        if f.parent not in folder_cache:
            folder_cache[f.parent] = (_folder_scale(f.parent), _parts_csv(f.parent))
        (fscale, fsource), parts = folder_cache[f.parent]
        zmeta = zelloscope_metadata(f)
        if "p2mm" in zmeta:
            scale, source = float(zmeta["p2mm"]), "png-metadata"
        else:
            scale, source = fscale, fsource
        records.append(ImageRecord(f, scale, source, parse_timestamp(f, zmeta), parts.get(f.name, {})))
    records.sort(key=lambda r: (r.timestamp or dt.datetime.max, r.path.name))
    return records
