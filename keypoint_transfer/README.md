# Key-point transfer pipeline (fast setting)

Same method as `../keypoint_pipeline`, tuned to run **about 2x faster** at the same median / p90 accuracy.
Transfers key points from a reference image to a new image of the same, non-rigidly deformed part
(dataset: `../dataset_out`, references in `../New folder`).

## What changed against `../keypoint_pipeline`
| | before | now |
|---|---|---|
| RoMa passes | coarse 336 px + upsample/refinement 448 px | **one coarse pass at 224 px, no refinement pass** |
| ONNX models | 12 files | 7 files (DINOv2, VGG19, 5 decoder refiners) |
| everything else (orientation, filtering, TPS field, NCC refinement) | | unchanged |

Why it is safe: the final full-resolution NCC refinement of every key point does the sub-pixel work, so the extra 448 px
RoMa pass did not improve the result. The two settings are `COARSE_RES` and `UPSAMPLE_RES` in `keypoint_transfer.py`
(env `ROMA_COARSE`, `ROMA_UPSAMPLE`; `ROMA_UPSAMPLE=0` = no refinement pass). The old setting is
`ROMA_COARSE=336 ROMA_UPSAMPLE=448` (it exports 12 models into the same folder next to the 7).

### Measured (same machine, onnxruntime 1.27.0, 80 dataset images = 10 per shape, reference features cached, one image at a time)
| setting | s / image | endpoint error p50 / p90 / p95 [px] |
|---|---|---|
| 336 + 448 (before) | 6.81 | 0.82 / 2.22 / 4.29 |
| **224, no refinement pass (this folder)** | **3.31 (2.06x)** | 0.83 / 2.24 / 3.91 |

Caveat: the tail is slightly heavier. Key points with > 10 px error: 5 of 480 (before: 2 of 480), worst 22 px (before 10 px);
the extra outliers are mostly on shape 7. Not yet run on all 800 images (`python run_evaluation.py`, about 45 min).

## Environment
Developed and measured on: **Windows 10, Python 3.12, CPU only** (Intel i7-11850H, 8 cores / 16 threads, no GPU).
Pinned packages are in `requirements.txt`:

| package | version | note |
|---|---|---|
| python | 3.12 | |
| onnxruntime | **1.27.0** | **must be >= 1.27**: on 1.19 - 1.23 the same models take about 2x longer (slow CPU GridSample kernel) |
| onnx | 1.23.2 | only for the one-time model export |
| torch | 2.14.1 (CPU) | RoMa's PyTorch parts (GP / transformer decoder), model export |
| romatch (RoMa) | 0.1.2.post2 | `pip install git+https://github.com/Parskatt/RoMa.git`; downloads RoMa + DINOv2 weights on first use |
| numpy / scipy | 2.4.6 / 1.17.0 | |
| opencv-python(-headless) | 5.0.x | |
| pillow, einops, timm, loguru | 10.4.0, 0.8.2, 1.0.30, 0.7.3 | |

Hardware notes: timings scale with the machine (this CPU: 6.8 s -> 3.3 s). `ROMA_THREADS` defaults to 8 (physical cores);
16 threads was slower. About 1.3 GB disk for `onnx_models/`, about 4 GB RAM.
```
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "import onnxruntime; print(onnxruntime.__version__)"     # must print 1.27.0 or newer
```

## Files
| File | What |
|---|---|
| `keypoint_transfer.py` | the method (also a CLI: `python keypoint_transfer.py REF NEW --points "x,y;x,y" --out overlay.png`) |
| `roma_onnx.py` | ONNX Runtime fp32 backend for RoMa; caches the reference's encoder features |
| `run_evaluation.py` | evaluates on the dataset, grouped by reference, resumable, writes `results/per_image.csv` + `percentiles.csv` |
| `export_models.py` | creates / repairs the ONNX models in `onnx_models/` (7 files); `--verify` compares them with PyTorch |
| `requirements.txt` | pinned packages |

## Run
```
python export_models.py --verify       # once (about 30 s - 1 min): exports the ONNX models, checks them against PyTorch
python run_evaluation.py               # all 800 images, results in ./results
python run_evaluation.py --shapes 8 --limit 10     # quick test
```
Settings (env): `ROMA_COARSE` (224), `ROMA_UPSAMPLE` (0 = off), `ROMA_THREADS` (8), `ROMA_BACKEND` (`onnx` | `torch`),
`ROMA_ONNX_DIR` (model folder). If you skip the export, the models are created during the first run.
The first image of every new reference is slower (its encoder features are computed and cached); production use with a fixed
reference pays this once.

## Pipeline (per image)
1. **Orientation**: SIFT matches + RANSAC rotation (PCA of the silhouettes as fallback), crop of the part.
2. **RoMa** dense matching, a single pass at 224 px, ONNX Runtime fp32; reference features cached.
3. Match filtering, **thin-plate-spline** field fit with outline snapping.
4. Local **NCC refinement** of every key point at full resolution.

## Where the time goes now (about 3.3 s / image)
DINOv2 encoder about 0.6 s, decoder refiners about 0.5 s, TPS field fit (dense solves) about 1.3 s, GP/transformer about 0.3 s,
SIFT orientation about 0.1 s, rest NCC / glue. Tried and rejected because they hurt accuracy: a coarser first TPS fit, striding the
outline points, fewer grid cells (median error 0.76 -> 1.0 px or worse).
