"""Create or repair the ONNX models used by the pipeline (roma_onnx.py), in ./onnx_models (or $ROMA_ONNX_DIR).

  python export_models.py              check existing files, delete broken/partial ones, export whatever is missing
  python export_models.py --force      delete everything and export all models again
  python export_models.py --verify     additionally compare the ONNX models with the plain PyTorch RoMa model

Needs torch, romatch, onnx, onnxruntime (see requirements.txt). The first run also downloads the RoMa / DINOv2 weights
if they are not cached yet. Exporting takes about a minute. Default (224 px, no refinement pass): 7 files (about 1.3 GB) = the DINOv2 encoder, the VGG19
encoder and the five decoder refiners. With ROMA_UPSAMPLE=448 (and ROMA_COARSE=336): 12 files (about 1.4 GB).
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--force", action="store_true", help="delete all existing models first")
ap.add_argument("--verify", action="store_true", help="compare with the PyTorch model on a synthetic image pair")
a = ap.parse_args()

import cv2
import torch
import onnxruntime as ort

import roma_onnx
import keypoint_transfer as kt

EXPECTED = 12 if kt.UPSAMPLE_RES else 7        # coarse pass only: DINOv2 + VGG19 + 5 refiners; with an upsample pass: 12
D = roma_onnx.MODEL_DIR
os.makedirs(D, exist_ok=True)
print("model folder:", D)

# 1. clean up: partial exports, forced re-export, files that do not load
for p in glob.glob(os.path.join(D, "*.tmp")):
    os.remove(p)
    print("removed partial export", os.path.basename(p))
for p in sorted(glob.glob(os.path.join(D, "*.onnx"))):
    name = os.path.basename(p)
    if a.force:
        os.remove(p)
        print("removed", name)
        continue
    try:
        ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    except Exception as ex:
        os.remove(p)
        print("removed broken model %s (%s)" % (name, str(ex).splitlines()[0][:80]))

# 2. export what is missing: one RoMa call on a synthetic pair exercises every module at both resolutions
rng = np.random.default_rng(0)
base = cv2.GaussianBlur((rng.random((700, 700)) * 255).astype(np.uint8), (0, 0), 4)
img0 = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)
M = cv2.getRotationMatrix2D((350, 350), 12, 1.0)
img1 = cv2.warpAffine(img0, M, (700, 700))
have = len(glob.glob(os.path.join(D, "*.onnx")))
t = time.time()
kt.BACKEND = "onnx"
kt.match_roma(img0, img1)
files = sorted(glob.glob(os.path.join(D, "*.onnx")))
print("%d models present (%d were missing), %.0fs" % (len(files), len(files) - have, time.time() - t))
for p in files:
    print("  %-62s %7.1f MB" % (os.path.basename(p), os.path.getsize(p) / 1e6))
print("total %.2f GB" % (sum(os.path.getsize(p) for p in files) / 1e9))
if len(files) != EXPECTED:
    print("WARNING: expected %d models, found %d" % (EXPECTED, len(files)))

# 3. optional check against the PyTorch model
if a.verify:
    from PIL import Image
    from romatch import roma_outdoor
    pil = lambda g: Image.fromarray(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    onnx_model = kt._ROMA[(kt.COARSE_RES, kt.UPSAMPLE_RES)]
    torch_model = roma_outdoor(device="cpu", coarse_res=kt.COARSE_RES, upsample_res=kt.UPSAMPLE_RES or kt.COARSE_RES, amp_dtype=torch.float32, symmetric=False)
    torch_model.upsample_preds = bool(kt.UPSAMPLE_RES)
    with torch.no_grad():
        w1, c1 = onnx_model.match(pil(img0), pil(img1), device="cpu")
        w0, c0 = torch_model.match(pil(img0), pil(img1), device="cpu")
    sel = c0 > 0.5
    dw = (w1 - w0).abs()[..., 2:][sel]
    px = 2 / (kt.UPSAMPLE_RES or kt.COARSE_RES)             # one pixel of the final output grid in normalised coordinates
    print("verify: warp difference on confident pixels: mean %.2e  max %.2e  (1 px = %.1e)  ->  %s"
          % (dw.mean(), dw.max(), px, "OK" if dw.max() < px / 5 else "TOO LARGE, re-run with --force"))
print("done")
