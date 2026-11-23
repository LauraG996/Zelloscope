"""ONNX Runtime (fp32, CPU) backend for the RoMa matcher used by keypoint_transfer.

Replaces the heavy submodules of a loaded torch RoMa model by ONNX Runtime sessions:
  * DINOv2 ViT-L encoder and VGG19 encoder  (exported with batch size 1)
  * the five decoder conv_refiner stages
The reference image is always the first image of RoMa's batch; its encoder features are cached (keyed by the
reference tensor itself), so only the new image goes through the encoders after the first call for a reference.

The ONNX files are exported on first use into ./onnx_models (static shapes: 336 px coarse / 448 px upsample pass);
`python export_models.py` creates / repairs them explicitly.
Requires: onnx, onnxruntime (export also needs torch, which keypoint_transfer already requires).
Results equal the torch model to ~1e-4 relative (fp32); no quantization (int8 was tested and broke accuracy).
"""
import hashlib
import os
from collections import OrderedDict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("ROMA_ONNX_DIR", os.path.join(HERE, "onnx_models"))   # where the exported .onnx files live
THREADS = int(os.environ.get("ROMA_THREADS", 8))        # physical cores; hyper-threads gave no gain
CACHE_MAX = 32                                          # reference-feature entries kept (2 per reference: coarse + upsample)

_SESSIONS = {}
_CACHE = OrderedDict()
STATS = {"cache_hits": 0, "cache_misses": 0}


class _Fn(torch.nn.Module):
    """Wrap a function so that torch.onnx.export sees the module's parameters."""

    def __init__(self, fn, mod):
        super().__init__()
        self.fn, self.m = fn, mod

    def forward(self, *a):
        return self.fn(*a)


def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # idle ORT threads must not fight other cores
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _get(name, fn, mod, tensors):
    """Session for `name` (export on first use)."""
    if name not in _SESSIONS:
        path = os.path.join(MODEL_DIR, name + ".onnx")
        if not os.path.exists(path):
            os.makedirs(MODEL_DIR, exist_ok=True)
            tmp = path + ".tmp"
            with torch.no_grad():
                torch.onnx.export(_Fn(fn, mod), tuple(tensors), tmp, opset_version=17, dynamo=False,
                                  input_names=["i%d" % i for i in range(len(tensors))])
            os.replace(tmp, path)
        _SESSIONS[name] = _session(path)
    return _SESSIONS[name]


def _run(sess, tensors):
    names = {i.name for i in sess.get_inputs()}               # the exporter drops unused inputs
    feed = {"i%d" % i: t.detach().float().numpy() for i, t in enumerate(tensors) if "i%d" % i in names}
    return sess.run(None, feed)


def _shape(t):
    return "x".join(map(str, t.shape))


def _patch_conv(k, cr):
    orig = cr.forward

    def fwd(x, y, warp, scale_factor=1, logits=None):
        tens = [x, y, warp] + ([logits] if logits is not None else [])
        name = "conv%s_%s_sf%.6f_lg%d" % (k, "_".join(_shape(t) for t in tens), scale_factor, logits is not None)
        if logits is not None:
            fn = lambda a, b, w, lg: orig(a, b, w, scale_factor=scale_factor, logits=lg)
        else:
            fn = lambda a, b, w: orig(a, b, w, scale_factor=scale_factor)
        d, c = _run(_get(name, fn, cr, tens), tens)
        return torch.from_numpy(d), torch.from_numpy(c)
    cr.forward = fwd


def _patch_encoder(m):
    enc = m.encoder
    cnn, dn = enc.cnn, enc.dinov2_vitl14[0]
    orig_cnn, orig_dino = cnn.forward, dn.forward_features

    def dino_fn(x):
        # bake the size-dependent positional embedding into a constant (its interpolation does not trace)
        with torch.no_grad():
            tok = torch.cat((dn.cls_token.expand(x.shape[0], -1, -1), dn.patch_embed(x)), dim=1)
            pe = dn.interpolate_pos_encoding(tok, x.shape[2], x.shape[3])

        def f(t):
            old = dn.interpolate_pos_encoding
            dn.interpolate_pos_encoding = lambda a, w, h: pe
            try:
                return orig_dino(t)["x_norm_patchtokens"]
            finally:
                dn.interpolate_pos_encoding = old
        return f

    def features(x, upsample):                                 # x: (1, 3, H, W)
        H, W = x.shape[-2:]
        vgg = _get("vgg_" + _shape(x), lambda t: tuple(orig_cnn(t)[s] for s in (1, 2, 4, 8)), cnn, [x])
        pyr = {s: torch.from_numpy(o) for s, o in zip((1, 2, 4, 8), _run(vgg, [x]))}
        if not upsample:
            dino = _get("dino_" + _shape(x), dino_fn(x), dn, [x])
            tok = torch.from_numpy(_run(dino, [x])[0])
            pyr[16] = tok.permute(0, 2, 1).reshape(1, tok.shape[2], H // 14, W // 14)
        return pyr

    def fwd(X, upsample=False):
        assert X.shape[0] == 2, "expected the batch [reference, new]"
        ref, new = X[:1], X[1:]
        key = (hashlib.blake2b(ref.numpy().tobytes(), digest_size=16).digest(), tuple(X.shape), bool(upsample))
        if key in _CACHE:
            _CACHE.move_to_end(key)
            STATS["cache_hits"] += 1
        else:
            STATS["cache_misses"] += 1
            _CACHE[key] = features(ref, upsample)
            while len(_CACHE) > CACHE_MAX:
                _CACHE.popitem(last=False)
        pr, pn = _CACHE[key], features(new, upsample)
        return {s: torch.cat([pr[s], pn[s]], 0) for s in pn}
    enc.forward = fwd


def install(model):
    """Route the heavy parts of a torch RoMa model (roma_outdoor, symmetric=False) through ONNX Runtime."""
    for k, cr in model.decoder.conv_refiner.items():
        _patch_conv(k, cr)
    _patch_encoder(model)
    return model
