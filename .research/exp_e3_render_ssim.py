"""실험 6 (E3/E4): 렌더 기반 SSIM — adaptive vs uniform 좌표, 실세계 150 아이콘."""
import sys, io
sys.path.insert(0, "/Users/limit/Projects")

import numpy as np
import pandas as pd
import resvg_py
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from huggingface_hub import hf_hub_download
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.arcs import ARCS
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer

N, RES, CANVAS = 150, 256, 300.0
df = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(N)
parser = SVGParser(normalize_canvas=CANVAS)


def render(svg):
    png = resvg_py.svg_to_bytes(svg_string=svg, width=RES, height=RES)
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, rgba).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def raw_d(e):
    parts = []
    for c in e.commands:
        v = c.abs_params or []
        k = c.command_type.value
        if k in ("M", "L"):
            parts.append(f"{k} {v[0]:.2f} {v[1]:.2f}")
        elif k in ("C",):
            parts.append("C " + " ".join(f"{x:.2f}" for x in v[:6]))
        elif k in ("Q",):
            parts.append("Q " + " ".join(f"{x:.2f}" for x in v[:4]))
        elif k == "Z":
            parts.append("Z")
    return " ".join(parts)


def recon_svg(doc, arcs_override=None, quant=True):
    parts = []
    for e in doc.elements:
        if quant:
            pt = PrimitiveTokenizer()
            if arcs_override is not None:
                pt.arcs = arcs_override
            res = pt.tokenize(e.commands)
            d = Detokenizer(pt.vocab, pt.arcs).detokenize(res.token_ids)
        else:
            d = raw_d(e)
        if d:
            parts.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="3"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">'
            + "".join(parts) + "</svg>")


scores = {"adaptive": [], "uniform": []}
fails = 0
for svg in df["Svg"]:
    try:
        doc = parser.parse_string(svg)
        if not doc.elements:
            continue
        ref = render(recon_svg(doc, None, quant=False))
        for key, arcs in (("adaptive", ARCS(canvas_size=CANVAS, max_level=6, min_level=2)),
                          ("uniform", ARCS(canvas_size=CANVAS, max_level=6, min_level=6))):
            out = render(recon_svg(doc, arcs))
            ink = (ref < 0.9) | (out < 0.9)
            iou = ((ref < 0.9) & (out < 0.9)).sum() / max(ink.sum(), 1)
            scores[key].append((ssim(ref, out, data_range=1.0), iou))
    except Exception:
        fails += 1

for k, v in scores.items():
    v = np.array(v)
    print(f"{k:<9} SSIM {v[:,0].mean():.4f}  inkIoU {v[:,1].mean():.4f} ± {v[:,1].std():.4f}  (n={len(v)}, fails={fails})")
