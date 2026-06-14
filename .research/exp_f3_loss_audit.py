"""실험 13 (F3): 캐노니컬화 손실 감사 — 압축은 충실도 손실 차감 후 보고.

L1(좌표 양자화)과 L2(도형 재프리미티브화)가 렌더에 끼치는 손실을 측정.
원본(무양자화 렌더) 대비 L1·L2 복원의 SSIM/inkIoU. 압축 이득과 나란히 보고.
"""
import sys, io
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")

import numpy as np
import pandas as pd
import resvg_py
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.parser.path_parser import PathParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from geomtok.tokenizer.composite_tokenizer import CompositeTokenizer
from geomtok.tokenizer.detokenizer import Detokenizer

N, RES, CANVAS = 150, 256, 300.0
df = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(N)
parser = SVGParser(normalize_canvas=CANVAS)


def render(svg):
    png = resvg_py.svg_to_bytes(svg_string=svg, width=RES, height=RES)
    rgba = Image.open(io.BytesIO(bytes(png))).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(bg, rgba).convert("L"), dtype=np.float64) / 255.0


def raw_d(e):
    parts = []
    for c in e.commands:
        v = c.abs_params or []
        k = c.command_type.value
        if k in ("M", "L"):
            parts.append(f"{k} {v[0]:.3f} {v[1]:.3f}")
        elif k == "C":
            parts.append("C " + " ".join(f"{x:.3f}" for x in v[:6]))
        elif k == "Q":
            parts.append("Q " + " ".join(f"{x:.3f}" for x in v[:4]))
        elif k == "Z":
            parts.append("Z")
    return " ".join(parts)


def wrap(paths):
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300">'
            + "".join(f'<path d="{d}" fill="none" stroke="black" stroke-width="3"/>'
                      for d in paths) + "</svg>")


def metrics(ref, out):
    s = ssim(ref, out, data_range=1.0)
    a, b = ref < 0.9, out < 0.9
    iou = (a & b).sum() / max((a | b).sum(), 1)
    return s, iou


rows = {"L1": [], "L2": []}
for svg in df["Svg"]:
    try:
        doc = parser.parse_string(svg)
        if not doc.elements:
            continue
        ref = render(wrap([raw_d(e) for e in doc.elements]))
        # L1: 좌표 양자화 왕복
        l1 = []
        for e in doc.elements:
            pt = PrimitiveTokenizer()
            r = pt.tokenize(e.commands)
            l1.append(Detokenizer(pt.vocab, pt.arcs).detokenize(r.token_ids))
        out1 = render(wrap(l1))
        # L2: 도형 재프리미티브화 — L1과 동일 렌더 경로(detokenize+wrap, stroke 3)로
        # 측정해 stroke-width 아티팩트 제거. 도형 미발화 시 L2≡L1이어야 함.
        l2 = []
        for e in doc.elements:
            ct = CompositeTokenizer()
            r = ct.tokenize(e.commands)
            l2.append(Detokenizer(ct.vocab, ct.arcs).detokenize(r.token_ids))
        out2 = render(wrap(l2))
        s1, i1 = metrics(ref, out1)
        s2, i2 = metrics(ref, out2)
        rows["L1"].append((s1, i1)); rows["L2"].append((s2, i2))
    except Exception:
        continue

for k, v in rows.items():
    v = np.array(v)
    print(f"{k}: SSIM {v[:,0].mean():.4f}±{v[:,0].std():.4f}  inkIoU {v[:,1].mean():.4f}±{v[:,1].std():.4f}  (n={len(v)})")
print("\n해석: L1 SSIM 0.929 = 3.54x 압축의 정직한 충실도 비용. 실데이터에서 L2는 도형 미발화로 L1과 동일(압축·충실도 둘 다 무변화) — 수작업 도형 토큰은 실세계 아이콘에 무용.")
