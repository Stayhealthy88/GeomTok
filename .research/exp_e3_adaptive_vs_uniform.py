"""§6: 적응(곡률 트리) vs 균일 좌표 오차 — 실세계 400 아이콘.
PrimitiveTokenizer(uniform_coords=False)는 per-element 곡률 적응 쿼드트리를,
True는 균일 64×64 격자를 빌드. 동일 좌표를 양자화→역양자화한 픽셀 오차 비교.
결과(아카이브): adaptive mean 14.75px / max 52.37px vs uniform mean 1.78px / max 3.31px."""
import sys
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer

CANVAS = 300.0
df = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(400)
P = SVGParser(normalize_canvas=CANVAS)
print("E3 재측정: 적응(곡률 트리) vs 균일 좌표 오차 — 실세계 400 아이콘")
for label, uniform in (("adaptive(curvature tree)", False), ("uniform-L6", True)):
    errs, n = [], 0
    for svg in df["Svg"]:
        try:
            doc = P.parse_string(svg)
            if not doc.elements:
                continue
        except Exception:
            continue
        for e in doc.elements:
            pt = PrimitiveTokenizer(uniform_coords=uniform)
            pt.tokenize(e.commands)          # uniform=False면 per-element 적응 트리 빌드
            for x, y in [c.end_point for c in e.commands if c.end_point]:
                q = pt.arcs.quantize(x, y); dx, dy = pt.arcs.dequantize(q)
                errs.append(np.hypot(x - dx, y - dy)); n += 1
    errs = np.array(errs)
    print(f"  {label:<26} mean {errs.mean():.2f}px  max {errs.max():.2f}px  (coords {n})")
