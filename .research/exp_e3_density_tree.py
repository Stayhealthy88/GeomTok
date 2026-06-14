"""실험 7 (E3): 밀도 적응 쿼드트리 — 같은 어휘 예산(≈4k)에서 깊이 8 허용.

가설: 적응 격자의 패배 원인은 '곡률만 보는' 분할 기준. 좌표 밀도 기반
코퍼스 전역 트리는 같은 어휘에서 균일 64×64(1.78px)보다 정밀하다.
"""
import sys
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.arcs import ARCS

CANVAS, N = 300.0, 400
df = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(N)
parser = SVGParser(normalize_canvas=CANVAS)

pts = []
for svg in df["Svg"]:
    try:
        doc = parser.parse_string(svg)
        pts += [c.end_point for e in doc.elements for c in e.commands if c.end_point]
    except Exception:
        pass
pts = np.array(pts)
print(f"코퍼스 좌표 {len(pts):,}개")


def leaves(node, acc):
    if node.is_leaf:
        acc.append(node)
    else:
        for ch in node.children:
            leaves(ch, acc)
    return acc


def eval_tree(arcs):
    errs, slots = [], set()
    for x, y in pts:
        q = arcs.quantize(x, y)
        dx, dy = arcs.dequantize(q)
        errs.append(np.hypot(x - dx, y - dy))
        slots.add((q.level, q.qx, q.qy))
    e = np.array(errs)
    return e.mean(), e.max(), len(slots)


# 코퍼스 전역 밀도 트리: 분할 기준 = 셀 내 좌표 수 > threshold
seg = [{"bbox": (x, y, x, y), "max_curvature": 1.0, "arc_length": 1.0} for x, y in pts]
print(f"\n{'tree':<22}{'thr':>7}{'leaves':>8}{'used':>7}{'mean_err':>9}{'max_err':>9}")
for max_level, thr in [(8, 40), (8, 80), (8, 160), (7, 40), (7, 80)]:
    arcs = ARCS(canvas_size=CANVAS, max_level=max_level, min_level=4, split_threshold=float(thr))
    arcs.build_from_curvatures(seg)
    n_leaves = len(leaves(arcs.root, []))
    m, mx, used = eval_tree(arcs)
    print(f"density L{max_level:<18}{thr:>7}{n_leaves:>8}{used:>7}{m:>9.3f}{mx:>9.2f}")

uni = ARCS(canvas_size=CANVAS, max_level=6, min_level=6)
m, mx, used = eval_tree(uni)
print(f"{'uniform 64x64':<22}{'-':>7}{4096:>8}{used:>7}{m:>9.3f}{mx:>9.2f}")
