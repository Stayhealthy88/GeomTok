"""실험 5 (E3): 좌표 체계 통제 비교 — 실세계 코퍼스.

질문: 같은 토큰 예산에서 어떤 좌표 인코딩이 좋은가?
  A) ARCS-adaptive (GPL 현행): 적응 쿼드트리, 좌표당 1토큰, 어휘 5,461
  B) Uniform-L6 (HiVG 부류): 64×64 고정, 좌표당 1토큰, 어휘 4,096
  C) OmniSVG식: 200×200 조인트, 좌표당 1토큰, 어휘 40,000
  D) cl100k BPE: 원문 (토큰 수만)

지표: 토큰/아이콘, 좌표 오차(평균·최대), 사용 어휘 슬롯(코퍼스 전체),
      오차당 어휘 효율 = 1 / (어휘슬롯 × 평균오차).
"""
import sys
sys.path.insert(0, "/Users/limit/Projects")

import numpy as np
import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.arcs import ARCS
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer

N_SAMPLE = 400
CANVAS = 300.0

df = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(N_SAMPLE)
parser = SVGParser(normalize_canvas=CANVAS)
enc = tiktoken.get_encoding("cl100k_base")

stats = {k: dict(tok=0, err=[], slots=set()) for k in ("arcs", "uniform", "omni")}
bpe_tok = 0
n_icons = 0

for svg in df["Svg"]:
    try:
        doc = parser.parse_string(svg)
        if not doc.elements:
            continue
    except Exception:
        continue
    n_icons += 1
    bpe_tok += len(enc.encode(svg))

    # 원본 좌표 수집 (정규화 캔버스 기준 end_point)
    pts = [c.end_point for e in doc.elements for c in e.commands if c.end_point]

    # A) ARCS-adaptive: 토크나이저 경유 (요소별 적응 트리)
    pt = PrimitiveTokenizer()
    n_tok = 0
    for e in doc.elements:
        res = pt.tokenize(e.commands)
        n_tok += len(res.token_ids)
        for x, y in [c.end_point for c in e.commands if c.end_point]:
            q = pt.arcs.quantize(x, y)
            dx, dy = pt.arcs.dequantize(q)
            stats["arcs"]["err"].append(np.hypot(x - dx, y - dy))
            stats["arcs"]["slots"].add((q.level, q.qx, q.qy))
    stats["arcs"]["tok"] += n_tok

    # B/C) 좌표 토큰 수는 A와 동일(좌표당 1토큰) — 오차·슬롯만 상이
    uni = ARCS(canvas_size=CANVAS, max_level=6, min_level=6)
    for x, y in pts:
        q = uni.quantize(x, y)
        dx, dy = uni.dequantize(q)
        stats["uniform"]["err"].append(np.hypot(x - dx, y - dy))
        stats["uniform"]["slots"].add((q.qx, q.qy))
        gx = min(199, int(x / CANVAS * 200)); gy = min(199, int(y / CANVAS * 200))
        ox = (gx + 0.5) * CANVAS / 200; oy = (gy + 0.5) * CANVAS / 200
        stats["omni"]["err"].append(np.hypot(x - ox, y - oy))
        stats["omni"]["slots"].add(gx * 200 + gy)
    stats["uniform"]["tok"] = stats["omni"]["tok"] = stats["arcs"]["tok"]

print(f"아이콘 {n_icons}개, BPE 평균 {bpe_tok/n_icons:.0f} tok/icon, GPL-L1 평균 {stats['arcs']['tok']/n_icons:.0f} tok/icon → 압축 {bpe_tok/stats['arcs']['tok']:.2f}x\n")
print(f"{'scheme':<10}{'vocab':>7}{'used':>7}{'mean_err':>9}{'max_err':>9}{'util%':>7}{'eff=1/(used·μerr)':>19}")
for k, vocab in (("arcs", 5461), ("uniform", 4096), ("omni", 40000)):
    e = np.array(stats[k]["err"]); used = len(stats[k]["slots"])
    eff = 1.0 / (used * e.mean()) * 1e4
    print(f"{k:<10}{vocab:>7}{used:>7}{e.mean():>9.3f}{e.max():>9.2f}{used/vocab*100:>6.1f}%{eff:>18.2f}")
