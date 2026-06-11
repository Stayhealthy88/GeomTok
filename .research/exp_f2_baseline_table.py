"""실험 12 (F2): 공정 베이스라인 표 — 실세계 아이콘에서 토큰 효율·좌표 충실도.

베이스라인: char-BPE(5,561) / GPL-L1(uniform-L6) / GPL-L1+학습BPE.
지표: tokens/icon, bits/icon, path 좌표 충실도(F1 확장 GeomTok-Eval).
cl100k는 참고 각주로만.
"""
import sys, os, math, re
sys.path.insert(0, "/Users/limit/Projects")

import numpy as np
import pandas as pd
import sentencepiece as spm
import tiktoken
from huggingface_hub import hf_hub_download
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.detokenizer import Detokenizer
from gpl_tokenizer.evaluation.value_fidelity import _extract_path_coords

CANVAS, VOCAB = 300.0, 5561
parser = SVGParser(normalize_canvas=CANVAS)
te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(400)

sp_text = spm.SentencePieceProcessor(model_file="/tmp/svg_bpe_5561.model")
sp_l1 = spm.SentencePieceProcessor(model_file="/tmp/l1_bpe.model")
cl = tiktoken.get_encoding("cl100k_base")


def l1_tokenize(doc):
    out = []
    for e in doc.elements:
        out += PrimitiveTokenizer().tokenize(e.commands).token_ids
    return out


tot = dict(charBPE=0, cl100k=0, gpl_l1=0, gpl_l1_bpe=0)
n = 0
fid_errs = []   # GPL-L1 좌표 충실도 (정규화 캔버스 px)
for svg in te["Svg"]:
    try:
        doc = parser.parse_string(svg)
        if not doc.elements:
            continue
        ids = l1_tokenize(doc)
        if not ids:
            continue
    except Exception:
        continue
    n += 1
    tot["charBPE"] += len(sp_text.encode(svg))
    tot["cl100k"] += len(cl.encode(svg))
    tot["gpl_l1"] += len(ids)
    tot["gpl_l1_bpe"] += len(sp_l1.encode("".join(chr(0x4E00 + i) for i in ids)))

    # F1 좌표 충실도: 원본 end_point vs L1 왕복 후 PathParser end_point (정렬 일치)
    from gpl_tokenizer.parser.path_parser import PathParser
    pp = PathParser()
    for e in doc.elements:
        o = [c.end_point for c in e.commands if c.end_point]
        pt = PrimitiveTokenizer()
        rr = pt.tokenize(e.commands)
        d = Detokenizer(pt.vocab, pt.arcs).detokenize(rr.token_ids)
        rc = pp.resolve_to_absolute(pp.parse(d))
        r = [c.end_point for c in rc if c.end_point]
        for (ox, oy), (rx, ry) in zip(o, r):
            fid_errs.append(math.hypot(ox - rx, oy - ry))

bits = dict(charBPE=math.log2(VOCAB), cl100k=math.log2(100277),
            gpl_l1=math.log2(VOCAB), gpl_l1_bpe=math.log2(7000))
base = tot["charBPE"] / n
print(f"n={n}  좌표 충실도(GPL-L1): mean {np.mean(fid_errs):.2f}px  max {np.max(fid_errs):.2f}px\n")
print(f"{'scheme':<12}{'tok/icon':>10}{'bits/icon':>11}{'압축vs charBPE':>16}")
for k in ("cl100k", "charBPE", "gpl_l1", "gpl_l1_bpe"):
    t = tot[k] / n
    note = " (참고)" if k == "cl100k" else ""
    print(f"{k:<12}{t:>10.1f}{t*bits[k]:>11.0f}{base/t:>15.2f}x{note}")
