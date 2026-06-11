"""실험 11: L2/L3 토큰이 실세계 코퍼스에서 실제로 발화하는가 + L1-위-BPE 대조.

신규성 비판 대응: L2/L3 발화율·압축 기여를 측정하고, L1 ID 스트림 위에
SentencePiece BPE(+1.5k 머지)를 학습해 hand-crafted 매크로 vs 학습 머지를 비교.
"""
import sys, os
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")

import pandas as pd
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from geomtok.tokenizer.composite_tokenizer import CompositeTokenizer
from geomtok.tokenizer.spatial_tokenizer import SpatialTokenizer

CANVAS = 300.0
parser = SVGParser(normalize_canvas=CANVAS)

tr = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/train-00000-of-00001.parquet",
                                     repo_type="dataset")).head(3000)
te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(400)


def l1_ids(svg):
    doc = parser.parse_string(svg)
    out = []
    for e in doc.elements:
        out += PrimitiveTokenizer().tokenize(e.commands).token_ids
    return out


# 1) 발화율 (test 400)
n = l1_tot = l2_tot = l3_tot = 0
fire_l2 = fire_l3 = 0
for svg in te["Svg"]:
    try:
        doc = parser.parse_string(svg)
        if not doc.elements:
            continue
        l1 = sum(len(PrimitiveTokenizer().tokenize(e.commands).token_ids) for e in doc.elements)
        l2r = [CompositeTokenizer().tokenize(e.commands) for e in doc.elements]
        l2 = sum(len(r.token_ids) for r in l2r)
        st = SpatialTokenizer()
        l3r = st.tokenize_multi([e.commands for e in doc.elements])
        l3 = len(l3r.token_ids)
    except Exception:
        continue
    n += 1
    l1_tot += l1; l2_tot += l2; l3_tot += l3
    if any(r.detected_shapes for r in l2r):
        fire_l2 += 1
    if any(60 <= t <= 70 for t in l3r.token_ids):
        fire_l3 += 1

print(f"n={n}: L1 {l1_tot/n:.0f}  L2 {l2_tot/n:.0f}  L3 {l3_tot/n:.0f} tok/icon")
print(f"L2 발화율 {fire_l2/n:.1%}, L3 발화율 {fire_l3/n:.1%}")
print(f"L2 압축기여 {1-l2_tot/l1_tot:.1%}, L3 {1-l3_tot/l1_tot:.1%}")

# 2) L1-위-BPE: L1 ID를 유니코드 문자로 인코딩해 SPM 학습
M = "/tmp/l1_bpe.model"
if not os.path.exists(M):
    with open("/tmp/l1_stream.txt", "w") as f:
        cnt = 0
        for svg in tr["Svg"]:
            try:
                ids = l1_ids(svg)
            except Exception:
                continue
            f.write("".join(chr(0x4E00 + i) for i in ids) + "\n")
            cnt += 1
    spm.SentencePieceTrainer.train(input="/tmp/l1_stream.txt", model_prefix="/tmp/l1_bpe",
                                   vocab_size=7000, model_type="bpe", character_coverage=1.0,
                                   max_sentence_length=100000)
sp = spm.SentencePieceProcessor(model_file=M)
bpe_tot = bn = 0
for svg in te["Svg"]:
    try:
        ids = l1_ids(svg)
        if not ids:
            continue
    except Exception:
        continue
    bpe_tot += len(sp.encode("".join(chr(0x4E00 + i) for i in ids)))
    bn += 1
print(f"\nL1+BPE(+1.5k 머지): {bpe_tot/bn:.0f} tok/icon → L1 대비 {1-bpe_tot/(l1_tot/n*bn):.1%} 절감")
