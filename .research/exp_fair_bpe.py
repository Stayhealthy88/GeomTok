"""실험 10: 공정 베이스라인 — SVG 도메인 학습 BPE(어휘 5,561 동일) vs GPL.

비판 대응: cl100k(100k 어휘, 비도메인)은 불공정 베이스라인.
- SVG-Icons train 분할로 sentencepiece BPE(vocab=5561) 학습
- test 분할에서 tokens/icon + bits/icon(=tokens×log2(V)) 비교
"""
import sys, io, math, os
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")

import pandas as pd
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from geomtok.parser.svg_parser import SVGParser
from geomtok.tokenizer.spatial_tokenizer import SpatialTokenizer
from geomtok.tokenizer.primitive_tokenizer import PrimitiveTokenizer

CANVAS, VOCAB_SIZE = 300.0, 5561
MODEL = "/tmp/svg_bpe_5561.model"

if not os.path.exists(MODEL):
    tr = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/train-00000-of-00001.parquet",
                                         repo_type="dataset"))
    txt = "/tmp/svg_train.txt"
    with open(txt, "w") as f:
        for s in tr["Svg"].head(20000):
            f.write(s.replace("\n", " ") + "\n")
    spm.SentencePieceTrainer.train(
        input=txt, model_prefix="/tmp/svg_bpe_5561", vocab_size=VOCAB_SIZE,
        model_type="bpe", character_coverage=1.0, max_sentence_length=100000)

sp = spm.SentencePieceProcessor(model_file=MODEL)
te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(400)
parser = SVGParser(normalize_canvas=CANVAS)

import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

tot = dict(bpe_dom=0, cl100k=0, gpl_l1=0, gpl_l3=0)
n = 0
for svg in te["Svg"]:
    try:
        doc = parser.parse_string(svg)
        if not doc.elements:
            continue
        st = SpatialTokenizer()
        l3 = len(st.tokenize_multi([e.commands for e in doc.elements]).token_ids)
        l1 = sum(len(PrimitiveTokenizer().tokenize(e.commands).token_ids) for e in doc.elements)
    except Exception:
        continue
    n += 1
    tot["bpe_dom"] += len(sp.encode(svg))
    tot["cl100k"] += len(enc.encode(svg))
    tot["gpl_l1"] += l1
    tot["gpl_l3"] += l3

bits = dict(bpe_dom=math.log2(VOCAB_SIZE), cl100k=math.log2(100277),
            gpl_l1=math.log2(VOCAB_SIZE), gpl_l3=math.log2(VOCAB_SIZE))
print(f"n={n}")
print(f"{'scheme':<10}{'tok/icon':>10}{'bits/icon':>11}{'tok 압축vsBPEdom':>17}{'bits 압축vsBPEdom':>18}")
base_t, base_b = tot["bpe_dom"] / n, tot["bpe_dom"] / n * bits["bpe_dom"]
for k in tot:
    t = tot[k] / n
    b = t * bits[k]
    print(f"{k:<10}{t:>10.1f}{b:>11.0f}{base_t / t:>16.2f}x{base_b / b:>17.2f}x")
