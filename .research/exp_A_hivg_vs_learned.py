"""실험 A+B (v0.7): HiVG식 규칙 머지 vs GPL 학습 BPE — 동일 머지 알고리즘·예산.

핵심 질문: 학습 비제약 머지가 HiVG식 구조-제약 그루핑을 이기는가?
세 기질(substrate)에 *동일한* greedy BPE를 N 머지까지 적용해 tokens/icon 비교:
  - char      : 원문 SVG 문자 스트림 (도메인 char-BPE)
  - l1_flat   : GPL-L1 토큰 ID 스트림 (비제약 — 명령 경계 넘어 머지 가능)
  - hivg_seg  : 명령+그 좌표를 1세그먼트로 선그룹핑(구조 제약) 후 세그먼트 머지

동일 알고리즘·동일 예산이므로 차이는 순수 '기질 구조'의 효과.
"""
import sys, collections
sys.path.insert(0, "/Users/limit/Projects")

import pandas as pd
from huggingface_hub import hf_hub_download
from gpl_tokenizer.parser.svg_parser import SVGParser
from gpl_tokenizer.tokenizer.primitive_tokenizer import PrimitiveTokenizer
from gpl_tokenizer.tokenizer.vocabulary import COORD_TOKEN_BASE

CANVAS = 300.0
N_TRAIN, N_TEST = 1200, 400
BUDGETS = [0, 100, 250, 500, 1000]

parser = SVGParser(normalize_canvas=CANVAS)
tr = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/train-00000-of-00001.parquet",
                                     repo_type="dataset")).head(N_TRAIN)
te = pd.read_parquet(hf_hub_download("starvector/svg-icons", "data/test-00000-of-00001.parquet",
                                     repo_type="dataset")).head(N_TEST)


def l1_ids(svg):
    doc = parser.parse_string(svg)
    out = []
    for e in doc.elements:
        out += PrimitiveTokenizer().tokenize(e.commands).token_ids
    return out


def to_segments(ids):
    """HiVG식: 명령/도형 토큰을 앵커로 그 뒤 좌표·속성을 한 세그먼트로 묶음."""
    segs, cur = [], None
    for t in ids:
        is_anchor = (10 <= t < 24) or (60 <= t < 71) or t in (1, 2, 3)
        if is_anchor:
            if cur:
                segs.append(tuple(cur))
            cur = [t]
        else:
            if cur is None:
                cur = [t]
            else:
                cur.append(t)
    if cur:
        segs.append(tuple(cur))
    return segs


def bpe_train(corpus, n_merges):
    """심볼 시퀀스 리스트에 greedy BPE. 머지 규칙 리스트 반환."""
    seqs = [list(s) for s in corpus]
    merges = []
    for _ in range(n_merges):
        pairs = collections.Counter()
        for s in seqs:
            for i in range(len(s) - 1):
                pairs[(s[i], s[i + 1])] += 1
        if not pairs:
            break
        (a, b), c = pairs.most_common(1)[0]
        if c < 2:
            break
        new = ("M", len(merges))
        merges.append((a, b, new))
        for s in seqs:
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    s[i:i + 2] = [new]
                else:
                    i += 1
    return merges


def bpe_apply(seq, merges):
    s = list(seq)
    for a, b, new in merges:
        i = 0
        while i < len(s) - 1:
            if s[i] == a and s[i + 1] == b:
                s[i:i + 2] = [new]
            else:
                i += 1
    return s


# 코퍼스 준비
train_l1 = [l1_ids(s) for s in tr["Svg"]]
train_l1 = [x for x in train_l1 if x]
test_l1 = [l1_ids(s) for s in te["Svg"]]
test_l1 = [x for x in test_l1 if x]
test_svg = [s for s in te["Svg"]]

subs = {
    "char":     ([list(s) for s in tr["Svg"]], [list(s) for s in te["Svg"]]),
    "l1_flat":  (train_l1, test_l1),
    "hivg_seg": ([to_segments(x) for x in train_l1], [to_segments(x) for x in test_l1]),
}

print(f"train={len(train_l1)} test={len(test_l1)} 아이콘")
print(f"{'budget':>7}" + "".join(f"{k:>12}" for k in subs))
base = {}
for N in BUDGETS:
    row = {}
    for name, (trc, tec) in subs.items():
        merges = bpe_train(trc, N) if N else []
        tot = sum(len(bpe_apply(s, merges)) for s in tec)
        row[name] = tot / len(tec)
    if N == 0:
        base = dict(row)
    print(f"{N:>7}" + "".join(f"{row[k]:>12.1f}" for k in subs))

print("\n해석: l1_flat < hivg_seg 이면 '비제약 학습 머지 > 구조 제약'(논지 i 성립).")
print("char 대비 l1_flat 비율이 '기하 토큰 기질' 이득. 동일 알고리즘·예산이라 공정.")
