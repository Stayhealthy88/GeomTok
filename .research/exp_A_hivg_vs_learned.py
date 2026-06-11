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


def is_anchor(t):
    """명령/도형/공간/특수 토큰 = 세그먼트 경계 앵커."""
    return isinstance(t, int) and ((10 <= t < 24) or (60 <= t < 71) or t in (1, 2, 3))


def bpe_train(corpus, n_merges, mode="none"):
    """greedy BPE — 증분 페어 카운트.
    mode: 'none'=비제약, 'boundary'=명령 경계 넘는 머지 차단,
          'hivg'=명령-루트 세그먼트만 성장(좌→앵커시작, 우→비앵커; 충실한 HiVG)."""
    seqs = [list(s) for s in corpus]
    anchor_of = {}   # 머지 심볼이 '앵커로 시작'하는지 추적

    def starts_anchor(sym):
        if isinstance(sym, tuple):
            return anchor_of.get(sym, False)
        return is_anchor(sym)

    def mergeable(a, b):
        if mode == "none":
            return True
        if mode == "boundary":
            return not starts_anchor(b)
        if mode == "hivg":   # 명령-루트 세그먼트 성장만 허용
            return starts_anchor(a) and not starts_anchor(b)
        return True

    pairs = collections.Counter()
    for s in seqs:
        for i in range(len(s) - 1):
            if mergeable(s[i], s[i + 1]):
                pairs[(s[i], s[i + 1])] += 1
    merges = []
    for _ in range(n_merges):
        if not pairs:
            break
        (a, b), c = max(pairs.items(), key=lambda kv: kv[1])
        if c < 2:
            break
        new = ("M", len(merges))
        anchor_of[new] = starts_anchor(a)
        merges.append((a, b, new))
        for s in seqs:
            if len(s) < 2:
                continue
            i = 0
            while i < len(s) - 1:
                if s[i] == a and s[i + 1] == b:
                    if i > 0 and mergeable(s[i - 1], a):
                        pairs[(s[i - 1], a)] -= 1
                    if i + 2 < len(s) and mergeable(b, s[i + 2]):
                        pairs[(b, s[i + 2])] -= 1
                    s[i] = new
                    del s[i + 1]
                    if i > 0 and mergeable(s[i - 1], new):
                        pairs[(s[i - 1], new)] += 1
                    if i + 1 < len(s) and mergeable(new, s[i + 1]):
                        pairs[(new, s[i + 1])] += 1
                else:
                    i += 1
        pairs.pop((a, b), None)
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

import math


def bits_per_icon(test_seqs, merges, train_seqs):
    """order-0 엔트로피: train 유니그램 분포로 test 시퀀스 비트 추정."""
    tr_enc = [bpe_apply(s, merges) for s in train_seqs]
    freq = collections.Counter(sym for s in tr_enc for sym in s)
    tot = sum(freq.values())
    bits = {sym: -math.log2(c / tot) for sym, c in freq.items()}
    avg_bit = math.log2(len(freq)) if freq else 0
    out = 0.0
    for s in test_seqs:
        enc = bpe_apply(s, merges)
        out += sum(bits.get(sym, avg_bit) for sym in enc)
    return out / len(test_seqs)


# 모든 arm 동일 머지 수. char/L1-비제약, L1-경계제약, L1-충실HiVG(명령루트).
subs = {
    "char_bpe":   ([list(s) for s in tr["Svg"]], [list(s) for s in te["Svg"]], "none"),
    "l1_uncon":   (train_l1, test_l1, "none"),
    "l1_bound":   (train_l1, test_l1, "boundary"),
    "l1_hivg":    (train_l1, test_l1, "hivg"),
}

print(f"train={len(train_l1)} test={len(test_l1)} 아이콘  (모든 arm 동일 머지 수)")
print(f"{'budget':>7}" + "".join(f"{k:>14}" for k in subs))
last = {}
for N in BUDGETS:
    row = {}
    for name, (trc, tec, mode) in subs.items():
        merges = bpe_train(trc, N, mode=mode) if N else []
        tot = sum(len(bpe_apply(s, merges)) for s in tec)
        row[name] = (tot / len(tec), merges, trc, tec)
    print(f"{N:>7}" + "".join(f"{row[k][0]:>14.1f}" for k in subs))
    last = row

print(f"\nbits/icon @{BUDGETS[-1]}머지 (order-0 엔트로피):")
for k in subs:
    _, merges, trc, tec = last[k]
    print(f"  {k:<10} {bits_per_icon(tec, merges, trc):>8.0f} bits")

# 동일 총어휘 대조: char-BPE에 L1 총어휘만큼 머지 부여
print("\n동일 총어휘 대조 (char-BPE에 L1과 같은 총어휘 예산):")
base_l1 = len({sym for s in train_l1 for sym in s})
base_ch = len({c for s in tr["Svg"] for c in s})
extra = base_l1 + BUDGETS[-1] - base_ch
m_ch = bpe_train([list(s) for s in tr["Svg"]], extra, mode="none")
tot_ch = sum(len(bpe_apply(list(s), m_ch)) for s in te["Svg"]) / len(te)
print(f"  L1 기저어휘 {base_l1}, char 기저 {base_ch} → char에 +{extra}머지: {tot_ch:.1f} tok/icon")
print(f"  vs L1-uncon {last['l1_uncon'][0]:.1f} → 동일 총어휘에서도 L1 {tot_ch/last['l1_uncon'][0]:.1f}배 우위")

print("\n해석: l1_uncon < l1_bound < l1_hivg 이면 '비제약 학습 > 구조 제약'(논지 i, 충실 HiVG 포함).")
