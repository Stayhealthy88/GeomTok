"""E1 (main-short): 머지 예산 다운스트림 곡선.
L1 위 자체 BPE를 budget N∈{0,250,500,1000,2000}로 학습 → held-out NLL bits/icon.
질문: 토큰/icon은 줄지만 NLL은 나빠지는가(압축≠모델가능성이 점이 아니라 곡선)?
2 seed.
"""
import sys, statistics
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")
import numpy as np, torch
import e_common as C

corpus = C.load_corpus("starvector/svg-icons", n_train=1000, n_test=200, max_len=160)
tr, te = corpus["train"]["l1"], corpus["test"]["l1"]
print(f"train {len(tr)}, test {len(te)} 아이콘\n")
print(f"{'budget':>7}{'tok/icon':>10}{'vocab':>8}{'held-out NLL bits/icon':>26}")

BUDGETS = [0, 250, 500, 1000, 2000]
rows = []
for N in BUDGETS:
    merges = C.bpe_train(tr, N) if N else []
    tr_enc = [C.bpe_apply(s, merges) for s in tr]
    te_enc = [C.bpe_apply(s, merges) for s in te]
    # train+test 함께 remap(테스트 심볼도 동일 id 공간)
    all_enc, V = C.remap(tr_enc + te_enc)
    tr_e, te_e = all_enc[:len(tr_enc)], all_enc[len(tr_enc):]
    tpi = np.mean([len(s) for s in te_e])
    bits = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        m = C.VanillaLM(V)
        C.train(m, tr_e, epochs=22, max_len=200, seed=seed)
        bits.append(C.heldout_bits(m, te_e, max_len=200))
    mean, sd = statistics.mean(bits), (statistics.stdev(bits) if len(bits) > 1 else 0)
    rows.append((N, tpi, V, mean, sd))
    print(f"{N:>7}{tpi:>10.1f}{V:>8}      {mean:>8.0f} ± {sd:<4.0f}", flush=True)

print("\n해석: budget↑ → tok/icon↓(압축↑)인데 NLL↑(모델링↓)이면 '압축≠모델가능성'이 곡선으로 확립.")
best = min(rows, key=lambda r: r[3])
print(f"  최저 NLL: budget={best[0]} ({best[3]:.0f} bits). budget 0(순수 L1)이 최저면 머지는 다운스트림 순손해.")
