"""E2 (main-short): 용량 추세 — 압축≠모델가능성 갭이 모델 크기에 따라 어떻게 변하는가.
모델 크기 d∈{96,192,320,448}에서 L1 vs L1+BPE(1000) held-out NLL 갭 측정. 2 seed.
갭이 용량 증가에도 유지되면 헤드라인 강건, 줄어들면 정직하게 보고.
"""
import sys, statistics
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")
import numpy as np, torch
import e_common as C

corpus = C.load_corpus("starvector/svg-icons", n_train=1000, n_test=200, max_len=160)
tr, te = corpus["train"]["l1"], corpus["test"]["l1"]

# L1 (budget 0)
l1_tr, V_l1 = C.remap(tr + te)
l1_tr, l1_te = l1_tr[:len(tr)], l1_tr[len(tr):]
# L1+BPE(1000)
merges = C.bpe_train(tr, 1000)
bpe_all, V_bpe = C.remap([C.bpe_apply(s, merges) for s in tr] + [C.bpe_apply(s, merges) for s in te])
bpe_tr, bpe_te = bpe_all[:len(tr)], bpe_all[len(tr):]

print(f"train {len(tr)}, test {len(te)}\n")
print(f"{'d_model':>8}{'params':>10}{'L1 NLL':>10}{'L1+BPE NLL':>12}{'gap(BPE-L1)':>13}")
SIZES = [(96, 3), (160, 4), (256, 4), (384, 4)]   # ~1M → ~12M params (CPU-tractable)
for d, nl in SIZES:
    def run(seqs, V):
        bits = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            m = C.VanillaLM(V, d=d, nl=nl, nh=max(4, d // 64), ff=d * 4)
            C.train(m, seqs, epochs=15, max_len=200, seed=seed)
            bits.append(C.heldout_bits(m, l1_te if V == V_l1 else bpe_te, max_len=200))
        return statistics.mean(bits)
    nparams = C.count_params(C.VanillaLM(V_l1, d=d, nl=nl, nh=max(4, d // 64), ff=d * 4))
    b_l1 = run(l1_tr, V_l1)
    b_bpe = run(bpe_tr, V_bpe)
    print(f"{d:>8}{nparams:>10,}{b_l1:>10.0f}{b_bpe:>12.0f}{b_bpe - b_l1:>13.0f}", flush=True)

print("\n해석: gap>0 유지(L1<L1+BPE)면 압축≠모델가능성이 용량에 강건. gap이 d↑로 축소되면 스케일 의존 — 정직 보고.")
