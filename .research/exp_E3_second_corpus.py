"""E3 (main-short): 2번째 코퍼스(svg-emoji)에서 §5.1 효율 + §5.3 다운스트림 재현.
단일 벤치마크 리스크 제거. char-BPE(corpus 텍스트 SP) / L1 / L1+BPE(자체 1000).
held-out NLL bits/icon 3 seed + tok/icon.
"""
import sys, statistics, math
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer/.research")
import numpy as np, torch
import e_common as C

DATASET = "starvector/svg-emoji"
corpus = C.load_corpus(DATASET, n_train=1000, n_test=200, max_len=160)
tr, te = corpus["train"], corpus["test"]
print(f"코퍼스 {DATASET}: train {len(tr['l1'])}, test {len(te['l1'])}\n")

# char-BPE: emoji 텍스트로 SP 학습(vocab 5561)
sp = C.train_sp_text(tr["svg"], 5561, "/tmp/svg_bpe_emoji_5561")
char_tr = [sp.encode(s)[:160] for s in tr["svg"]]
char_te = [sp.encode(s)[:160] for s in te["svg"]]
# L1
l1_all, V_l1 = C.remap(tr["l1"] + te["l1"]); l1_tr, l1_te = l1_all[:len(tr["l1"])], l1_all[len(tr["l1"]):]
# L1+BPE(1000)
merges = C.bpe_train(tr["l1"], 1000)
bpe_all, V_bpe = C.remap([C.bpe_apply(s, merges) for s in tr["l1"]] + [C.bpe_apply(s, merges) for s in te["l1"]])
bpe_tr, bpe_te = bpe_all[:len(tr["l1"])], bpe_all[len(tr["l1"]):]

arms = {
    "char_bpe": (char_tr, char_te, 5561),
    "l1":       (l1_tr, l1_te, V_l1),
    "l1_bpe":   (bpe_tr, bpe_te, V_bpe),
}
print(f"{'arm':<10}{'vocab':>7}{'tok/icon':>10}{'held-out NLL bits/icon (3 seed)':>32}")
res = {}
for arm, (xtr, xte, V) in arms.items():
    bits = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        m = C.VanillaLM(V)
        C.train(m, xtr, epochs=22, max_len=200, seed=seed)
        bits.append(C.heldout_bits(m, xte, max_len=200))
    res[arm] = bits
    tpi = np.mean([len(s) for s in xte])
    mean, sd = statistics.mean(bits), statistics.stdev(bits)
    print(f"{arm:<10}{V:>7}{tpi:>10.1f}      {mean:>8.0f} ± {sd:<5.0f}", flush=True)

print(f"\n해석(2번째 코퍼스): L1 {statistics.mean(res['l1']):.0f} < L1+BPE {statistics.mean(res['l1_bpe']):.0f} "
      f"< char {statistics.mean(res['char_bpe']):.0f} 이면 svg-icons 결과가 도메인 독립으로 재현 — 단일벤치 리스크 해소.")
