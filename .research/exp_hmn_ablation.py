"""실험 2: HMN 초기화 ablation — 검증된 적 없던 "학습 도움" 주장의 첫 측정.

설계: use_hmn_init ∈ {True, False} × seed ∈ {0,1,2}, 동일 데이터(500샘플/15에폭),
지표: 최종 train/val loss, val accuracy, 유효 SVG 생성률(n=30).
"""
import sys, time, statistics
sys.path.insert(0, "/Users/limit/Projects/gpl-tokenizer")

import torch
from torch.utils.data import DataLoader
from geomtok.tokenizer.vocabulary import GPLVocabulary
from geomtok.tokenizer.arcs import ARCS
from geomtok.training.synthetic_dataset import SyntheticSVGDataset, SVGCollator
from geomtok.training.gpl_transformer import GPLTransformer, GPLTransformerConfig
from geomtok.training.trainer import GPLTrainer, TrainingConfig
from geomtok.training.generator import GPLGenerator

VOCAB = GPLVocabulary(max_coord_level=6)
ARCS_INST = ARCS(max_level=6)

train_ds = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=500, max_seq_len=64, seed=123)
val_ds = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=100, max_seq_len=64, seed=456)
coll = SVGCollator(VOCAB, max_seq_len=64)

results = {True: [], False: []}
for use_hmn in (True, False):
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=32, collate_fn=coll, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_ds, batch_size=32, collate_fn=coll)
        cfg = GPLTransformerConfig(d_model=128, d_type=16, d_coord=32, n_heads=4,
                                   n_layers=4, d_ff=512, max_seq_len=64, dropout=0.1,
                                   use_hmn_init=use_hmn)
        model = GPLTransformer(VOCAB, cfg)
        tcfg = TrainingConfig(epochs=15, batch_size=32, learning_rate=1e-3,
                              checkpoint_dir=f"/tmp/gpl_abl_{use_hmn}_{seed}",
                              save_every=100, eval_every=1, patience=100)
        t0 = time.time()
        hist = GPLTrainer(model, tcfg).train(train_loader, val_loader, verbose=False)
        gen = GPLGenerator(model, VOCAB, ARCS_INST)
        valid = sum(1 for _ in range(30) if gen.generate_unconditional(max_len=64).is_valid) / 30
        r = dict(val_loss=hist["val_loss"][-1], val_acc=hist["val_acc"][-1],
                 train_loss=hist["train_loss"][-1], valid_rate=valid, sec=time.time() - t0)
        results[use_hmn].append(r)
        print(f"HMN={use_hmn} seed={seed}: val_loss={r['val_loss']:.4f} val_acc={r['val_acc']:.3f} "
              f"train_loss={r['train_loss']:.4f} valid_SVG={valid:.0%} ({r['sec']:.0f}s)", flush=True)

print()
for use_hmn in (True, False):
    vl = [r["val_loss"] for r in results[use_hmn]]
    vr = [r["valid_rate"] for r in results[use_hmn]]
    print(f"HMN={use_hmn}: val_loss {statistics.mean(vl):.4f}±{statistics.stdev(vl):.4f} | "
          f"valid_SVG {statistics.mean(vr):.1%}±{statistics.stdev(vr):.1%}")
