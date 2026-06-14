"""실험 8 (E5): HMN v2 + NTL — 4 arm × 3 seed, 생성 100개로 valid율 CI 강화.

기준 (실험 2): HMN v1 val_loss 3.77 / valid 68%, random 4.91 / 82%.
성공: v2가 val_loss 우위 유지 + valid율 ≥ random.
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

ARMS = {
    "random":   dict(use_hmn_init=False, hmn_version=1, ntl_lambda=0.0),
    "hmn_v1":   dict(use_hmn_init=True,  hmn_version=1, ntl_lambda=0.0),
    "hmn_v2":   dict(use_hmn_init=True,  hmn_version=2, ntl_lambda=0.0),
    "v2_ntl":   dict(use_hmn_init=True,  hmn_version=2, ntl_lambda=0.5),
}

for arm, kw in ARMS.items():
    vls, accs, vrs = [], [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        tl = DataLoader(train_ds, batch_size=32, collate_fn=coll, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
        vl = DataLoader(val_ds, batch_size=32, collate_fn=coll)
        cfg = GPLTransformerConfig(d_model=128, d_type=16, d_coord=32, n_heads=4,
                                   n_layers=4, d_ff=512, max_seq_len=64, dropout=0.1, **kw)
        model = GPLTransformer(VOCAB, cfg)
        tcfg = TrainingConfig(epochs=15, batch_size=32, learning_rate=1e-3,
                              checkpoint_dir=f"/tmp/gpl_e5_{arm}_{seed}",
                              save_every=100, eval_every=1, patience=100)
        hist = GPLTrainer(model, tcfg).train(tl, vl, verbose=False)
        gen = GPLGenerator(model, VOCAB, ARCS_INST)
        valid = sum(1 for _ in range(100) if gen.generate_unconditional(max_len=64).is_valid) / 100
        vls.append(hist["val_loss"][-1]); accs.append(hist["val_acc"][-1]); vrs.append(valid)
        print(f"{arm} s{seed}: vl={vls[-1]:.3f} acc={accs[-1]:.3f} valid={valid:.0%}", flush=True)
    print(f">> {arm}: val_loss {statistics.mean(vls):.3f}±{statistics.stdev(vls):.3f} "
          f"acc {statistics.mean(accs):.3f} valid {statistics.mean(vrs):.1%}±{statistics.stdev(vrs):.1%}\n",
          flush=True)
