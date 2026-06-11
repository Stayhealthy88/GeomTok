"""Adversarial check: is the HMN val_loss advantage a geometric-structure effect,
or an init-scale optimization artifact from weight tying?

Arms (same data, same 15 epochs, same LR):
  random          : default nn.Embedding (row-norm ~11.3) — the paper's baseline
  random_unitnorm : SAME random vectors, rescaled to unit row-norm (matches HMN v2 scale)
  hmn_v2          : structured HMN
If random_unitnorm closes most of the gap to hmn_v2, the "HMN helps training"
claim is confounded by output-layer init scale, not geometry.
"""
import sys, statistics
sys.path.insert(0, "/Users/limit/Projects")
import torch
from torch.utils.data import DataLoader
from gpl_tokenizer.tokenizer.vocabulary import GPLVocabulary
from gpl_tokenizer.tokenizer.arcs import ARCS
from gpl_tokenizer.training.synthetic_dataset import SyntheticSVGDataset, SVGCollator
from gpl_tokenizer.training.gpl_transformer import GPLTransformer, GPLTransformerConfig
from gpl_tokenizer.training.trainer import GPLTrainer, TrainingConfig

VOCAB = GPLVocabulary(max_coord_level=6)
ARCS_INST = ARCS(max_level=6)
train_ds = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=500, max_seq_len=64, seed=123)
val_ds   = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=100, max_seq_len=64, seed=456)
coll = SVGCollator(VOCAB, max_seq_len=64)

def build(arm, seed):
    torch.manual_seed(seed)
    use_hmn = (arm == "hmn_v2")
    cfg = GPLTransformerConfig(d_model=128, d_type=16, d_coord=32, n_heads=4,
                               n_layers=4, d_ff=512, max_seq_len=64, dropout=0.1,
                               use_hmn_init=use_hmn, hmn_version=2)
    m = GPLTransformer(VOCAB, cfg)
    if arm == "random_unitnorm":
        with torch.no_grad():
            w = m.embedding.token_embedding.weight
            n = w.norm(dim=1, keepdim=True).clamp_min(1e-8)
            w.div_(n)  # unit row-norm, same directions as random
    return m

for arm in ("random", "random_unitnorm", "hmn_v2"):
    tr, vl = [], []
    for seed in (0, 1):
        torch.manual_seed(seed)
        tl = DataLoader(train_ds, batch_size=32, collate_fn=coll, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
        vloader = DataLoader(val_ds, batch_size=32, collate_fn=coll)
        model = build(arm, seed)
        tcfg = TrainingConfig(epochs=15, batch_size=32, learning_rate=1e-3,
                              checkpoint_dir=f"/tmp/ic_{arm}_{seed}",
                              save_every=1000, eval_every=1, patience=100)
        hist = GPLTrainer(model, tcfg).train(tl, vloader, verbose=False)
        tr.append(hist["train_loss"][-1]); vl.append(hist["val_loss"][-1])
        print(f"{arm:16s} s{seed}: train={tr[-1]:.3f} val={vl[-1]:.3f}", flush=True)
    print(f">> {arm:16s}: train {statistics.mean(tr):.3f}  val {statistics.mean(vl):.3f}\n", flush=True)
