"""실험 9 (E4): FSA 제약 디코딩 + 렌더 평가기.

2 arm (unconstrained vs constrained) × 3 seed × 생성 100, HMN v2 모델.
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
from gpl_tokenizer.training.generator import GPLGenerator
from gpl_tokenizer.training.render_evaluator import RenderEvaluator

VOCAB = GPLVocabulary(max_coord_level=6)
ARCS_INST = ARCS(max_level=6)
train_ds = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=500, max_seq_len=64, seed=123)
val_ds = SyntheticSVGDataset(VOCAB, ARCS_INST, n_samples=100, max_seq_len=64, seed=456)
coll = SVGCollator(VOCAB, max_seq_len=64)
ev = RenderEvaluator()

res = {False: [], True: []}
for seed in (0, 1, 2):
    torch.manual_seed(seed)
    tl = DataLoader(train_ds, batch_size=32, collate_fn=coll, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    vl = DataLoader(val_ds, batch_size=32, collate_fn=coll)
    cfg = GPLTransformerConfig(d_model=128, d_type=16, d_coord=32, n_heads=4,
                               n_layers=4, d_ff=512, max_seq_len=64, dropout=0.1,
                               use_hmn_init=True, hmn_version=2)
    model = GPLTransformer(VOCAB, cfg)
    GPLTrainer(model, TrainingConfig(epochs=15, batch_size=32, learning_rate=1e-3,
                                     checkpoint_dir=f"/tmp/gpl_e4_{seed}", save_every=100,
                                     eval_every=1, patience=100)).train(tl, vl, verbose=False)
    for constrained in (False, True):
        gen = GPLGenerator(model, VOCAB, ARCS_INST, constrained=constrained)
        samples = [gen.generate_unconditional(max_len=64) for _ in range(100)]
        m = ev.evaluate(samples)
        res[constrained].append(m)
        print(f"s{seed} constrained={constrained}: valid={m['grammar_valid']:.0%} "
              f"render={m['renderable']:.0%} non_degen={m['non_degenerate']:.0%} "
              f"ink={m['mean_ink']:.3f} div={m['diversity']:.3f}", flush=True)

for c in (False, True):
    for k in ("grammar_valid", "non_degenerate", "diversity"):
        v = [m[k] for m in res[c]]
        print(f"constrained={c} {k}: {statistics.mean(v):.3f}±{statistics.stdev(v):.3f}")
