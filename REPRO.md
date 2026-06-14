# GeomTok — Reproducibility Appendix

Camera-ready reproducibility record for the experiments in [PAPER.md](PAPER.md). Every headline number is produced by a script in `.research/` and archived verbatim in `.research/results_*.txt`.

## 1. Environment

| | |
|---|---|
| Python | 3.14.5 |
| Hardware | Apple M4, 10 cores, **CPU-only** (`DEV="cpu"` in all scripts) |
| Install | `python -m venv .venv && .venv/bin/pip install -r requirements-repro.txt && .venv/bin/pip install -e .` |
| Package | imports as `geomtok` (v1.0.0); tests `pytest` (191 assertions) |

Pinned dependencies in [`requirements-repro.txt`](requirements-repro.txt). **The rasterizer version is load-bearing**: render metrics (SSIM, ink-IoU, renderable rate) depend on `resvg-py==0.3.2`; a different rasterizer will shift absolute SSIM by small amounts (rankings are robust).

## 2. Data

Public Hugging Face datasets, pinned by revision so `df.head(N)` row order is stable:

| Dataset | Revision (sha) | Used by |
|---|---|---|
| `starvector/svg-icons` | `0dfc1bee7132` | §5.1–5.4, §6, E1, E2, A |
| `starvector/svg-emoji` | `a4209d1752ba` | E3 (second corpus) |

Each experiment applies `df.head(N)` then filters by parse-success and L1-token length. The **exact icons used** (by `Filename`) are checked in under [`.research/splits/`](.research/splits/) (regenerate with `.research/make_splits.py`). Split sizes after filtering, one manifest per loader: `f5_icons` (F5 §5.1/5.3/5.4, `head(n·2)`, max_len 160) = **1,045 / 213**; `e12_icons` (E1, E2; `head(n·3)`, max_len 160) = **1,000 / 200**; `e3_emoji` (E3; `head(n·3)`, max_len 160) = **401 / 48**; `expA_icons` (§5.2; `head(N)`, parse-success only) = **1,200 / 400**.

## 3. Model configuration (tokenizer-swap experiments)

A tokenizer-agnostic vanilla decoder-only transformer (`.research/f5_run.py::VanillaLM`), identical across arms — the only variable is the tokenizer.

| hyperparameter | value (default 2.5M model) |
|---|---|
| d_model | 192 |
| layers | 4 |
| heads | 6 (`d//32`) |
| ffn | 4× d_model |
| context | 256 (sequences truncated to max_len 160–200) |
| norm | Pre-LN, GELU |
| LM head | tied to embedding |
| optimizer | AdamW, lr 3e-4, grad-clip 1.0 |
| epochs | 25 (F5 main), 22 (E1, E3), 15 (E2), 12 (E4) |
| seeds | {0,1,2} F5/E3; {0,1} E1/E2/E4 |
| total params | 2,502,464 (default); E2 sweeps 0.9M–9.3M (d∈{96,160,256,384}) |

Seeds are set with `torch.manual_seed(seed); np.random.seed(seed)` before each model init + DataLoader. The GeomTok package's own decoder (`geomtok.training.GPLTransformer`) is 1,529,104 params after the v0.7 dead-cross-attention removal; it is used only in the synthetic-data tests, not in the paper's tokenizer-swap experiments.

## 4. Experiment → script → result map

| Paper § | Script (`.research/`) | Archived result |
|---|---|---|
| §5.1 efficiency table | `exp_f2_baseline_table.py` | `results_f2_baseline_table.txt` |
| §5.2 learned vs structure-constrained | `exp_A_hivg_vs_learned.py` | `results_A_hivg_vs_learned.txt` (split 1200/400 verified via manifest + script, not printed in the archived table) |
| §5.3 main swap (3 seed) | `f5_run.py` | `results_f5_3seed.txt` |
| §5.3 E1 budget sweep | `exp_E1_budget_sweep.py` | `results_E1_budget_sweep.txt` |
| §5.3 E2 capacity trend | `exp_E2_capacity.py` | `results_E2_capacity.txt` |
| §5.3 E3 second corpus | `exp_E3_second_corpus.py` | `results_E3_second_corpus.txt` |
| §5.4 generation quality | `f5_gen.py` | `results_f5_gen.txt` |
| §6 HMN norm confound | `exp_init_norm_confound.py` | `results_f4_norm_confound.txt` |
| §6 adaptive vs uniform | `exp_e3_adaptive_vs_uniform.py` | `results_e3_adaptive_vs_uniform.txt` |
| §6 density tree | `exp_e3_density_tree.py` | `results_e3_density_tree.txt` |
| §7 E4 continuous arm | `exp_E4_continuous.py` | `results_E4_continuous.txt` |
| §5.1 aux-marker ablation (E5) | `ablate_curv_cont.py` | `results_ablate_curv_cont.txt` |
| Fig 1 | `fig1_pipeline.py` | `assets/fig1_pipeline.svg` |
| Fig 2 | `fig2_compression_modelability.py` | `assets/fig2_compression_modelability.svg` |
| Fig 3 | `fig_assets.py` + `fig3_render_panel.py` | `assets/fig3_render_panel.svg` |
| Fig 4 | `fig4_gen_samples.py` + `fig4_panel.py` | `assets/fig4_gen_samples.svg` |

Run any script with `PYTHONPATH=$(pwd) .venv/bin/python .research/<script>.py`.

## 5. Wall-clock (Apple M4, CPU)

Approximate, single run: §5.1/§5.2 minutes; §5.3 main (9 trainings) ~10 min; E1 (10 trainings) ~8 min; E2 (16 trainings, up to 9.3M) ~30 min; E3 (9 trainings) ~8 min; E4 (4 trainings + per-position loss) ~25 min; F5-gen / Fig4 (train + 200 generations + render) ~15 min. The full paper reproduces on a laptop CPU in a few hours; no GPU required.

## 6. Determinism notes

- Token encode/decode is deterministic given `tokenizer_version` + `vocab_id`; the GeomTok vocab is a fixed 5,561-entry table.
- Training is seed-deterministic on CPU up to PyTorch's documented float-reduction nondeterminism; reported means use 2–3 seeds and we report std.
- The domain BPE baselines are trained once with SentencePiece (`/tmp/svg_bpe_5561.model`, `/tmp/l1_bpe.model`) and reused; retraining SentencePiece is deterministic given the same input file.
- Render metrics are deterministic given the pinned `resvg-py`.

## 7. License & naming

Released under **Apache-2.0** ([LICENSE](LICENSE)) — required for the protocol/methodology framing (a closed artifact cannot serve as a community evaluation standard). The names **GeomTok** and **GeomTok-Eval** were checked against arXiv/web (2026-06-12) and are unclaimed in the SVG/vector-graphics domain; *GeoToken* (geolocalization) and *GeoBPE* (protein) are distinct domains and disambiguated in §2.

**Pre-submission name re-check (2026-06-15).** Re-verified via a 5-angle search sweep (arXiv, web, GitHub, PyPI, HF papers) plus an adversarial "prove it is taken" pass and a skeptical synthesis. Verdict: **no third-party collision** — the only internet occurrence of "GeomTok"/"GeomTok-Eval" is the authors' own repository (`github.com/Stayhealthy88/GeomTok`); no arXiv/Papers-with-Code paper and no PyPI `geomtok` package use the name. Nearest names are distinct in spelling and/or domain: *GeoToken* (image geolocalization), *GeoBPE* (protein structure), *GloTok* / *CompTok* (raster visual tokenizers), *GeoEval* (geometry word-problem QA). The §2 disambiguation of *GeoToken* and *GeoBPE* remains accurate and sufficient. **No rename required.** Re-confirm the arXiv listing once more on the day of submission as a final formality.
