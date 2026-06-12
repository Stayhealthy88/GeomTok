# Geometry Is the Better Substrate: Tokenizing Vector Graphics for Sample-Efficient Generation

*Working title — for ACL/EMNLP/CVPR-style submission. All numbers from this repository's experiments (`.research/results_*.txt`), reported post adversarial review.*

---

## Abstract

Large language models tokenize vector graphics (SVG) the way they tokenize prose, shredding a coordinate such as `150.5` into characters that carry no spatial meaning. We ask a narrow, falsifiable question: **does it matter, downstream, what unit a model reads geometry in?** We introduce **GeomTok**, a geometry-native tokenizer that maps SVG to a small vocabulary of primitive tokens (commands, recognized shapes, and fixed-point coordinates), together with **GeomTok-Eval**, a render-based intrinsic evaluation protocol that is decoupled from any generator. On a 2,682-icon real-world benchmark, GeomTok parses and round-trips 100% of inputs and compresses 3.54× over a domain-trained text tokenizer at matched 5,561 vocabulary. Our central finding is a controlled counterexample to a common assumption: **better compression does not imply better downstream modeling.** Training an identical 2.5M-parameter transformer under the tokenizer-swap protocol, geometric primitive tokens yield the best held-out modeling of real icons (576 ± 4 vs 775 ± 6 bits/icon for text BPE, a 26% reduction at ~28σ), while learned BPE merges that compress 1.8× *more* model *worse* (666 ± 2 bits) and, when generating, drop from 84% to 35% renderable output. A text tokenizer generates 0% renderable SVG. We further show that unconstrained learned merges over primitive tokens beat HiVG-style structure-constrained merges (172 vs 236 tokens/icon at matched budget). We release the protocol, baselines, and a set of honest negative results — including a retracted adaptive-quadtree scheme and a retracted embedding-initialization claim — as a methodological contribution.

---

## 1. Introduction

Vector graphics underlie nearly every digital interface: icons, logos, UI components, charts, and maps. Yet generative models remain unreliable at producing them — circles fail to close, rectangles drift off-grid, coordinates land in the wrong place. A widely cited cause is **tokenization**: a coordinate `150.5` is fragmented by a text tokenizer into `1`, `50`, `.`, `5`, destroying the fact that it denotes a point in space (Xing et al., LLM4SVG, 2024).

Recent work attacks this with geometry-aware tokenization — HiVG (2026) merges geometric *segment* tokens by frequency; OmniSVG (2025) folds each `(x,y)` into a single grid token; StrokeNUWA (2024) learns a VQ codebook. These works establish that geometry-aware tokens compress SVG and improve fidelity. **What the field has not isolated is the downstream consequence of the tokenization *unit* itself**, under a controlled, generator-decoupled comparison against fair baselines. That gap is the subject of this paper.

We make three contributions:

1. **GeomTok**, a geometry-native tokenizer with a fixed-point coordinate codec that adds zero vocabulary (§3), achieving 100% parse and 3.54× compression over a matched-vocabulary text tokenizer on real data (§5.1).
2. **The compression-is-not-modelability result** (§5.3): under the tokenizer-swap protocol on real icons, primitive tokens model best, while the most-compressed tokenizer models and generates worst — a clean counterexample to the assumption that intrinsic compression predicts downstream quality.
3. **GeomTok-Eval** (§4), a render-based intrinsic protocol, plus a discipline of **adversarially-verified negative results** (§6) that we argue is necessary for this subfield.

---

## 2. Related Work and Positioning

**Geometry-aware SVG tokenization.** HiVG (arXiv:2604.05072) is the closest prior work: it performs frequency-based merging over geometric segment tokens with uniform coordinate quantization, reporting ~2.7× compression on a 3B VLM. OmniSVG (arXiv:2504.06263) merges `(x,y)` into single grid tokens; LLM4SVG (CVPR 2025) adds semantic tokens to an LLM vocabulary; StrokeNUWA (ICML 2024) learns a VQ-Stroke codebook (~6.9% size, lossy). We do **not** claim to be first to merge over geometric primitives — HiVG and, in the protein domain, GeoBPE (arXiv:2511.11758) precede us. Our novelty is comparative and methodological: the controlled char-BPE / primitive / learned-merge / structure-constrained comparison nobody has run, plus a generator-decoupled protocol.

**Numeric tokenization.** xVal, FoNE, and Number Token Loss study how to encode numbers in transformers. Our fixed-point codec (§3.2) is a pragmatic instance — a quadtree leaf grid reinterpreted as a base-64 numeral system — that we keep deliberately simple to isolate the substrate effect.

**Tokenizer evaluation.** The NAACL'24 "tokenizer-swap on a fixed small model" protocol and "Beyond Text Compression" (ACL'25) establish that compression alone is insufficient evidence. We adopt the swap protocol and contribute its render-based instantiation for graphics.

---

## 3. GeomTok

### 3.1 Pipeline

SVG → **parse** (flatten transforms + group inheritance, normalize viewBox, drop non-rendered `defs`/`clipPath`) → **quantize** (uniform 64×64 coordinate grid; §3.2) → **tokenize** (commands, shapes, coordinates) → optional **grammar-constrained decode** (an FSA guarantees valid SVG). The vocabulary is 5,561 tokens: 5,461 coordinate tokens plus ~100 structural tokens (5 special, 8 command, 4 shape, 4 continuity, 16 curvature, 11 spatial), with the remaining IDs held as reserved ranges for forward-compatible extension.

### 3.2 Scalar fixed-point codec

Earlier internal versions encoded a scalar (radius, gap) as a fake 2-D point `(v,v)` through the coordinate quadtree, which on coarse cells produced up to 17.5px error (a radius of 20 reconstructed as 37.5) and broke repeat counts. We replace this with a **fixed-point codec**: the level-6 grid (64×64 = 4096 cells) is reinterpreted as a base-64 numeral system, so a scalar `v ∈ [0,canvas)` maps to a single coordinate token `qx·64+qy`. This adds **zero vocabulary** and bounds scalar error to `canvas/(4096−1)/2 ≈ 0.04px`. Counts (e.g. REPEAT_N) use an exact integer variant with lossless round-trip up to 4095.

### 3.3 Coordinate scheme: a negative result

We initially used an *adaptive* quadtree (finer cells where curvature is high). On 400 real icons (27,414 coordinates), held at equal token count, a **plain uniform 64×64 grid beat it decisively**: mean coordinate error 1.78px vs 14.75px, render SSIM 0.929 vs 0.846, ink-IoU 0.655 vs 0.284. Curvature-driven splitting starves straight content, which dominates real icons. We report this as a negative result and ship uniform-grid as the default. A density-driven global tree narrowly improved mean error (1.58px) but was fit on the test split and is reported only as such.

---

## 4. GeomTok-Eval

A geometric tokenizer needs intrinsic metrics that (a) work on real, path-only icons, (b) resist gaming, and (c) are computable without a GPU. GeomTok-Eval reports:

- **Parse / round-trip rate** — fraction of a corpus that tokenizes and reconstructs.
- **Token efficiency** — tokens/icon *and* bits/icon (= tokens × log₂ vocab), the latter making cross-vocabulary comparison fair.
- **Value-level fidelity** — per-coordinate L2 error after a tokenize→detokenize round-trip, extended to path coordinates (negative and exponent numbers, positional matching) so it does not return ∞ on path-only icons.
- **Render fidelity** — SSIM and ink-IoU between rasterized original and reconstruction (resvg, 64–256px).

We motivate the protocol with a concrete failure it catches: an earlier "structural score" awarded a perfect 1.000 to random token sequences while scoring ground truth 0.987 — a metric that ranked tokenizers backwards.

---

## 5. Experiments

All experiments use the public **StarVector svg-icons** benchmark. Tokenizer-swap experiments train an identical 2.5M-parameter decoder-only transformer (vanilla, no GeomTok-specific embedding) on each tokenizer's stream — the only variable is the tokenizer (NAACL'24 protocol). The domain text baseline is a SentencePiece BPE trained on SVG text at vocab 5,561 (a *fair* baseline; GPT-4's cl100k, at 100k vocab and not domain-trained, is reported only as a reference).

### 5.1 Tokenizer efficiency (intrinsic), 400 real icons

| Scheme | tokens/icon | bits/icon | Compression vs text BPE |
|---|---|---|---|
| GPT-4 cl100k (reference) | 2021 | 33,580 | 0.46× |
| Domain char-BPE (vocab 5,561) | 935 | 11,636 | 1.00× |
| **GeomTok-L1** | **264** | **3,287** | **3.54×** |
| GeomTok-L1 + learned BPE | 151 | 1,925 | 6.21× |

GeomTok-L1's coordinate fidelity is **1.78px mean / 3.37px max** on a 300px canvas (within the analytic bound), with render SSIM **0.929** — the honest fidelity cost of the 3.54× compression. Hand-crafted shape tokens (L2) fire on **0%** of real icons and add nothing.

### 5.2 Learned vs structure-constrained merges (intrinsic)

Applying an identical greedy BPE to three substrates at a matched merge budget (train 1,200 / test 400):

| Budget | char-BPE | L1 unconstrained | L1 boundary-constrained | L1 HiVG-faithful |
|---|---|---|---|---|
| 0 | 4893 | 264 | 264 | 264 |
| 1000 | 824 | **172** | 222 | 236 |

bits/icon @1000: char 7245 ≫ L1 1687 < boundary 1890 < HiVG 1934 — identical ranking. **Unconstrained learned merges over primitive tokens beat HiVG-style structure-constrained merges by 27% (172 vs 236)**: the constraint forbids cross-command repeated patterns that unconstrained BPE exploits. (L1 bits/icon saturate by ~100 merges; we report bits alongside tokens to avoid overstating merge gains.)

### 5.3 Downstream: the compression–modelability split

Identical 2.5M model, real icons (1,045 train / 213 test), tokenizer swapped, 3 seeds, held-out bits/icon (lower = better):

| Arm | tokens/icon | held-out bits/icon |
|---|---|---|
| char-BPE | 159 | 775 ± 6 |
| L1 + learned BPE | 59 | 666 ± 2 |
| **GeomTok-L1** | 104 | **576 ± 4** |

Two findings, both far outside seed noise (separations of ~28σ and ~20σ respectively):
1. **Geometric primitive tokens are the best downstream substrate** — 26% better modeling than a domain text tokenizer (576 vs 775, ~28σ).
2. **Compression does not predict modelability** — L1+BPE compresses 1.8× more than L1 (59 vs 104 tokens) yet models *worse* (666 vs 576, ~20σ), because merges raise per-token entropy (11.3 vs 5.6 bits/token). The most-compressed tokenizer is not the best to learn from.

### 5.4 Downstream: generation quality

Generating 200 samples per arm, rendering, and scoring:

| Arm | renderable | FID-lite ↓ | diversity ↑ | novelty ↑ |
|---|---|---|---|---|
| char-BPE | **0%** | — | — | — |
| GeomTok-L1 | **84%** | 0.197 | 0.892 | 0.840 |
| L1 + learned BPE | 35% | 0.194 | 0.920 | 0.848 |

A text tokenizer generates **0% renderable SVG** — geometric tokens are *necessary*, not merely convenient. L1 is **2.4× more reliable** than L1+BPE (84% vs 35%): aggressive merges that win on compression cripple generation validity. On renderable samples FID is comparable; no mode collapse (diversity ≈0.9), no memorization (novelty ≈0.84).

---

## 6. Negative Results and Rigor

We retract three claims that an earlier version of this work asserted, each falsified by a control we ran:

- **"7.6× compression vs a text tokenizer"** — a strawman against GPT-4's 100k-vocab cl100k. At matched 5,561 vocabulary the honest figure is **3.54×** (bits/icon identical).
- **"Geometry-aware embedding initialization accelerates training"** — a matched-norm control (random vectors rescaled to unit row-norm, *zero* geometric structure) closes the entire gap: random (default norm ~11.3) val 4.51 → random-unit-norm **3.40** ≈ HMN 3.47. The effect was a weight-tying embedding-*norm* artifact, not geometry.
- **"Adaptive quadtree coordinates"** — beaten by a uniform grid on real data (§3.3).

We also note that a saturated validity metric (100% after a definition change) has no discriminative power, and that hand-crafted L2/L3 macros fire on ~0–15% of real icons. We argue this discipline — competing baselines, matched budgets, controls for confounds, and reporting what dies — is what this subfield, awash in self-defined metrics on toy data, most needs.

---

## 7. Limitations

Experiments are CPU-scale: a 2.5M model on ~1k icons, monochrome path icons, held-out NLL and a lightweight FID rather than large-model human-aligned generation. The downstream claims are scoped to the small-model, sample-efficient regime; we do not claim they transfer unchanged to 3–8B VLM scale. Color/style and full SVG features (gradients, text, filters) are out of scope. The fixed-point codec overloads a token's meaning (position vs scalar vs count), an embedding confound we have not measured.

---

## 8. Conclusion

The unit a model reads geometry in is not a free choice. On real vector graphics, geometric primitive tokens model and generate better than text — and, against intuition, better than the more-compressed learned-merge variant. Compression is not the objective; the *right substrate* is. We release GeomTok and GeomTok-Eval to let the field measure this directly.

---

### Reproducibility

All numbers are produced by scripts in `.research/` against the public StarVector svg-icons benchmark; raw outputs are archived in `.research/results_*.txt`. The package installs as `geomtok` (`pip install -e .`); tests run via `pytest` (191 assertions). Key scripts: `exp_f2_baseline_table.py` (§5.1), `exp_A_hivg_vs_learned.py` (§5.2), `f5_run.py` (§5.3), `f5_gen.py` (§5.4), `exp_init_norm_confound.py` (§6), `exp_e3_*` (§3.3).
