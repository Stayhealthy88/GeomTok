# Compression Is Not Modelability: A Controlled Study of Tokenization Units for Vector Graphics

*Working title — for ACL/EMNLP submission (Findings or main-short track; see §10). Backup title: "Does the Tokenization Unit Matter? A Controlled Study of Geometry-Native SVG Tokenization." All numbers from this repository's experiments (`.research/results_*.txt`), reported post adversarial review.*

---

## Abstract

Large language models tokenize vector graphics (SVG) the way they tokenize prose, shredding a coordinate like `150.5` into characters that carry no spatial meaning. We ask a narrow, falsifiable question: does the *unit* a model reads geometry in change downstream modeling? We introduce **GeomTok**, a geometry-native SVG tokenizer (commands, shapes, and a zero-vocabulary fixed-point coordinate codec), and **GeomTok-Eval**, a render-based intrinsic *tokenizer* protocol decoupled from any generator. Under a controlled tokenizer-swap on an identical small transformer, geometric primitive tokens model held-out icons 18–26% better (held-out NLL) than a domain-trained text BPE — a result that replicates across two corpora (icons and emoji) and *grows* with model capacity over a 10× parameter range — and, under identical plain sampling, a text tokenizer generates 0% renderable SVG versus 84% for GeomTok. Our central result is a controlled counterexample, in the graphics setting, to the assumption that intrinsic compression predicts downstream modeling — a relationship already shown non-monotonic for text by PathPiece (Schmidt et al., 2024): more compression never reliably helps — on icons each added BPE merge monotonically *worsens* modeling (more compressed, higher per-token entropy), and on emoji the most-compressed variant is within noise of the least. We release the protocol, fair baselines, and a set of adversarially-verified negative results — including retracted claims — as a methodological contribution. We scope all model-level claims to the small-model, sample-efficient regime.

---

## 1. Introduction

Vector graphics underlie nearly every digital interface: icons, logos, UI components, charts, and maps. Yet generative models remain unreliable at producing them — circles fail to close, rectangles drift off-grid, coordinates land in the wrong place. A widely cited cause is **tokenization**: a coordinate `150.5` is fragmented by a text tokenizer into `1`, `50`, `.`, `5`, destroying the fact that it denotes a point in space (Xing et al., LLM4SVG, 2024).

Recent work attacks this with geometry-aware tokenization — HiVG (2026) merges geometric *segment* tokens by frequency; OmniSVG (2025) folds each `(x,y)` into a single grid token; StrokeNUWA (2024) learns a VQ codebook. These works establish that geometry-aware tokens compress SVG and improve fidelity. **What the field has not isolated is the downstream consequence of the tokenization *unit* itself**, under a controlled, generator-decoupled comparison against fair baselines. That gap is the subject of this paper. We treat it as a *tokenizer-evaluation* question in the lineage of the fixed-model tokenizer-swap (Ali et al., Findings of NAACL 2024; Lotz et al., ACL 2025) and the text-domain finding that fewer tokens need not mean better downstream behavior (Schmidt et al., PathPiece, EMNLP 2024).

A note on rigor, up front. An earlier version of this work asserted a 7.6× compression headline, a training benefit from geometry-aware embedding initialization, and an adaptive-quadtree coordinate scheme. Each was falsified by a control we ran — a matched-vocabulary baseline, a matched-embedding-norm baseline, and a real-data comparison, respectively (§6). We report these retractions as part of the contribution; the surviving claims are the ones that survived adversarial verification.

We make three contributions:

1. **GeomTok** — a geometry-native tokenizer whose fixed-point coordinate codec adds zero vocabulary, achieving 100% parse and bounded-error round-trip and 3.54× compression at matched vocabulary (§3, §5.1).
2. **Compression is not modelability** — under a controlled, generator-decoupled tokenizer-swap, primitive tokens model and generate best, while the most-compressed variant models and generates *worst* — a clean counterexample, for vector graphics, to the assumption that intrinsic compression predicts downstream quality (§5.3–5.4).
3. **GeomTok-Eval and a negative-results discipline** — a render-based intrinsic protocol that resists gaming, released alongside adversarially-verified retractions of our own earlier claims (§4, §6).

---

## 2. Related Work and Positioning

**Geometry-aware SVG tokenization.** HiVG (arXiv:2604.05072) is the closest prior work: it performs frequency-based merging over geometric segment tokens with uniform coordinate quantization. OmniSVG (arXiv:2504.06263) merges `(x,y)` into single grid tokens; LLM4SVG (CVPR 2025; arXiv:2412.11102) adds semantic tokens to an LLM vocabulary; StrokeNUWA (ICML 2024; arXiv:2401.17093) learns a VQ-Stroke codebook reaching a ~6.9% compression ratio (lossy); InternSVG (arXiv:2510.11341) unifies SVG understanding/editing/generation with SVG-specific special tokens. We do **not** claim to be first to merge over geometric primitives — HiVG and, in the protein domain, GeoBPE (ICLR 2026; arXiv:2511.11758) precede us. Our novelty is comparative and methodological: to our knowledge the first controlled char-BPE / primitive / learned-merge / structure-constrained comparison on real SVG under a single fixed model, plus a generator-decoupled, render- and value-grounded protocol. A concurrent line takes the opposite route — modeling SVG numbers continuously rather than as discrete tokens (Ogezi et al., "From Tokens to Numbers," arXiv:2602.02820, 2026) — without isolating the tokenization unit; we discuss it as an orthogonal challenger in §7.

**Compression vs. downstream.** That intrinsic compression does not monotonically predict downstream quality is established for *text* by PathPiece (Schmidt et al., EMNLP 2024) and by Lotz et al. (ACL 2025), and the fixed-model tokenizer-swap protocol originates in Ali et al. (Findings of NAACL 2024). We contribute the first *vector-graphics* instance with a render-grounded generation consequence.

**Numeric tokenization.** xVal (arXiv:2310.02989), FoNE (arXiv:2502.09741), and Number Token Loss (ICML 2025; arXiv:2411.02083) study how to encode numbers in transformers. Our fixed-point codec (§3.2) is a deliberately simple instance — a quadtree leaf grid reinterpreted as a base-64 numeral system — chosen to isolate the substrate effect, not to advance numeric encoding.

**Name disambiguation.** "GeomTok" is unrelated to *GeoToken* (Ghasemi et al., arXiv:2511.01082), which tokenizes raster images into S2 *geographic* cells for image geolocalization; we operate on SVG geometry primitives. Render-based generator-agnostic SVG metrics also exist for generated *code structure* (Zhu et al., arXiv:2604.08809, 2026); GeomTok-Eval instead targets the *tokenizer* and adds value-level fidelity and cross-vocabulary bits/icon (§4).

---

## 3. Method: GeomTok

### 3.1 Pipeline

SVG → **parse** (flatten transforms + group inheritance, normalize viewBox, drop non-rendered `defs`/`clipPath`) → **quantize** (uniform 64×64 coordinate grid; §3.2) → **tokenize** (commands, shapes, coordinates) → optional **grammar-constrained decode** (an FSA guarantees valid SVG; used only when a downstream application wants a hard validity guarantee — it is *not* used in any experiment below). We adopt the uniform grid as the shipped default; the comparison against an adaptive scheme that justifies this choice is reported with the other controls in §6.

The vocabulary is 5,561 tokens: **5,461 coordinate tokens** plus ~100 structural tokens (5 special, 8 command, 4 shape, 4 continuity, 16 curvature, 11 spatial), with the remaining IDs held as reserved ranges for forward-compatible extension. The 5,461 coordinate tokens cover quadtree levels 0–6; the level-6 leaf grid is 64×64 = 4,096 cells (the deepest level), which the codec below reuses as a numeral system.

### 3.2 Scalar fixed-point codec

A scalar such as a radius or gap must be encoded without inflating the vocabulary. We reinterpret the level-6 grid (64×64 = 4096 cells) as a base-64 numeral system, so a scalar `v ∈ [0,canvas)` maps to a single coordinate token `qx·64+qy`. This adds **zero vocabulary** and bounds scalar error to `canvas/(4096−1)/2 ≈ 0.04px`. Counts (e.g. a repeat count) use an exact integer variant with lossless round-trip up to 4095. (An earlier `(v,v)` encoding through the 2-D quadtree produced up to 17.5px error and broke repeat counts; full account in §6.)

### 3.3 Token levels

L1 emits geometric primitives (commands + coordinates) and is the default. L2 applies learned (frequency-based) merges over L1; L3 adds spatial-relation tokens. As §5.3 shows that more aggressive merging is *not* downstream-optimal, **L1 is the default and compression is opt-in**, documented as a trade-off rather than a free win.

---

## 4. GeomTok-Eval

A geometric tokenizer needs intrinsic metrics that (a) work on real, path-only icons, (b) resist gaming, and (c) are computable without a GPU. GeomTok-Eval reports:

- **Parse / round-trip rate** — fraction of a corpus that tokenizes and reconstructs.
- **Token efficiency** — tokens/icon *and* **encoding bits/icon** (= tokens × log₂ vocab), the latter making cross-vocabulary comparison fair. (We distinguish this *intrinsic* encoding-bits/icon from the *held-out NLL* bits/icon of §5.3, which is a trained-model quantity.)
- **Value-level fidelity** — per-coordinate L2 error after a tokenize→detokenize round-trip, extended to path coordinates (negative and exponent numbers, positional matching) so it does not return ∞ on path-only icons.
- **Render fidelity** — SSIM and ink-IoU between rasterized original and reconstruction (resvg, 64–256px).

We motivate the protocol with a concrete failure it catches: an earlier "structural score" awarded a perfect 1.000 to random token sequences while scoring ground truth 0.987 — a metric that ranked tokenizers backwards. GeomTok-Eval is *generator-decoupled* (it scores the tokenizer's reconstruction, not a generator's output), which is what makes random-sequence gaming impossible.

---

## 5. Experiments

All experiments use the public **StarVector svg-icons** benchmark (monochrome, path-centric). Sections 5–6 evaluate the tokenizer of §3 under the protocol of §4. Tokenizer-swap experiments train an identical 2.5M-parameter decoder-only transformer (a vanilla model with no GeomTok-specific embedding) on each tokenizer's stream — the only variable is the tokenizer (Ali et al., 2024). The domain text baseline is a SentencePiece BPE trained on SVG text at vocab 5,561 (a *fair* baseline; GPT-4's cl100k, at 100k vocab and not domain-trained, is reported only as a reference). Crucially, generation (§5.4) uses **identical plain top-k sampling for every arm with no grammar constraint**, so the geometric arm receives no FSA-validity advantage.

> **Figure 1** (`assets/fig1_pipeline.svg`). Pipeline: SVG → parse/flatten → quantize → tokenize → (optional FSA decode), with a coordinate `150.5` shredded by text BPE (4 fragments) vs one GeomTok coordinate token, and the same-icon token-stream contrast.
> **Figure 2** (`assets/fig2_compression_modelability.svg`). Compression–modelability scatter: x = tokens/icon (intrinsic compression), y = held-out NLL bits/icon (modelability) — the non-monotonic relationship (L1+BPE more compressed yet worse; the L1 budget-sweep trend overlaid).
> **Figure 3** (`assets/fig3_render_panel.svg`, §6). Qualitative render panel: original vs uniform-L6 vs adaptive-quadtree round-trip, per-icon SSIM.
> **Figure 4** (`assets/fig4_gen_samples.svg`, §5.4). Generated samples: GeomTok-L1 (48/60 renderable) vs text BPE (0/60), identical plain sampling.

### 5.1 Tokenizer efficiency (intrinsic), 400 real icons

| Scheme | tokens/icon | encoding bits/icon | Compression vs text BPE |
|---|---|---|---|
| GPT-4 cl100k (reference) | 2021 | 33,580 | 0.46× |
| Domain char-BPE (vocab 5,561) | 935 | 11,636 | 1.00× |
| **GeomTok-L1** | **264** | **3,287** | **3.54×** |
| GeomTok-L1 + learned BPE | 151 | 1,925 | 6.21× |

GeomTok-L1's coordinate fidelity is **1.78px mean / 3.37px max** on a 300px canvas (within the analytic bound), with render SSIM **0.929** — the fidelity cost of the 3.54× compression (the round-trip is geometric and bounded-error, not lossless). Hand-crafted shape tokens (L2) fire on **0%** of real icons and add nothing.

### 5.2 Learned vs structure-constrained merges (intrinsic)

Applying an identical greedy BPE to three substrates at a matched merge budget (train 1,200 / test 400):

| Budget | char-BPE | L1 unconstrained | L1 boundary-constrained | L1 HiVG-faithful |
|---|---|---|---|---|
| 0 | 4893 | 264 | 264 | 264 |
| 1000 | 824 | **172** | 222 | 236 |

encoding bits/icon @1000: char 7245 ≫ L1 1687 < boundary 1890 < HiVG 1934 — identical ranking. **At matched budget, unconstrained learned merges over primitive tokens use 27% fewer tokens than HiVG-style structure-constrained merges (172 vs 236)**: the constraint forbids cross-command repeated patterns that unconstrained BPE exploits. (L1 encoding-bits/icon saturate by ~100 merges; we report bits alongside tokens to avoid overstating merge gains.) This is an intrinsic-compression comparison at fixed budget, not a claim of downstream superiority for L1+BPE — see §5.3.

### 5.3 Downstream: the compression–modelability split

Identical 2.5M model, real icons (1,045 train / 213 test), tokenizer swapped, 3 seeds, **held-out NLL bits/icon** (lower = better):

| Arm | tokens/icon | held-out NLL bits/icon (mean ± std, n=3) |
|---|---|---|
| char-BPE | 159 | 775 ± 6 |
| L1 + learned BPE | 59 | 666 ± 2 |
| **GeomTok-L1** | 104 | **576 ± 4** |

Two findings; both gaps are large relative to seed variance (std across the 3 seeds is 2–6 bits, against gaps of 90–199 bits):
1. **Geometric primitive tokens are the best downstream substrate in this regime** — 26% better modeling than a domain text tokenizer (576 vs 775).
2. **Compression does not reliably predict modelability** — L1+BPE compresses 1.8× more than L1 (59 vs 104 tokens) yet models *worse* (666 vs 576), because merges raise per-token entropy (11.3 vs 5.6 bits/token). The most-compressed tokenizer is not the best to learn from. We strengthen this with a budget sweep, a capacity trend, and a second corpus below, and present it as a counterexample to the assumption that compression predicts downstream quality — not a universal law of the reverse.

**Budget sweep (the counterexample is a curve, not a point).** Sweeping the merge budget N applied to L1 on svg-icons (1,000 train / 200 test, n=2): as N rises, tokens/icon falls monotonically (105→71→70→68→65 at N=0/250/500/1k/2k) while held-out NLL rises monotonically (608→635→649→668→680). Every added merge compresses more and models worse; pure L1 (N=0) is the downstream optimum. Visualized in Figure 2 (`assets/fig2_compression_modelability.svg`).

**Capacity trend (the substrate gap does not vanish with scale).** Running the L1-vs-L1+BPE(1k) swap at four model sizes from 0.9M to 9.3M parameters, L1 wins at every size and the gap *grows* with capacity:

| params | L1 NLL | L1+BPE NLL | gap |
|---|---|---|---|
| 0.9M | 754 | 788 | 35 |
| 2.2M | 708 | 755 | 47 |
| 4.6M | 661 | 723 | 62 |
| 9.3M | 634 | 696 | 62 |

The L1>L1+BPE advantage is not a small-model artifact across this 10× parameter range; whether it persists at 100M–1B is future work (§10).

**Second corpus (the primary claim replicates; the compression direction is corpus-dependent).** Repeating the swap on **svg-emoji** (colored, 401 train / 48 test, n=3), the *primary* substrate result replicates strongly — both geometric arms crush text BPE: L1 1117 ± 27 and L1+BPE 1086 ± 44 vs char-BPE 1356 ± 5 (geometric tokens 18% better than text). The *specific* compression direction, however, does not transfer: on emoji L1+BPE (1086) is statistically tied with L1 (1117) (gap 31 < combined std ~52), rather than worse as on svg-icons. We report this honestly: across a budget sweep and two corpora, more compression *never reliably helps* — it monotonically hurts on svg-icons and is within noise on svg-emoji — which is the evidence for "compression does not reliably predict modelability." A genuine predictor would show a consistent monotone relationship; we observe none. The robust, two-corpus claim is the substrate result (geometry ≫ text); the compression–modelability relationship is non-predictive rather than reliably inverted.

### 5.4 Downstream: generation quality

Generating 200 samples per arm under **identical plain top-k sampling (no FSA for any arm)**, rendering, and scoring:

| Arm | renderable | FID-lite ↓ | diversity ↑ | novelty ↑ |
|---|---|---|---|---|
| char-BPE | **0%** | — | — | — |
| GeomTok-L1 | **84%** | 0.197 | 0.892 | 0.840 |
| L1 + learned BPE | 35% | 0.194 | 0.920 | 0.848 |

In this small-model regime, a text tokenizer yields **0% renderable SVG** under the same sampling that gives GeomTok 84% (Figure 4) — geometric tokens appear *required* for renderable output here, not merely more efficient (we scope this to the 2.5M-parameter setting; large VLMs with massive SVG corpora may close the gap). L1 yields **2.4× more renderable samples than L1+BPE (84% vs 35%)**: aggressive merges that win on compression halve generation validity. On renderable samples FID is comparable; no mode collapse (diversity ≈0.9), no memorization (novelty ≈0.84). The diversity/novelty metrics are mask-based and we treat them as sanity checks, not headline numbers.

---

## 6. Controls and Retracted Claims

We subjected each component to an adversarial control; three plausible claims did not survive, and we report them so the field does not repeat them.

- **"7.6× compression vs a text tokenizer" → retracted to 3.54×.** The 7.6× figure compared against GPT-4's cl100k (100k vocab, not domain-trained) — a strawman. At matched 5,561 vocabulary the honest figure is **3.54×** (encoding bits/icon identical).
- **"Geometry-aware embedding initialization accelerates training" → retracted.** A matched-norm control — random vectors rescaled to unit row-norm, with *zero* geometric structure — closes the entire gap: random (default norm ~11.3) val 4.51 → random-unit-norm **3.40** ≈ geometry-structured 3.47. The apparent benefit was a weight-tying embedding-*norm* artifact, not geometry. (We highlight this control in §1 as the cleanest example of the discipline: a subtle confound masquerading as a geometry effect.)
- **"Adaptive quadtree coordinates" → retracted in favor of a uniform grid.** On 400 real icons (27,414 coordinates), at equal token count, a plain uniform 64×64 grid beat a curvature-adaptive quadtree decisively: mean coordinate error **1.78px vs 14.75px**, render SSIM **0.929 vs 0.846**, ink-IoU **0.655 vs 0.284** (Figure 3 shows the visual difference). Curvature-driven splitting starves straight content, which dominates real icons. A density-driven global tree narrowly improved mean error (1.58px) but was fit on the test split and is reported only as such.

We also note that a saturated validity metric (100% after a definition change) has no discriminative power, and that hand-crafted L2/L3 macros fire on ~0–15% of real icons. We argue this discipline — competing baselines, matched budgets, controls for confounds, and reporting what dies — is what this subfield, which often relies on self-defined metrics and small synthetic corpora, most needs.

---

## 7. Limitations and the Continuous-Coordinate Challenge

Experiments are CPU-scale: models up to 9.3M parameters on ~0.4–1k icons, path-centric icons (and colored emoji as a second corpus), held-out NLL and a lightweight FID rather than large-model human-aligned generation. We therefore scope the model-level claims (§5.3–5.4) to the small-model, sample-efficient regime; we do not claim they transfer unchanged to 3–8B VLM scale. We note, however, that the substrate gap does *not* shrink over the 0.9M–9.3M range we tested (it grows; §5.3), which weakens the "small-model artifact" reading, though confirming persistence at 100M–1B remains the most important follow-up (§10). Color/style and full SVG features (gradients, text, filters) are out of scope. The fixed-point codec overloads a token's meaning (position vs scalar vs count), an embedding confound we have not measured.

The sharpest challenge to our premise is the *continuous-coordinate* camp (Ogezi et al., 2026): if coordinates are modeled as continuous values rather than discrete tokens, the entire notion of a tokenization "unit" dissolves. We test it directly. On an identical backbone and data, holding the coordinate *input* continuous for both arms and varying only the output head — discrete cell-classification (4096-way) vs continuous (x,y) regression — we measure held-out, teacher-forced next-coordinate prediction error in pixels. The **discrete head is 18% more accurate** (mean 35.7px vs 43.3px; median 29.7 vs 38.9; n=2 seeds): the categorical formulation is more robust than regression in this small-data regime. (Absolute errors are large because predicting the *next* coordinate from context is inherently hard — this is a predictive comparison of the two output parameterizations, not the 1.78px reconstruction floor of §5.1.) For our setting, dropping discrete tokens does not help; we make no claim about the large-data regime where Ogezi et al. operate.

---

## 8. Conclusion

The unit a model reads geometry in is not a free choice. On real vector graphics, in the small-model regime, geometric primitive tokens model and generate better than text — and, against intuition, better than the more-compressed learned-merge variant. We do not claim compression is irrelevant; we exhibit a controlled counterexample to the assumption that it predicts downstream quality, with an information-theoretic mechanism (merges trade token count for per-token entropy). We release GeomTok and GeomTok-Eval, with our negative-results ledger, to let the field measure this directly.

---

## 9. Reproducibility

All numbers are produced by scripts in `.research/` against the public StarVector svg-icons benchmark; raw outputs are archived in `.research/results_*.txt`. The package installs as `geomtok` (`pip install -e .`); tests run via `pytest` (191 assertions). Key scripts: `exp_f2_baseline_table.py` (§5.1), `exp_A_hivg_vs_learned.py` (§5.2), `f5_run.py` (§5.3 main), `exp_E1_budget_sweep.py` / `exp_E2_capacity.py` / `exp_E3_second_corpus.py` (§5.3 sweep/capacity/second-corpus), `f5_gen.py` (§5.4), `exp_E4_continuous.py` (§7 continuous-coordinate arm), `exp_init_norm_confound.py` (§6), `exp_e3_adaptive_vs_uniform.py` / `exp_e3_density_tree.py` (§6). Figure 2: `fig2_compression_modelability.py` → `assets/fig2_compression_modelability.svg`.

A full reproducibility record — pinned environment, dataset revision hashes, the exact per-experiment split manifests (`.research/splits/`), the model-config table, seeds, per-experiment wall-clock, determinism notes, and the experiment→script→result map — is in [REPRO.md](REPRO.md) with [`requirements-repro.txt`](requirements-repro.txt). Datasets are pinned by revision (`starvector/svg-icons`@`0dfc1bee7132`, `starvector/svg-emoji`@`a4209d1752ba`); the rasterizer is pinned (`resvg-py==0.3.2`) because render metrics depend on it. The package is released under **Apache-2.0**, which the methodological/protocol framing requires. The full paper reproduces on a laptop CPU (Apple M4) in a few hours; no GPU required.

---

## 10. Submission Plan (not for camera-ready)

Per adversarial review, the framing/wording/citation fixes plus the three experiments now in §5.3 (budget sweep, capacity trend, second corpus — all **done**) target an **ACL/EMNLP main-short** bar. Remaining items, in priority order:

1. ✅ **Merge-budget sweep, downstream** (§5.3) — the counterexample is now a monotonic curve on svg-icons.
2. ✅ **Capacity trend** (§5.3) — the substrate gap holds and grows over 0.9M–9.3M params.
3. ✅ **Second corpus** (§5.3) — the substrate claim replicates on svg-emoji; the compression direction is reported as corpus-dependent.
4. ✅ **Continuous-regression arm** (§7) — on the same backbone, a discrete coordinate-token head predicts held-out coordinates 18% more accurately than a continuous regression head (35.7 vs 43.3px); dropping discrete tokens does not help in this regime.
5. **Nice-to-have:** κ-curvature/continuity token ablation; coordinate-error tail (>2px perceptible rate); StrokeNUWA VQ point on the compression–fidelity scatter; small human forced-choice eval; Figures 1, 3, 4 (Figure 2 done).
