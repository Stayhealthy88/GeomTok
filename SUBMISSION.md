# GeomTok Paper — Submission Tracker

3-reviewer adversarial pass (hostile area-chair · related-work/positioning · writing/venue). Consensus: the science is honest and the **compression≠modelability counterexample + protocol + negative-results ledger** is the durable contribution. The paper was reframed away from the "geometry is the better substrate" universal toward the counterexample, which turns toy-scale from a fatal weakness into a feature (a counterexample needs only one regime).

## Verdict & venue

| Path | Status | Gating work |
|---|---|---|
| **EMNLP/ACL Findings · tokenization/efficiency workshop** | **Ready now** (after this revision) | Wording/citation/structure fixes — **done** in PAPER.md |
| **ACL/EMNLP main short** | **E1–E4 + Figs 1–4 + camera-ready artifact done**; ready to draft submission | — |
| **ACL/EMNLP main long · CVPR** | Out of scope | Requires scale-up (≥100M, real generator, color SVG, human eval) — a second paper |

**Recommendation:** target **EMNLP Findings** (the work is a tokenizer-evaluation contribution in the NAACL'24 swap lineage, not a vision paper). Run E1+E2 for a credible main-short attempt.

## Revision applied (this pass) — done

- [x] Retitled to lead with the counterexample ("Compression Is Not Modelability…"); backup neutral title noted
- [x] Abstract tightened (~205w), leads with the falsifiable question then the counterexample
- [x] Overclaims fixed: "necessary"→"required, scoped to 2.5M"; dropped "~28σ/20σ" → std-vs-gap; "round-trip"→"bounded-error round-trip"; "nobody has run"→"to our knowledge first under a single fixed model"; "better substrate" universal removed from title; §8 conclusion scoped
- [x] §5.4 FSA confound defused: stated explicitly that **all arms use identical plain top-k, no FSA** (verified in `f5_gen.py`) — the 0% vs 84% is apples-to-apples
- [x] Must-add citations: PathPiece (EMNLP'24, text precedent for fewer-tokens≠better), CNM "From Tokens to Numbers" (2602.02820, concurrent continuous-coord challenger), InternSVG, LOO structural-metrics (2604.08809), GeoToken disambiguation
- [x] Citation wording: StrokeNUWA "6.9% compression ratio (lossy)"; GeoBPE→ICLR'26; NTL→ICML'25; NAACL'24 = Findings (Ali et al.)
- [x] Structure: §3 is method-only; all retractions consolidated into §6 ("Controls and Retracted Claims"), reframed as controls-we-ran not confessions
- [x] Two "bits/icon" disambiguated: **encoding bits/icon** (intrinsic, §5.1) vs **held-out NLL bits/icon** (§5.3)
- [x] Reconciled 5,461 coord vocab vs 4,096 grid (level-6 leaf = deepest); retraction count = 3
- [x] Contributions reworded to one load-bearing claim each

## Experiments for main-short (CPU-feasible) — todo

- [x] **E1. Merge-budget sweep, downstream** — DONE. budget 0→2000: tokens 105→65, NLL 608→680 (monotonic). Pure L1 is downstream-optimal. `results_E1_budget_sweep.txt`.
- [x] **E2. Capacity trend** — DONE. 0.9M→9.3M params: L1 wins at every size, gap widens then plateaus 35→47→62→62 (single seed/size; trend, not precision). NOT a small-model artifact. `results_E2_capacity.txt`.
- [x] **E3. Second corpus (svg-emoji)** — DONE. Substrate claim replicates (L1/L1+BPE 1117/1086 ≪ char 1356); compression direction is corpus-dependent (tied on emoji) — reported honestly. `results_E3_second_corpus.txt`.
- [x] **E4. Continuous-regression arm** — DONE. discrete coord-token head 18% more accurate than continuous regression (35.7 vs 43.3px mean) on same backbone; CNM does not help in this regime. `results_E4_continuous.txt`.
- [x] **E5. κ/continuity token ablation** — DONE (3 seed). Markers = 24.2% of L1, derivable (57/57 byte-identical round-trip), yet stripping *improves* held-out NLL 687.2→602.4 (−84.8, ~12%) and shortens 120.8→91.6 tok/icon. Lean L1 recommended; full L1 kept as conservative as-shipped config. `results_ablate_curv_cont.txt`.
- [ ] **E6. (optional)** coord-error >2px tail; StrokeNUWA VQ scatter point; n≈20 human forced-choice.

## Figures to produce — todo

- [x] **Fig.1** Pipeline + token-stream contrast — DONE (`assets/fig1_pipeline.svg`).
- [x] **Fig.2** Compression–modelability scatter — DONE (`assets/fig2_compression_modelability.svg`, rendered).
- [x] **Fig.3** Render panel original/uniform/adaptive + SSIM — DONE (`assets/fig3_render_panel.svg`).
- [x] **Fig.4** Generated samples GeomTok (48/60 renderable) vs text BPE (0/60) — DONE (`assets/fig4_gen_samples.svg`).

## Camera-ready artifact — todo

- [x] HF dataset revisions pinned (svg-icons@0dfc1bee7132, svg-emoji@a4209d1752ba); split manifests in `.research/splits/` (make_splits.py)
- [x] `requirements-repro.txt` (Python 3.14.5, torch 2.12, resvg-py 0.3.2, etc.)
- [x] seeds + 2.5M config table → REPRO.md §3 (d=192, 4 layers, AdamW 3e-4, tied head)
- [x] hardware (Apple M4 CPU) + wall-clock + determinism → REPRO.md §5–6
- [x] **License = Apache-2.0** (LICENSE file present)
- [x] name check done (GeomTok/GeomTok-Eval clear in SVG domain, 2026-06-12); RE-CHECK arXiv listing immediately before submission

## Spot-checks before submission

- [x] §2 avoids HiVG's exact 2.7×/3B numbers (no precise unverified claim) — kept generic
- [x] verified: 7.6× appears only in §1 disclosure + §6 retraction; no σ-multipliers / 'nobody has run' anywhere

## Distinctiveness verdict — deep-research (2026-06-14, 18 claims at 3-0/2-1)

Each of the 5 core claims checked against primary sources. **Net: the headline survives; one honest reframe + one new must-cite.**

| Claim | Verdict | Evidence |
|---|---|---|
| 1. compression≠modelability in vector graphics | **domain-first, NOT absolute-first** | text: PathPiece (EMNLP'24), Lotz (ACL'25); **raster images: arXiv:2412.16326 (NeurIPS'25 Spotlight)** — capacity-dependent, VQ-latents, no SVG. No SVG/tokenizer-swap instance exists → ours is first in vector graphics. |
| 2. geometric substrate via held-constant-backbone tokenizer-swap + held-out NLL | **genuinely distinctive** | HiVG does end-to-end comparison only, **no controlled swap, no perplexity/NLL**; InternSVG adds special tokens, no swap. No SVG precedent. |
| 3. GeomTok-Eval = generator-decoupled tokenizer protocol | **distinctive** | LOO (2604.08809) scores *generators*' output structure (purity/coverage/...); InternSVG SArena is generation-coupled. No public tokenizer-decoupled protocol with parse/bits-per-icon/value-fidelity/render set. |
| 4. learned unconstrained merges beat HiVG structure-constrained | **distinctive comparison** | HiVG confirmed to use geometry-constrained segment merges (segment boundaries + command arity); the direct head-to-head is unpublished — we run it. |
| 5. uniform 64×64 + scalar fixed-point codec, **zero added vocab** | **distinctive engineering** | HiVG adds 2,384 coord tokens (784×784); OmniSVG ~40k (200×200); CNM drops coords. Zero-added-vocab scalar codec is unmatched. GeoBPE = 3D protein, not SVG. |

**Action taken:** added arXiv:2412.16326 to §2 as the raster-image precedent (the one prior visual-domain compression-vs-generation result), with the honest contrast that our SVG result runs the opposite direction (more compression never reliably helps). **Name check:** no 2026-Q2 SVG paper preempting the headline surfaced; GeomTok/GeomTok-Eval still appear unclaimed in the SVG domain (re-confirm arXiv listing at submission). **Framing rule:** claim 1 = "first in vector graphics" (never "first ever"); claims 2–5 = genuinely distinctive.
