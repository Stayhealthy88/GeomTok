# GeomTok Paper — Submission Tracker

3-reviewer adversarial pass (hostile area-chair · related-work/positioning · writing/venue). Consensus: the science is honest and the **compression≠modelability counterexample + protocol + negative-results ledger** is the durable contribution. The paper was reframed away from the "geometry is the better substrate" universal toward the counterexample, which turns toy-scale from a fatal weakness into a feature (a counterexample needs only one regime).

## Verdict & venue

| Path | Status | Gating work |
|---|---|---|
| **EMNLP/ACL Findings · tokenization/efficiency workshop** | **Ready now** (after this revision) | Wording/citation/structure fixes — **done** in PAPER.md |
| **ACL/EMNLP main short** | **E1–E4 done**; needs Figs 1/3/4 (Fig 2 done) | see below |
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
- [x] **E2. Capacity trend** — DONE. 0.9M→9.3M params: L1 wins at every size, gap grows 35→62. NOT a small-model artifact. `results_E2_capacity.txt`.
- [x] **E3. Second corpus (svg-emoji)** — DONE. Substrate claim replicates (L1/L1+BPE 1117/1086 ≪ char 1356); compression direction is corpus-dependent (tied on emoji) — reported honestly. `results_E3_second_corpus.txt`.
- [x] **E4. Continuous-regression arm** — DONE. discrete coord-token head 18% more accurate than continuous regression (35.7 vs 43.3px mean) on same backbone; CNM does not help in this regime. `results_E4_continuous.txt`.
- [ ] **E5. (optional)** κ/continuity token ablation; coord-error >2px tail; StrokeNUWA VQ scatter point; n≈20 human forced-choice.

## Figures to produce — todo

- [ ] **Fig.1** Pipeline + same-icon char-BPE vs GeomTok token-stream contrast (highest value; conveys whole thesis)
- [x] **Fig.2** Compression–modelability scatter — DONE (`assets/fig2_compression_modelability.svg`, rendered).
- [ ] **Fig.3** Qualitative render panel: original vs GeomTok vs adaptive-quadtree round-trip, captioned SSIM/ink-IoU
- [ ] **Fig.4** Generated-samples grid: GeomTok renderable icons vs text-tokenizer (empty/garbage)

## Camera-ready artifact — todo

- [ ] HF dataset id + revision hash; checked-in split manifests (1,045/213, 1,200/400)
- [ ] `requirements.txt` + Python + **resvg version** (render metrics depend on rasterizer)
- [ ] 3 seeds + seed mechanism; 2.5M model config table (layers/width/ctx/LR/steps/batch)
- [ ] per-experiment hardware + wall-clock; determinism note
- [ ] **License decision** — must be open (Apache-2.0) for the protocol/methodology framing to hold (currently repo notes "proprietary")
- [ ] arXiv final-listing name check before submission (GeomTok/GeomTok-Eval still clear as of 2026-06-12)

## Spot-checks before submission

- [ ] Verify HiVG's "~2.7×/3B VLM" specifics against the PDF (cited precisely in earlier drafts; current §2 avoids the exact numbers — keep it that way unless verified)
- [ ] Confirm no stray retracted number (7.6×, σ-multipliers) leaks outside §6
