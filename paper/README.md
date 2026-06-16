# GeomTok — ACL/ARR submission package

LaTeX source for *"Compression Is Not Modelability: A Controlled Study of
Tokenization Units for Vector Graphics"*, converted from [`../PAPER.md`](../PAPER.md).

## Build

Three modes, one source. The author name + repo URL are **anonymity-aware**: they
auto-hide in `review` and appear only in `preprint`/`final` (driven by the acl
class option — no manual editing).

```bash
brew install tectonic   # one-time (self-contained LaTeX, no sudo)
make review             # -> acl_latex.pdf          ANONYMOUS — submit THIS to ARR
make preprint           # -> acl_latex_preprint.pdf non-anonymous, for arXiv
make final              # -> acl_latex_final.pdf     camera-ready (after acceptance)
```

**Overleaf:** upload the whole `paper/` folder, compiler **pdfLaTeX**, main file
`acl_latex.tex`. For the non-anonymous version on Overleaf, change
`\usepackage[review]{acl}` → `[preprint]`.

Verified: both builds compile to **9 pages** (≈7.5 content + references + a
1-paragraph appendix) with tectonic 0.16.9 — within the 8-page long-paper limit.

## Files

| File | What |
|---|---|
| `acl_latex.tex` | the paper; `[review]`=anonymous default, author auto-hidden |
| `references.bib` | 16 references, **each verified to exist** (adversarial check) |
| `Makefile` | `review` / `preprint` / `final` builds |
| `SUBMISSION_GUIDE.md` | **where to submit + step-by-step method** (EACL 2027 via ARR) |
| `RESPONSIBLE_NLP_CHECKLIST.md` | pre-filled checklist draft for OpenReview |
| `figures/fig{1,2,3,4}.pdf` | vector PDFs, converted from `../assets/fig*.svg` via `rsvg-convert` |
| `acl.sty`, `acl_natbib.bst` | official ACL style files (acl-org/acl-style-files) |
| `acl_template_reference.tex` | the upstream ACL template, kept for reference |

## Status of the pre-submission pass

Done:
- ✅ Author filled (`Gyuwook Byun`), guarded so `review` stays anonymous.
- ✅ Anonymity-aware resource URL + PDF metadata (no author leak in the review PDF).
- ✅ Three 2026 preprints (`hivg2026`, `ogezi2026cnm`, `zhu2026svgmetrics`)
  re-verified live on arXiv (2026-06-16); cited as `@misc` (no peer-reviewed venue).
- ✅ Page budget checked — fits 8-page long-paper limit.
- ✅ Responsible NLP Checklist drafted (`RESPONSIBLE_NLP_CHECKLIST.md`).

You must still do (see `SUBMISSION_GUIDE.md`):
- ⚠️ Create the **anonymous.4open.science** mirror and paste its URL into the
  abstract footnote (placeholder is `anonymous.4open.science/r/geomtok`).
- ⚠️ Confirm **affiliation + email** (currently "Independent Researcher").
- ⚠️ Decide **preprint timing** (hold for anonymity incentive vs arXiv now) and the
  **AI-assistance disclosure** wording.
- Create the OpenReview account + ORCID; submit; register as a reviewer.

## Provenance note

The bibliography was assembled by an adversarial verification pass: all 16
citations resolve to real papers (none fabricated). Two had method-names used as
titles, now corrected in the `.bib` (`geobpe2026`, `ntl2025`). One in-text factual
error was fixed in both this `.tex` and `../PAPER.md`: HiVG uses
geometry-constrained segment tokens + hierarchical mean-noise init, **not**
frequency-based merging with uniform quantization.
