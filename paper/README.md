# GeomTok — ACL/ARR submission package

LaTeX source for *"Compression Is Not Modelability: A Controlled Study of
Tokenization Units for Vector Graphics"*, converted from [`../PAPER.md`](../PAPER.md).

## Build

**Overleaf (recommended):** upload this whole `paper/` folder, set the compiler to
**pdfLaTeX**, main document `acl_latex.tex`. It compiles as-is.

**Local (tectonic, no TeX install needed):**

```bash
brew install tectonic        # one-time
tectonic acl_latex.tex       # -> acl_latex.pdf (runs bibtex automatically)
```

**Local (traditional TeX Live / MacTeX):**

```bash
pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex
```

Verified: compiles to **9 pages** (≈7.5 content + references + a 1-paragraph
reproducibility appendix) with tectonic 0.16.9. Content fits the 8-page
long-paper limit.

## Files

| File | What |
|---|---|
| `acl_latex.tex` | the paper (in `[review]` = anonymous, line-numbered mode) |
| `references.bib` | 16 references, **each verified to exist** (adversarial check) |
| `figures/fig{1,2,3,4}.pdf` | vector PDFs, converted from `../assets/fig*.svg` via `rsvg-convert` |
| `acl.sty`, `acl_natbib.bst` | official ACL style files (acl-org/acl-style-files) |
| `acl_template_reference.tex` | the upstream ACL template, kept for reference |

## Before you submit — open items

These are deliberately **not** yet done; they are the next pass:

1. **Mode switch.** `\usepackage[review]{acl}` → `[final]` for camera-ready, or
   `[preprint]` for a non-anonymous arXiv version. Fill in `\author{...}` for those.
2. **Anonymization.** The abstract footnote points to
   `https://anonymous.4open.science/r/geomtok` — create that anonymized mirror of
   the repo (the real `github.com/Stayhealthy88/GeomTok` link de-anonymizes you).
   Review is double-blind.
3. **Three 2026 preprints** (`hivg2026`, `ogezi2026cnm`, `zhu2026svgmetrics`) are
   cited as `@misc`/arXiv (correct — they have no peer-reviewed venue). Re-verify
   their arXiv IDs on arxiv.org on submission day.
4. **Page trim if needed.** If a reviewer-track wants ≤8 content pages, the
   second-corpus paragraph and one of the retraction bullets compress easily.
5. **Responsible NLP Checklist** is filled in on OpenReview at submission, not here.

## Provenance note

The bibliography was assembled by an adversarial verification pass: all 16
citations resolve to real papers (none fabricated). Two had method-names used as
titles, now corrected in the `.bib` (`geobpe2026`, `ntl2025`). One in-text factual
error was fixed in both this `.tex` and `../PAPER.md`: HiVG uses
geometry-constrained segment tokens + hierarchical mean-noise init, **not**
frequency-based merging with uniform quantization.
