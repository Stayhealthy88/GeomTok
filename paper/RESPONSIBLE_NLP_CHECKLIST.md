# Responsible NLP Checklist — draft answers

Copy these into the OpenReview form at submission time. ARR/ACL groups the
checklist into sections A–E. Answers below are pre-filled from the paper and
[REPRO.md](../REPRO.md); ⚠️ marks the few that need your confirmation.

## A. General
- **A1. Limitations discussed?** **Yes** — dedicated *Limitations* section (CPU
  scale; small-model regime; color/style and full SVG features out of scope; the
  fixed-point codec's position/scalar/count overloading is an unmeasured confound;
  continuous-coordinate caveat).
- **A2. Potential risks discussed?** **N/A / low risk** — a tokenizer and
  evaluation protocol for monochrome vector icons; no generative misuse surface,
  no human data. State "no significant risks identified."
- **A3. Abstract & intro state the claims and scope?** **Yes** — both scope the
  claims to the small-model regime explicitly.
- **A4. ⚠️ AI assistants used in writing/research?** **Yes — must disclose.** This
  manuscript and its experiments were prepared with the assistance of an AI coding
  assistant (Claude Code). Recommended wording: *"The authors used an AI assistant
  for code scaffolding, LaTeX preparation, and citation verification; all
  scientific claims, experiments, and final text were checked by the authors."*
  → **You decide the exact phrasing.**

## B. Scientific artifacts
- **B1. Cite the creators of artifacts you use?** **Yes** — StarVector svg-icons /
  svg-emoji (dataset), `resvg-py` (rasterizer), SentencePiece (BPE baseline).
  Ensure each is cited; add a `\citep` for StarVector if a reviewer expects it.
- **B2. ⚠️ License of the artifacts you use?** Our code/protocol: **Apache-2.0**.
  → **Confirm the StarVector `svg-icons`/`svg-emoji` dataset license** permits
  research use and state it (it is a public Hugging Face dataset; verify its card).
- **B3. Intended use consistent with the artifact's terms?** **Yes** — research
  evaluation, the datasets' intended use.
- **B4. PII / offensive content in the data?** **No** — monochrome icon glyphs and
  colored emoji; no personal data.
- **B5. Documentation of artifacts you release?** **Yes** — README, API_STATUS,
  REPRO with environment, dataset revisions, seeds, split manifests.
- **B6. Relevant statistics (e.g., #examples, splits)?** **Yes** — every split
  size is reported (1,045/213; 1,000/200; 401/48; 1,200/400; 980/205) in the paper
  and REPRO.

## C. Computational experiments
- **C1. Report #parameters and compute budget?** **Yes** — 0.9M–9.3M params; CPU
  only (Apple M4); full paper reproduces in a few hours, no GPU.
- **C2. Experimental setup / hyperparameters?** **Yes** — model config table,
  optimizer, lr, epochs per experiment in REPRO §3.
- **C3. Descriptive statistics, multiple runs?** **Yes** — 2–3 seeds with mean±std
  on the main panels; single-seed trend explicitly flagged where used (capacity).
- **C4. Software packages + versions?** **Yes** — pinned in `requirements-repro.txt`
  (resvg-py==0.3.2, etc.); package installs as `geomtok`.

## D. Human annotators / participants
- **N/A** — no human subjects, no crowdsourcing, no annotation study. (The
  "nice-to-have" human forced-choice eval is *not* in this submission.)

## E. AI assistants (if separate from A4)
- Disclosed under A4. If the form has a dedicated field, repeat the A4 statement.

---

### Open items for you (⚠️)
1. **A4** — approve the AI-assistance disclosure wording.
2. **B2** — confirm and state the StarVector dataset license.
