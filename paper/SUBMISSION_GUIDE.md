# Where to submit GeomTok, and how

_Last updated 2026-06-16. Deadlines are AoE (anywhere-on-Earth)._

## TL;DR

Submit the **anonymous** build (`make review` → `acl_latex.pdf`) to the **ACL
Rolling Review (ARR) August 2026 cycle** by **Aug 3, 2026**, and **commit to
EACL 2027**.

| | |
|---|---|
| **Primary venue** | **EACL 2027** — Athens, Greece, Mar 9–14, 2027 |
| **Route** | ACL Rolling Review (ARR), August 2026 cycle |
| **ARR submission deadline** | **Aug 3, 2026** (AoE) |
| **Reviewer reg. (all authors)** | ~Aug 5, 2026 (≈2 days after; miss = desk reject) |
| **Commitment to EACL 2027** | ~Oct 11, 2026 |
| **Paper class** | long paper (8 content pages) — we are at ≈7.5, fits |

Why EACL 2027: the May cycle (EMNLP 2026) already closed; the **next** ARR cycle
is August, and EACL 2027 is the conference that commits from it. It is a strong,
appropriately-scoped home for a controlled, methodological vector-graphics study.

### Two parallel options (not exclusive)
- **A co-located EMNLP 2026 workshop** (Budapest, Oct 2026): workshop CFPs usually
  close ~Aug 2026 and are a great, faster home for a methodological / negative-
  results short paper. Watch the EMNLP 2026 workshop list.
- **arXiv now**: maximizes visibility/citability. ⚠️ But see "preprint decision".

## Step by step (ARR → EACL 2027)

1. **OpenReview account** (openreview.net) with your real name + a complete
   profile (ARR matches reviewers from your profile; a thin profile risks poor
   reviewers). New profiles take a few days to activate — **do this first**.
2. **Anonymous code mirror.** Create one at <https://anonymous.4open.science>
   pointing at `github.com/Stayhealthy88/GeomTok`. Put the resulting URL in the
   paper's abstract footnote (the `[review]` build already prints a placeholder
   `anonymous.4open.science/r/geomtok` — update it to the real anon URL).
3. **Build the anonymous PDF**: `make review` → `acl_latex.pdf` (line-numbered,
   "Anonymous ACL submission"). Double-check no author name appears.
4. **Create the ARR submission** on OpenReview for the August cycle. Upload
   `acl_latex.pdf`. Fill in title, abstract, and the **Responsible NLP Checklist**
   (draft in [`RESPONSIBLE_NLP_CHECKLIST.md`](RESPONSIBLE_NLP_CHECKLIST.md)).
5. **Register as a reviewer** (you, as an author) by the reviewer-registration
   deadline. This is mandatory.
6. **Declare the target** if the cycle asks for a binding conference choice
   (EACL 2027).
7. **Author response** window (~1 week, mid-cycle): respond to reviewers.
8. **Commit** the reviewed paper to EACL 2027 by the commitment deadline.

## The preprint decision (affects arXiv timing)

Since Feb 2024, ARR has **no anonymity period** — a public arXiv + public repo
does **not** disqualify you. But there is a trade-off:

- **Stay anonymous until meta-reviews** → eligible for the anonymous-submission
  incentives (award eligibility, priority on borderline decisions). Hold the arXiv
  post. _Recommended for a first submission._
- **Preprint now** → immediate visibility, but forfeit those incentives, and never
  choose the binding "no non-anonymous preprint" option if you do this.

When you do preprint: `make preprint` → `acl_latex_preprint.pdf` (non-anonymous,
shows the author + the real repo URL). Primary arXiv category `cs.CL`, cross-list
`cs.CV`/`cs.GR`.

## Decisions only you can make
- [ ] **Affiliation + contact email** for the author block (currently "Independent
      Researcher, igotthepower0128@gmail.com" — confirm or change).
- [ ] **Venue target**: EACL 2027 (recommended) vs an EMNLP 2026 workshop vs wait
      for a later ARR cycle (NAACL/ACL 2027).
- [ ] **Preprint timing**: hold for the anonymity incentive vs arXiv now.
- [ ] **AI-assistance disclosure** wording (Responsible NLP Checklist A4).

## Actions only you can perform (I cannot do these for you)
- Create the OpenReview account + ORCID.
- Create the anonymous.4open.science mirror (needs your GitHub login).
- Submit on OpenReview and register as a reviewer.

**Sources:** [EACL 2027](https://2027.eacl.org/) · [EACL 2027 CFP](https://2027.eacl.org/calls/papers/) · [ARR dates](https://aclrollingreview.org/dates) · [ARR CFP](https://aclrollingreview.org/cfp) · [ARR anonymity policy](https://aclrollingreview.org/anonymity/)
