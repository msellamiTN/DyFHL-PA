# Status — version 2 (complete, 2026-11-14)

Target journal: **Computers & Security** (Elsevier). Manuscript: `paper/main.tex` → `paper/main.pdf`.
Pre-submission review (three simulated referees) and point-by-point responses: `paper/response.tex` → `paper/response.pdf`.

## Done in version 2
- Running example (three units, six rules, two requests) used throughout; every number in it is computed by
  `scripts/running_example.py` and checked against the exact oracle.
- Lemmas 1–3 (matching, overlap, subsumption as incidence conditions) and Proposition (reduction of hierarchies,
  ranges and negation to set membership).
- One section per anomaly type (conflict, redundancy, gap, irrelevancy): decision rule, propositions with
  proofs, algorithm, worked example, figure; summary table of guarantees.
- Privacy section rewritten: exposure defined, predicted and measured; choice of B; sensitivity and
  composition; deviating members. New exposure-vs-accuracy figure.
- Natural-anomaly study (no injection), healthcare case study, several-sketches check, B = 4096 row in the
  main table.
- **Hash corrected**: the v1 multiplicative hash was not uniform (optimistic results). Replaced by keyed
  splitmix64; all experiments re-run. Irrelevancy decision corrected to the exact per-attribute definition.
- All figures redrawn to the Elsevier artwork rules (140 mm, Liberation Sans embedded, >= 7 pt text and
  >= 6 pt subscripts at printed size, 0.3–1.3 pt lines); `scripts/check_figures.py` verifies them.
  Production files (PDF, EPS, 1000 dpi TIFF) in `paper/figures_submission/`.

## Completed on 2026-11-14
- Ablation (81/81 rows) and noise (72/72 rows) studies finished on the corrected hash.
- Ablation and noise paragraphs rewritten to the final numbers; response letter updated to match.
- Both PDFs compile without errors or overfull lines; no pending values remain; all 11 figures pass
  `scripts/check_figures.py`; 6 unit tests pass.

## Open decisions for the author
1. Title: the learned decoder is optional (see `paper/response.pdf`, Part III); consider a title that
   names the method rather than "learning".
2. Author list, affiliations, CRediT statement and the generative-AI declaration.
3. Final proofreading of `paper/main.pdf` before submission to Computers & Security.

## Main results after the re-run (5 seeds, macro-F1)
| | Agentic | XS-ABAC | AMZ-Access |
|---|---|---|---|
| DyFHL-PA-R, B = 512 | 0.802 | 0.680 | 0.808 |
| DyFHL-PA-R, B = 4096 | 0.971 | 0.887 | 0.947 |
| DyFHL-PA-L, B = 512 | 0.782 | 0.655 | 0.676 |
| best non-pooling baseline (RF with 20 % audited labels) | 0.515 | 0.756 | 0.581 |

Shared-scope rules exposed to the other members of a scope: about 66 % at B = 512, 95 % at B = 4096.
