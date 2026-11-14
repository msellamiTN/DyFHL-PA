# Status — version 2 (work in progress, 2026-11-14 01:20)

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

## Still to do (resume here)
1. Finish the two interrupted studies (both resume row by row; run each in chunks of < 10 min):
   ```
   DYFHL_THREADS=1 timeout 560 python3 scripts/run_experiments.py ablation   # 80 of 81 rows done (1 left)
   DYFHL_THREADS=1 timeout 560 python3 scripts/run_experiments.py noise      # 37 of 72 rows done (AMZ-Access left)
   python3 scripts/fix_irrelevancy.py && python3 scripts/make_figures.py && python3 scripts/check_figures.py
   ```
2. Rewrite two paragraphs whose wording predates the re-run: **RQ2 ablations** (Section "RQ2: ablations")
   and **Noise** (Section "RQ3", paragraph "Noise"). All numbers are macros and already update; only the
   qualitative sentences must be checked against `paper/generated/tab_abl_*.tex` and `tab_noise.tex`.
3. Recompile (`pdflatex main; bibtex main; pdflatex main ×2; pdflatex response ×2`) and re-read the PDFs.
4. Decide on the title (the learned decoder is optional; see `paper/response.pdf`, Part III).

## Main results after the re-run (5 seeds, macro-F1)
| | Agentic | XS-ABAC | AMZ-Access |
|---|---|---|---|
| DyFHL-PA-R, B = 512 | 0.802 | 0.680 | 0.808 |
| DyFHL-PA-R, B = 4096 | 0.971 | 0.887 | 0.947 |
| DyFHL-PA-L, B = 512 | 0.782 | 0.655 | 0.676 |
| best non-pooling baseline (RF with 20 % audited labels) | 0.515 | 0.756 | 0.581 |

Shared-scope rules exposed to the other members of a scope: about 66 % at B = 512, 95 % at B = 4096.
