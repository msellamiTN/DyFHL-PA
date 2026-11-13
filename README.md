# DyFHL-PA

Cross-domain access-control policy anomaly analysis without pooling policies.

Each ABAC rule is a hyperedge over attribute-value vertices and per-attribute wildcard
vertices. Domains exchange only vertex-level sums of hashed hyperedge signatures,
partitioned by collaboration scope and effect. A decoder that is disjunctive inside an
attribute and conjunctive across attributes recovers four cross-domain anomaly types:
conflict, redundancy, irrelevancy and coverage gaps. Intra-domain anomalies are decided
exactly by each domain on its own rules.

Two decoders are provided:

- **DyFHL-PA-R** — fixed threshold rule, no training, one-sided error.
- **DyFHL-PA-L** — small per-task networks trained with FedAvg on a label-free
  pseudo-domain task.

## Quick start

```bash
pip install -r requirements.txt
bash scripts/fetch_data.sh                 # Xu–Stoller case studies + Amazon access data
python -m pytest -q tests                  # oracle, soundness and delta-exactness tests
python scripts/run_experiments.py --quick  # 2 seeds; drop --quick for the paper's suite
python scripts/leakage.py                  # rule exposure vs sketch size + healthcare case study
python scripts/multihash.py                # check: several smaller sketches vs one large sketch
python scripts/running_example.py          # the paper's running example, checked against the oracle
python scripts/diagrams.py                 # architecture figure
python scripts/make_figures.py             # tables, figures and number macros -> paper/generated
python scripts/check_figures.py            # verify every figure against the Elsevier artwork rules
streamlit run app/streamlit_app.py         # interactive explorer
```

Colab: open `notebooks/DyFHL_PA_colab.ipynb`, set `REPO` in the first cell and run all.

## Layout

| Path | Content |
|---|---|
| `dyfhl/policy.py` | rule model, three rule sources, multi-domain world generator, exact pooled oracle |
| `dyfhl/model.py` | signatures, vertex aggregation, conjunctive evidence, rule and learned decoders |
| `dyfhl/fed.py` | label-free pseudo-domain training, FedAvg, context exchange, evaluation |
| `dyfhl/dynamic.py` | exact delta maintenance under policy churn |
| `dyfhl/baselines.py` | random forests with audited labels; count-completion and clique-expansion variants |
| `dyfhl/viz.py` | hypergraph projections of rules and anomalies (Plotly) |
| `scripts/run_experiments.py` | every experiment in the paper (`main ablation sketch noise scale stream`) |
| `scripts/make_figures.py` | produces every table, figure and in-text number from `results/*.json` |
| `app/streamlit_app.py` | dataset picker, detection, baselines, policy stream, hypergraph views |
| `dyfhl/figstyle.py` | one figure style: 90/140/190 mm widths, Arial-compatible embedded fonts, >= 7 pt text |
| `scripts/running_example.py` | the six-rule running example; every number in the worked examples |
| `scripts/check_figures.py` | width, font embedding/family, minimum printed text size, line weights |
| `paper/` | LaTeX source (elsarticle), compiled PDF, `response.pdf` (pre-submission review and responses) |
| `paper/figures_submission/` | every figure as vector PDF and EPS plus 1000 dpi TIFF for production |
| `results/` | raw result files of the reported runs |

## Datasets

- `agentic` — synthetic multi-domain environment (agents, MCP servers, tools, applications,
  legacy systems, humans).
- `xu_stoller` — rules generated from the real user/resource attribute data of the
  Xu–Stoller ABAC case studies (GPLv3, fetched, not redistributed here).
- `amazon` — rules generated from records of the Amazon employee access data.

Domains, collaboration scopes and anomaly injection are produced by the generator in all
three cases; labels always come from the exact pooled oracle on the final rule sets.

## Figures and the journal's artwork rules

All figures are drawn by `dyfhl/figstyle.py` at Elsevier's 140 mm (1.5-column) width with
Liberation Sans (metric-compatible with Arial) embedded as TrueType, text of at least 7 pt and
subscripts of at least 6 pt at printed size, and line weights between 0.3 and 1.3 pt.
`scripts/check_figures.py` checks this for every figure and exits non-zero on a violation.

## What the results say, briefly

The rule decoder recovers most of the pooled analysis and never misses a conflict or
redundancy when no noise is added. The learned decoder ranks findings better but does not
decide better. Accuracy depends on the sketch size and drops quickly under additive noise,
so this design does not give differential privacy at small epsilon. See the paper's
discussion for the leakage analysis.

## Licence

Code: MIT. The fetched datasets keep their own licences.
