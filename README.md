# Data Bias Mitigation under Coverage Constraints & The Price of Fairness

Code for the FAccT 2026 paper:

> Bruno Scarone, Alfredo Viola, Renée J. Miller. *Data Bias Mitigation under Coverage Constraints & The Price of Fairness*. ACM FAccT 2026, Montreal, QC, Canada. [https://doi.org/10.1145/3805689.3812359](https://doi.org/10.1145/3805689.3812359)

## Overview

We extend the bias mitigation framework of [Scarone et al. (AIES 2025)](https://ojs.aaai.org/index.php/AIES/article/view/36712) to incorporate **coverage constraints** that enforce sufficient representation across intersectional demographic groups. The framework:

- Characterizes all exact solutions to the bias mitigation problem and derives tight approximation error bounds
- Derives sample complexity bounds (via Serfling's inequality) for estimating external source distributions
- Formulates bias mitigation as an **integer linear program (ILP)** that finds globally optimal solutions for any fairness tolerance ε
- Characterizes the **price of fairness**: the minimum data modification cost as a function of fairness tolerance
- Empirically demonstrates that bias mitigation via the ILP preserves predictive accuracy across multiple classifiers

## Requirements

- Python 3.9+
- [Gurobi](https://www.gurobi.com/) (with a valid license — free academic licenses are available)
- See `requirements.txt` for Python dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The paper uses three publicly available datasets:

| Dataset | n | Groups | Labels | Sensitive Attributes |
|---------|---|--------|--------|----------------------|
| [COMPAS](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) | 60,798 | 4 | 3 | sex, race (binary) |
| [Adult](https://doi.org/10.24432/C5XW20) | 48,842 | 4 | 2 | sex, race (binary) |
| [Default](https://doi.org/10.24432/C55S3H) | 30,000 | 8 | 2 | sex, education |

Raw CSVs are included in `data/`. To re-download them:

```bash
python download_datasets.py
```

## Repository Structure

```
├── data/                        # Raw datasets and sketches
│   ├── adult/
│   ├── compas/
│   └── default_credit/
├── bias_coverage_ilp.py         # Core ILP formulation (Section 5)
├── bias_coverage_utils.py       # Group/coverage utility functions
├── bias_closed_form.py          # Closed-form mitigation algorithm (Section 4)
├── bias_closed_form_solution.py
├── sample_complexity_bounds.py  # Serfling/Hoeffding sample size bounds (Section 4.4)
├── dataframe_sketches.py        # Contingency table sketch operations
├── config_datasets.py           # Dataset and sketch configurations
├── config_*.py                  # Experiment configurations
├── ml_evaluation.py             # ML evaluation pipeline (Section 6)
├── ml_models.py                 # Classifier definitions
├── ml_preprocessing.py          # Preprocessing pipeline
├── run_*.py                     # Experiment entry points
├── run_experiments.sh           # Main experiment runner
├── sampling.py                  # Sampling utilities
└── solution_io.py               # Solution serialization
```

## Reproducing the Experiments

### Step 1: Generate sketches

```bash
python generate_sketches.py
```

### Step 2: Run ILP experiments

```bash
# All experiments with default objectives
./run_experiments.sh -e "vary_epsilon vary_coverage vary_cost_ratio vary_budget"

# Specific objectives
./run_experiments.sh -e vary_epsilon -o "min_size min_changes"

# With figures
./run_experiments.sh -e vary_epsilon -p
```

Available experiments: `vary_epsilon`, `vary_coverage`, `vary_cost_ratio`, `vary_budget`, `compare_strategies`

Available objectives: `min_size`, `min_changes`, `min_additions`

### Step 3: ML evaluation

ML evaluation runs automatically unless skipped with `-s`. To run it separately:

```bash
python run_ml_eval.py -e vary_epsilon -o min_changes
```

### Closed-form solutions (Section 4)

```bash
python run_closed_form_solutions.py
```

### Sample complexity bounds (Section 4.4)

```bash
python sample_complexity_bounds.py
```

This reproduces the COMPAS sampling experiment: n=1,210 samples (2% of dataset), with all estimation errors below the theoretical 5% bound.

## Core API

The main entry point for the ILP is `bias_coverage_mitigation_ilp` in `bias_coverage_ilp.py`:

```python
from bias_coverage_ilp import bias_coverage_mitigation_ilp
from bias_coverage_utils import get_groups, get_scaled_coverage

# Load your sketch (contingency table)
df_sketch = pd.read_csv("data/compas/compas_Sex_Code_Text_race_binary_ScoreText_sketch.csv")

groups = get_groups(df_sketch, sensitive_attrs=["Sex_Code_Text", "race_binary"])
coverage = get_scaled_coverage(df_sketch, ["Sex_Code_Text", "race_binary"], "ScoreText", scale=0.5)

solution = bias_coverage_mitigation_ilp(
    df_sketch=df_sketch,
    label_attr="ScoreText",
    groups=groups,
    grouplabel_coverage=coverage,
    eps=0.05,               # fairness tolerance
    objective="min_changes" # or "min_size", "min_additions", "min_cost"
)
# solution: {group_label_tuple: delta} where delta > 0 means additions, delta < 0 means deletions
```

## Citation

```bibtex
@inproceedings{scarone2026bias,
  title     = {Data Bias Mitigation under Coverage Constraints \& The Price of Fairness},
  author    = {Scarone, Bruno and Viola, Alfredo and Miller, Ren\'{e}e J.},
  booktitle = {Proceedings of the 2026 ACM Conference on Fairness, Accountability, and Transparency},
  series    = {FAccT '26},
  year      = {2026},
  doi       = {10.1145/3805689.3812359},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

## License

MIT License. See `LICENSE`.