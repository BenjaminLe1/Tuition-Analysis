# Purdue Tuition Analysis

Regression model that predicts a university's out-of-state tuition from institutional features, applied to Purdue University to evaluate whether tuition is priced in line with peers and which levers (instructional expenditure, graduation rate) could justify an increase.

## Executive summary

**Purpose.** Purdue University has held out-of-state tuition flat for over a decade — one of the longest tuition freezes of any major public research university. This project quantifies the opportunity cost of upholding that freeze.

**Goals.**
1. Estimate the opportunity cost of the freeze — how much revenue Purdue is leaving on the table relative to peer institutions.
2. Identify defensible avenues to increasing revenue — the institutional features that most strongly drive peer tuition.
3. Recommend where Purdue should focus effort if a future tuition increase is on the table.

**Results.** Built a predictive tuition model on ~500 US colleges that **uncovered a ~25% pricing discrepancy** between Purdue's 2018 out-of-state tuition and the model's peer-benchmarked prediction — Purdue was priced roughly a quarter below what comparable institutions charged.

## Headline results

Numbers below reflect the current code (post-leakage-fix, RandomizedSearch-tuned RF, log-target experiment, permutation importance).

| Model | CV MAE (mean ± std) | Test MAE | Test R² |
|---|---|---|---|
| Naive mean (baseline) | — | ~$8.5k | 0.00 |
| **Linear Regression** | $5,434 ± $536 | **$5,666.70** | **0.7030** |
| Random Forest | $5,355 ± $404 | $5,597.95 | 0.6845 |

**Model selection.** After removing the `state_mean` target leakage, LR and RF are within noise of each other on MAE, but LR has higher R² and is simpler and more interpretable. A log-transform experiment on the target lost on both metrics and was not used. LR is saved as the primary model in [models/tuition_model.pkl](models/tuition_model.pkl).

## Data

Two public datasets merged on college name

- [raw_data/College_Data.csv](raw_data/College_Data.csv) — ISLR `College` dataset (777 US colleges, institutional features: apps, acceptance, faculty, expenditure, grad rate, etc.).
- [raw_data/tuition_cost.csv](raw_data/tuition_cost.csv) — 2018 tuition by institution (in-state / out-of-state, public/private, degree length).

### Caveats (important)

- **Temporal mismatch.** The predictor features come from the ISLR dataset, while the target is 2018 tuition. The model assumes the characteristics captured in the ISLR dataset are similar enough in 2018 to be predictive.
- **Merge.** Fuzzy name matching drops unmatched rows, and rows with duplicated matches are removed from the left side before merging. Final N is smaller than either source.
- **Single fixed split** (`random_state=67`, `test_size=0.3`). Cross-validation is used for hyperparameter selection; final test evaluation is one split.
- **Scope.** The cost side of the "suggestions" (what it costs Purdue to move grad rate or expenditure) is out of scope.

## Repository structure

```
├── raw_data/                    # raw data
├── data/                        # cleaned data
├── notebooks/
│   ├── Data_Wrangling.ipynb     # merge + clean
│   ├── EDA.ipynb                # feature selection + visualizations
│   ├── Modeling.ipynb           # split, grid search, final model, Purdue analysis
│   └── lib/sb_utils.py          # save helper
├── models/tuition_model.pkl     # trained Linear Regression pipeline
└── reports/
    ├── Capstone_Final_Report.pdf
    └── model_metrics.txt
```

## Model features

`Private`, `Top10perc`, `Top25perc`, `Room.Board`, `PhD`, `S.F.Ratio`, `perc.alumni`, `Expend`, `Grad.Rate`, `state_mean`.

## What I'd do differently

1. **Target leakage.** Computing `state_mean` on the full dataset leaked each row's state's test tuition into training.

2. **Model complexity rarely wins on small datasets.** With ~500 training rows and 10 features, a tuned RF, or other complex models does not necessarily beat a plain LR. LR ended up having a higher test R² than all other more complex models.

3. **Default RF feature importance.** Impurity-based `feature_importances_` over-weights continuous predictors like `Expend` and `state_mean`. Switching to `permutation_importance` on the test set produced a more honest ranking.

4. **Temporal mismatch.** The predictor features come from the ISLR dataset, while the target is 2018 tuition. The model assumes the characteristics captured in the ISLR dataset are similar enough in 2018 to be predictive.

5. **Hyperparameter grids proportionate to the problem.** The original RF grid had way too many iterations. A 60-iteration `RandomizedSearchCV` found essentially the same best region.