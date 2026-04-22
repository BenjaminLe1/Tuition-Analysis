# Purdue Tuition Analysis

Regression model that predicts a university's out-of-state tuition from institutional features, applied to Purdue University at West Lafayette to evaluate whether tuition is priced in line with peers and which levers (instructional expenditure, graduation rate) could justify an increase.

> **TL;DR.** The original capstone concluded Purdue had roughly \$5k of tuition upside versus peer institutions. After revising the pipeline on review — removing target leakage in a state-mean feature, swapping the tuned Random Forest for a simpler Linear Regression, and adding a bootstrap prediction interval — the conclusion inverts. Purdue's 2018 out-of-state tuition (\$28,794) lands **within \$450 of the model's point estimate (\$29,239)** and comfortably inside an **80% prediction interval of \$28,077–\$30,428**. Tuition is priced roughly in line with peers; the earlier "room to raise" claim was largely an artifact of leakage and overconfident point estimation. See [What changed on review](#what-changed-on-review-and-what-id-do-differently) for the methodology walk-through.

## Headline results

Numbers below reflect the current code (post-leakage-fix, RandomizedSearch-tuned RF, log-target experiment, permutation importance).

| Model | CV MAE (mean ± std) | Test MAE | Test R² |
|---|---|---|---|
| Naive mean (baseline) | — | ~$8.5k | 0.00 |
| **Linear Regression** (SelectKBest k=10) — *saved primary* | $5,434 ± $536 | **$5,666.70** | **0.7030** |
| Random Forest (RandomizedSearch, 60 iters) | $5,355 ± $404 | $5,597.95 | 0.6845 |
| Linear Regression on log-target | — | $6,185.76 | 0.6759 |

Full metrics in [reports/model_metrics.txt](reports/model_metrics.txt). Full write-up in [reports/Capstone_Final_Report.pdf](reports/Capstone_Final_Report.pdf) (note: the report predates the leakage fix and the LR swap).

**Model selection.** After removing the `state_mean` target leakage, LR and RF are within noise of each other on MAE, but LR has higher R² and is simpler and more interpretable. A log-transform experiment on the target lost on both metrics and was not used. LR is saved as the primary model in [models/tuition_model.pkl](models/tuition_model.pkl).

**Purdue finding.** The LR model predicts Purdue's out-of-state tuition at **$29,239** vs actual **$28,794** — within $450. A 500-iteration bootstrap gives an **80% prediction interval of $28,077–$30,428** and a 95% interval of $27,470–$31,111, both of which contain the actual tuition. The conclusion is therefore weaker than the original capstone claim: **Purdue's 2018 tuition sits roughly at the peer-comparable level**, not obviously below it. The sensitivity scenarios in [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb) still quantify what an increase in `Expend` or `Grad.Rate` would be worth, but the starting point is not under-priced.

## Data

Two public datasets merged on college name via fuzzy matching (`thefuzz`, score ≥ 80):

- [raw_data/College_Data.csv](raw_data/College_Data.csv) — ISLR `College` dataset (777 US colleges, institutional features: apps, acceptance, faculty, expenditure, grad rate, etc.).
- [raw_data/tuition_cost.csv](raw_data/tuition_cost.csv) — 2018 tuition by institution (in-state / out-of-state, public/private, degree length).

Target: `out_of_state_tuition` (from the 2018 dataset).

### Caveats (important)

- **Temporal mismatch.** The predictor features come from the ISLR dataset (mid-1990s), while the target is 2018 tuition. The model assumes the institutional characteristics captured in the 1990s still rank-order schools similarly enough in 2018 to be predictive. Absolute tuition levels in 2018 are being predicted from dated features, which is a meaningful limitation.
- **Merge is lossy.** Fuzzy name matching drops unmatched rows, and rows with duplicated matches are removed from the left side before merging. Final N is smaller than either source.
- **Single fixed split** (`random_state=67`, `test_size=0.3`). Cross-validation is used for hyperparameter selection; final test evaluation is one split.
- **Scope.** The cost side of the "suggestions" (what it costs Purdue to move grad rate or expenditure) is out of scope.

## Repository structure

```
├── raw_data/                    # unmodified source CSVs
├── data/                        # cleaned / EDA intermediates
├── notebooks/
│   ├── Data_Wrangling.ipynb     # merge + clean
│   ├── EDA.ipynb                # feature selection + visualizations
│   ├── Modeling.ipynb           # split, grid search, final model, Purdue analysis
│   └── lib/sb_utils.py          # save helper
├── models/tuition_model.pkl     # trained Linear Regression pipeline (v2.0)
└── reports/
    ├── Capstone_Final_Report.pdf
    └── model_metrics.txt
```

## Running

Python 3.13 (tested). The pickled model in [models/tuition_model.pkl](models/tuition_model.pkl) was produced with scikit-learn 1.7.0 — [requirements.txt](requirements.txt) pins the exact versions to avoid unpickling warnings.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Execute the notebooks in order:

1. [notebooks/Data_Wrangling.ipynb](notebooks/Data_Wrangling.ipynb) → writes [data/college_data_cleaned.csv](data/college_data_cleaned.csv)
2. [notebooks/EDA.ipynb](notebooks/EDA.ipynb) → writes [data/college_data_EDA.csv](data/college_data_EDA.csv)
3. [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb) → writes [models/tuition_model.pkl](models/tuition_model.pkl)

## Model features

`Private`, `Top10perc`, `Top25perc`, `Room.Board`, `PhD`, `S.F.Ratio`, `perc.alumni`, `Expend`, `Grad.Rate`, `state_mean`.

## Revision log (interview-review cleanup)

- ~~`state_mean` computed on the full dataset before the train/test split (target leakage).~~ **Fixed** — now target-encoded from training data only inside [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb); states unseen in training fall back to the overall training mean.
- ~~RF feature importance uses biased impurity-based scoring.~~ **Fixed** — replaced with `sklearn.inspection.permutation_importance` on the test set.
- ~~Purdue prediction is a single point estimate with no uncertainty.~~ **Fixed** — 500-iteration bootstrap prediction interval added.
- ~~RF grid is ~2,400 combos × 5-fold (~12k fits), wasteful for this dataset.~~ **Fixed** — switched to `RandomizedSearchCV` with 60 iterations (~300 fits).
- ~~Saved model was RF, but LR has stronger post-leakage-fix metrics.~~ **Fixed** — LR is now the saved primary.
- ~~No R² reported alongside MAE.~~ **Fixed** — both metrics printed for LR and RF on test set.
- ~~No residual or actual-vs-predicted diagnostic plots.~~ **Fixed** — added to [notebooks/Modeling.ipynb](notebooks/Modeling.ipynb) for the chosen LR model.
- Log-transformed target tried and kept as a negative result ($6,186 MAE vs $5,667); left in the notebook for transparency.

## What changed on review, and what I'd do differently

These are the methodology lessons I'd want to talk about in a review or interview setting. Each one is a thing the original capstone got wrong or glossed, and what the data told me when I fixed it.

1. **Target leakage is sneaky when it's engineered into a "helper" feature.** `state_mean` looked like an innocuous geographic aggregate, but computing it on the full dataset leaked each row's state's test tuition into training. The tell: the tuned Random Forest had a ~\$630 lower test MAE than Linear Regression before the fix; after re-encoding `state_mean` on training data only, that gap collapsed to noise (see the [headline table](#headline-results)). Any feature built from the target must be computed fold-wise or post-split — no exceptions.

2. **Model complexity rarely wins on small tabular data.** With ~500 training rows and 10 features, a tuned RF (best params: `n_estimators=100, max_depth=10, min_samples_leaf=4`) does not meaningfully beat a plain LR + `SelectKBest` pipeline. LR has a higher test R², is fully interpretable via coefficients, and is the saved primary. The RF's apparent advantage before the leakage fix was the leakage, not the trees.

3. **Default RF feature importance misled me.** Impurity-based `feature_importances_` over-weights continuous, high-cardinality predictors like `Expend` and `state_mean`. Switching to `permutation_importance` on the held-out test set produced a more honest ranking — and revealed that `state_mean` still dominates the signal post-fix, which in turn means a large fraction of the "model" is really a geographic lookup. Useful thing to know when interpreting the Purdue prediction.

4. **Point predictions invite over-confident conclusions.** The original capstone claimed "Purdue could raise tuition by \$5,300" from a single point estimate under the leaked RF. A 500-iteration bootstrap under the clean LR puts the point at \$29,239, the 80% interval at \$28,077–\$30,428, and the 95% interval at \$27,470–\$31,111. The actual 2018 tuition (\$28,794) is inside both. The headline reverses. Any policy recommendation on top of a regression model should travel with an interval.

5. **Temporal mismatch is a methodology red flag the original report hand-waved.** ISLR features are from ~1995; the target is 2018 tuition. The model gets away with it because institutional rank-ordering is fairly stable across 20 years, but predicting absolute dollar tuition from 20-year-old predictors is a real concern that should have been flagged up front. A rewrite should either use contemporaneous 2018 predictors or explicitly reframe as "predict peer rank, not dollars."

6. **Hyperparameter grids should be proportionate to the problem.** The original RF grid was 2,376 combinations × 5-fold = ~12 k fits for a few-hundred-row dataset — mostly signaling. A 60-iteration `RandomizedSearchCV` found essentially the same best region in ~300 fits and is the current default.

## Still open (nice-to-haves)

- Temporal mismatch between ISLR (1995) predictor features and 2018 target tuition is acknowledged in this README but not called out inside the wrangling notebook yet.
- Cazenovia grad-rate correction lives in [notebooks/EDA.ipynb](notebooks/EDA.ipynb); cleaning would be more appropriate in [notebooks/Data_Wrangling.ipynb](notebooks/Data_Wrangling.ipynb).
- [notebooks/Data_Wrangling.ipynb](notebooks/Data_Wrangling.ipynb) does not log how many rows are dropped at the fuzzy-match and `dropna` steps.
