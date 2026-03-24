# San Francisco Rental Price Prediction

Two notebooks exploring linear regression and regularized regression to predict rental prices in San Francisco.

---

## Dataset

`sf_clean.csv` — 989 listings, pre-cleaned.

| Column | Description |
|---|---|
| `price` | Monthly rent in USD (target) |
| `sqft` | Square footage |
| `beds` | Number of bedrooms |
| `bath` | Number of bathrooms |
| `laundry` | In-unit / on-site / no laundry |
| `pets` | Pet policy |
| `housing_type` | Single / double / multi-unit |
| `parking` | Parking type |
| `hood_district` | Neighbourhood district (numeric code) |

---

## Notebooks

### 1. `ols.ipynb` — OLS Linear Regression

Iterative model-building workflow using `statsmodels` OLS with 5-fold cross-validation. The target is log-transformed (`log(price)`) and MAE is reported in original dollar units.

**Feature Engineering**

- Log-transform of `price` as the response variable
- `laundry` collapsed to binary: in-unit vs. not in-unit
- `hood_district` grouped into named zones: `west`, `southwest`, `central`, `marina`, `north_beach`, `fidi_soma`
- Parking and pet policy encoded as indicator variables
- One-hot encoding of remaining categoricals

**Model Comparison (5-fold CV)**

| Model | Features | CV R² | CV MAE |
|---|---|---|---|
| M1_baseline | sqft, beds, bath | 0.717 | $594.7 |
| M2_laundry | + laundry | 0.745 | $566.3 |
| M3_parking | + parking | 0.760 | $543.8 |
| M5_pets | + pets | 0.759 | $545.4 |
| M6_housing_type | + housing type | 0.762 | $542.4 |
| **M4_district** | **+ district zones** | **0.796** | **$509.1** |

**Final Model: M4_district**

| Metric | Score |
|---|---|
| Test R² | 0.785 |
| Test MAE | $440.4 |

---

### 2. `regularized_regression.ipynb` — Ridge, Lasso, and ElasticNet

Extends the OLS project by applying regularization. Builds on the same feature set as M4_district with additional engineered interaction terms. Features are standardized with `StandardScaler` prior to fitting.

**Additional Feature Engineering**

- Valet parking interactions: `valet_x_sqft`, `valet_x_beds`, `valet_x_bath`
- Single-family interactions: `single_x_beds`, `single_x_bath`

**Hyperparameter Tuning**

All three models use scikit-learn's built-in CV variants (`RidgeCV`, `LassoCV`, `ElasticNetCV`) with a 200-point log-spaced alpha grid over [10⁻³, 10³] and 5-fold CV.

**Training Results**

| Model | Tuned α | (λ) | Train R² | Train MAE |
|---|---|---|---|---|
| Ridge | 20.49 | — | 0.817 | $479.8 |
| Lasso | 0.001 | — | 0.817 | $482.3 |
| ElasticNet | 0.026 | 0.01 | 0.816 | $479.9 |

**Final Model: Ridge**

| Metric | Score |
|---|---|
| Test R² | 0.779 |
| Test MAE | $469.0 |

---

## Dependencies

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```
