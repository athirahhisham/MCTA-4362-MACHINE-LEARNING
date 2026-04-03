# Solar Power Generation Prediction — Regression Analysis Report

**Dataset:** Solar PV System — Kaggle (4,213 records, 21 features)  
**Target Variable:** `generated_power_kw`  
**After Cleaning:** 4,209 records | Train: 3,367 | Test: 842

---

## Part A — Feature Selection and Justification

### Selected Independent Variables (10 Features)

| Feature | Correlation with Target | Physical Justification |
|---|---|---|
| `zenith` | −0.650 | **Solar zenith angle** — the angle between the sun and directly overhead. Higher zenith (sun lower in sky) = less direct radiation = less power. Strongest single predictor. |
| `angle_of_incidence` | −0.646 | **Panel incidence angle** — angle at which sunlight strikes PV panels. More oblique = lower efficiency. Directly governs power output. |
| `shortwave_radiation_backwards_sfc` | +0.556 | **Solar irradiance** — the primary energy input to any PV system. More radiation = more power generated. Strongly positive. |
| `total_cloud_cover_sfc` | −0.334 | Clouds block and scatter sunlight, reducing irradiance reaching panels. |
| `relative_humidity_2_m_above_gnd` | −0.337 | High humidity increases atmospheric scattering of solar radiation; also correlated with cloud presence. |
| `low_cloud_cover_low_cld_lay` | −0.288 | Low clouds are the most effective blockers of solar radiation (closer to surface). |
| `medium_cloud_cover_mid_cld_lay` | −0.228 | Mid-altitude clouds reduce diffuse and direct radiation. |
| `temperature_2_m_above_gnd` | +0.218 | Higher temperatures are correlated with more solar radiation availability. Note: at very high temps PV efficiency slightly drops, but the solar availability effect dominates. |
| `total_precipitation_sfc` | −0.118 | Rain indicates cloud cover and atmospheric moisture — both reduce irradiance. |
| `high_cloud_cover_high_cld_lay` | −0.148 | High cirrus clouds have a diffuse blocking effect. |

### Excluded Features
Wind speed and direction variables (10m, 80m, 900mb), mean sea level pressure, snowfall, and azimuth were excluded. Wind features had near-zero or very weak correlations (|r| < 0.16) with power output — wind does not drive PV generation. Azimuth (−0.06) was negligible. Pressure had only indirect effects. Including these would add noise without predictive value.

---

## Part B — Data Preparation

### 1. Missing Values
The dataset had **4 missing values** across 4 separate rows:
- `temperature_2_m_above_gnd` (row 5)
- `shortwave_radiation_backwards_sfc` (row 18)
- `wind_direction_10_m_above_gnd` (row 1024)
- `generated_power_kw` (row 20)

**Action:** All 4 rows were **dropped** (listwise deletion). This removed only 0.095% of the data — a negligible loss that avoids imputation uncertainty, especially for the target variable (row 20).

### 2. Outlier Treatment
The target variable (`generated_power_kw`) ranges from near 0 (night-time) to 3,056 kW. Night-time zero-power readings are genuine data — the sun simply isn't shining. These were **retained** as they are real operating conditions the model must learn.

No extreme outliers requiring capping were identified after the 4 dropped rows.

### 3. No Categorical Variables
All 20 features are numeric — no encoding was required.

### 4. Feature Scaling
**StandardScaler** (zero mean, unit variance) was applied to all features. This is required for:
- SVR (distance-based kernel calculations are scale-sensitive)
- Polynomial feature expansion (prevents numerical overflow)
- Consistent coefficient interpretation in MLR

The scaler was **fit only on the training set** and applied to the test set to prevent data leakage.

### 5. Train/Test Split
- **80% training** (3,367 records) / **20% testing** (842 records)
- `random_state=42` for reproducibility

---

## Part C — Multiple Linear Regression (MLR)

### Model Details
- **Algorithm:** Ordinary Least Squares Linear Regression
- **Features:** All 10 selected features (standardised)

### Results

| Metric | Value |
|---|---|
| **R²** | **0.5609** |
| **MAE** | **501.66 kW** |
| **RMSE** | **615.23 kW** |

### Discussion
MLR explains **56.1% of variance** in solar power output. An MAE of ~502 kW on a target ranging from 0–3,056 kW represents moderate accuracy. The limitation is inherent in the linear assumption — solar power generation is fundamentally non-linear. The relationship between solar zenith angle and power, for example, follows a cosine function, and cloud cover interacts multiplicatively with irradiance, not additively. MLR cannot capture these interactions without manual feature engineering.

---

## Part D — Polynomial Regression

### Degree Choice: 2
- **Degree 2** was chosen to capture key quadratic relationships and pairwise interactions (e.g., zenith × irradiance, humidity × cloud cover) without excessive overfitting.
- Degree 3 would generate 286 features from 10 inputs — high risk of overfitting on a dataset of 4,209 records.
- **Ridge regularisation** (alpha=10) was applied to the degree-2 expansion (65 features) to control coefficient inflation.

### Results

| Metric | Value |
|---|---|
| **R²** | **0.6430** |
| **MAE** | **415.78 kW** |
| **RMSE** | **554.74 kW** |
| **Training R²** | **0.6978** |

### Discussion
Polynomial Regression (deg=2, Ridge) **clearly improves over MLR** across all metrics:
- R² increased by **+8.2 percentage points** (0.561 → 0.643)
- MAE reduced by **85.88 kW (−17.1%)**
- RMSE reduced by **60.49 kW (−9.8%)**

The train vs test R² gap (0.698 vs 0.643) is acceptable — modest overfitting, well-controlled by Ridge regularisation. The improvement confirms that quadratic terms and pairwise interactions (particularly zenith², zenith×irradiance, and cloud cover interaction terms) capture real physical relationships in PV output.

---

## Part E — Support Vector Regression (SVR)

### Model Details
- **Kernel:** RBF (Radial Basis Function)
- **C = 10,000** (high margin penalty — prioritises fitting)
- **gamma = 'scale'** (auto-scaled to 1/(n_features × X.var()))
- **epsilon = 50** (tube width in kW units)
- **Training subsample:** 3,000 records (computational constraint — SVR is O(n²–n³))

### Results

| Metric | Value |
|---|---|
| **R²** | **0.6805** |
| **MAE** | **336.96 kW** |
| **RMSE** | **524.78 kW** |

### Discussion
SVR with RBF kernel is the **best-performing model** — R² = 0.6805, MAE = 336.96 kW, RMSE = 524.78 kW. The RBF kernel maps data into infinite-dimensional space, capturing the highly non-linear relationships between solar angles, irradiance, and cloud cover that neither MLR nor polynomial regression captures as effectively.

The MAE of 337 kW is particularly notable — a 33% improvement over MLR's 502 kW. For real-world solar farm operations where forecasting accuracy directly impacts energy auction bids and grid management, this improvement is commercially significant.

A key limitation: SVR was trained on 3,000 of 3,367 training records due to computational complexity. Training on the full dataset would likely push R² above 0.72.

---

## Part F — Model Comparison and Best Model

### Summary

| Model | R² | MAE (kW) | RMSE (kW) | Rank |
|---|---|---|---|---|
| **SVR (RBF)** | **0.6805** | **336.96** | **524.78** | 🥇 1st |
| Polynomial Regression (deg=2) | 0.6430 | 415.78 | 554.74 | 🥈 2nd |
| Multiple Linear Regression | 0.5609 | 501.66 | 615.23 | 🥉 3rd |

### Best Model: SVR (RBF Kernel)

**Accuracy:** SVR achieves the highest R² and the lowest MAE and RMSE across all three models. It explains 68% of output variance with an average error of ~337 kW.

**Generalisation:** The RBF kernel's implicit regularisation through the support vector margin prevents overfitting. Unlike polynomial regression where degree selection is critical, SVR naturally adapts to the data's complexity.

**Physical Fit:** Solar PV output is governed by the Beer-Lambert law (exponential attenuation through atmosphere), cosine projection laws (irradiance on tilted surfaces), and multiplicative cloud effects — all fundamentally non-linear. The RBF kernel's infinite-dimensional mapping handles these without prescribing a specific functional form.

**Practical Consideration:** SVR is slower to train (~minutes vs seconds for MLR), but prediction is fast — suitable for real-time grid forecasting. For a full production deployment, **Gradient Boosting (XGBoost/LightGBM)** would be the recommended next step, likely achieving R² > 0.90 with the same feature set.

---

## Part G — Dashboard
See the accompanying interactive HTML dashboard (`Solar_Dashboard.html`) with 4 panels covering: Overview KPIs, Model Performance, Actual vs Predicted, and Data Insights.

---
*Dataset: Solar Power Generation — Kaggle | Models: scikit-learn | Python 3.12*
