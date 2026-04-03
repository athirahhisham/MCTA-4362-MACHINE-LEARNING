# California Housing Price Prediction — Regression Analysis Report

---

## Question 1: Regression Models for Predicting Median House Value

**Dataset:** California Housing Census (20,640 records, 10 features)  
**Target Variable:** `median_house_value`

---

## Part A — Feature Selection and Justification

### Selected Independent Variables

After exploratory analysis, the following features were selected:

**Directly selected (original):**
| Feature | Justification |
|---|---|
| `median_income` | Strongest single predictor of house value (correlation ≈ 0.69). Higher-income neighbourhoods can afford and sustain higher property prices. |
| `housing_median_age` | Older stock in prime areas often commands premiums; newer homes in suburban sprawl may be cheaper. Moderate correlation with target. |
| `latitude` / `longitude` | Geographic location is fundamental — coastal California (San Francisco, LA) commands far higher values than the Central Valley. |
| `ocean_proximity` | Categorical proxy for location desirability (NEAR BAY, NEAR OCEAN vs INLAND). Directly related to demand. |

**Engineered features (derived):**
| Feature | Formula | Justification |
|---|---|---|
| `rooms_per_household` | total_rooms / households | Average space per household; more rooms per family = higher quality housing. |
| `bedrooms_per_room` | total_bedrooms / total_rooms | A high ratio indicates dense bedroom-heavy stock (lower quality); lower ratio suggests spacious homes. |
| `population_per_household` | population / households | Overcrowded districts tend to have lower property values. |

**Excluded features:** `total_rooms`, `total_bedrooms`, `population`, and `households` were excluded as raw counts because they are confounded by block size. Their derived per-household equivalents carry far more signal without multicollinearity.

---

## Part B — Data Preparation

### 1. Handling Missing Values
- **`total_bedrooms`** had **207 missing values** (~1% of data). These were imputed using the **median** (robust to outliers) before computing `bedrooms_per_room`.

### 2. Treating Outliers
- The dataset contains a cap at **$500,001** (artificially truncated values). These 965 records were **removed** because they do not represent true market values and would distort model learning. Records below the 1st percentile were also removed.
- Engineered features were clipped for infinite values (e.g., `total_rooms / households = 0` → division by zero).

### 3. Encoding Categorical Variables
- `ocean_proximity` (5 categories) was **one-hot encoded** using `pd.get_dummies` with `drop_first=True` to avoid the dummy variable trap. This produced 4 binary columns: `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` (with `<1H OCEAN` as the reference category).

### 4. Feature Scaling
- **StandardScaler** (zero mean, unit variance) was applied to all numeric features.
- Scaling is essential for SVR (which uses distance-based kernels) and improves the conditioning of polynomial feature matrices.
- The scaler was **fit only on training data** and applied to the test set to prevent data leakage.

### 5. Train/Test Split
- Data was split **80% training / 20% testing** using `random_state=42` for reproducibility.
- Final cleaned dataset: **19,475 records** → Training: 15,580 | Test: 3,895

---

## Part C — Multiple Linear Regression (MLR)

### Model Details
- **Algorithm:** Ordinary Least Squares (OLS) Linear Regression
- **Library:** scikit-learn `LinearRegression`
- **Features used:** All 11 features (including engineered + one-hot encoded)

### Results

| Metric | Value |
|---|---|
| **R²** | **0.6027** |
| **MAE** | **$45,726** |
| **RMSE** | **$62,033** |

### Discussion
The MLR model explains approximately **60.3% of the variance** in house prices. The MAE of ~$45,700 means that on average, predictions deviate from actual values by this amount — a moderate error given median house values of ~$180,000.

MLR assumes a **linear relationship** between features and the target. In reality, housing prices exhibit complex non-linear patterns (e.g., the effect of income is not purely additive with location). This explains the ceiling around R² = 0.60. Nevertheless, MLR provides a solid, interpretable baseline — `median_income`, `latitude/longitude`, and `ocean_proximity` emerge as the most influential predictors.

---

## Part D — Polynomial Regression

### Model Details and Degree Choice
- **Degree chosen:** 2
- **Features used for polynomial expansion:** 5 core numeric features (`median_income`, `housing_median_age`, `rooms_per_household`, `bedrooms_per_room`, `population_per_household`)
- **Regularisation:** Ridge (L2) with alpha=1000
- **Why degree 2?** Degree 2 adds quadratic terms and interactions (e.g., income × age, income²) that capture known non-linearities. Degree 3+ caused extreme overfitting (negative R² on test) due to the high-dimensional feature space created when combining geographic variables — which means the polynomial expansion must be limited to the core numeric predictors.
- **Why Ridge regularisation?** The 5 features at degree 2 generate 20 polynomial features. Without regularisation, OLS overfits severely (train R² = 0.55, test R² = 0.28). Ridge alpha=1000 stabilises the test score considerably.

### Results

| Metric | Value |
|---|---|
| **R²** | **0.5353** |
| **MAE** | **$50,549** |
| **RMSE** | **$67,088** |

### Discussion
Polynomial Regression at degree 2 performs **below MLR** on this dataset (R² = 0.535 vs 0.603). This is a significant finding: adding polynomial terms did not improve performance. There are two main reasons:

1. **Geographic features dominate** — latitude/longitude and ocean proximity are the strongest location signals, but polynomial expansion on these creates interactions (lat² × income) that don't generalise well.
2. **Limited to 5 features** — to avoid catastrophic overfitting, the polynomial model was restricted to 5 non-geographic features, which excludes key location predictors that MLR uses.

The conclusion is that simple polynomial expansion is not the right tool for spatially structured data — tree-based or spatial models would be more appropriate.

---

## Part E — Support Vector Regression (SVR)

### Model Details
- **Kernel:** RBF (Radial Basis Function)
- **Hyperparameters:** C=100,000 | gamma=0.1 | epsilon=5,000
- **Training sample:** 5,000 records (sub-sampled from training set for computational feasibility — SVR has O(n²–n³) complexity)
- **Why RBF?** The RBF kernel projects data into an infinite-dimensional space and captures complex non-linear patterns without specifying the form of non-linearity upfront. For housing data with non-linear geographic and income effects, RBF outperforms linear and polynomial SVR kernels.

### Results

| Metric | Value |
|---|---|
| **R²** | **0.6886** |
| **MAE** | **$37,528** |
| **RMSE** | **$54,921** |

### Discussion
SVR with RBF kernel is the **best-performing model**, achieving R² = 0.689, MAE = $37,528, and RMSE = $54,921. Compared to MLR:
- R² improved by **+8.6 percentage points**
- MAE reduced by **$8,198 (−17.9%)**
- RMSE reduced by **$7,112 (−11.5%)**

The RBF kernel's ability to capture non-linear decision boundaries is well-suited to housing data where value jumps non-linearly at geographic and income thresholds. The main limitation is computational — SVR was trained on only 5,000 of ~15,500 training samples. Training on the full dataset would likely push R² higher still.

---

## Part F — Model Comparison and Best Model Selection

### Performance Summary

| Model | R² | MAE ($) | RMSE ($) | Rank |
|---|---|---|---|---|
| **SVR (RBF)** | **0.6886** | **37,528** | **54,921** | **1st** |
| Multiple Linear Regression | 0.6027 | 45,726 | 62,033 | 2nd |
| Polynomial Regression (deg=2, Ridge) | 0.5353 | 50,549 | 67,088 | 3rd |

### Best Model: SVR (RBF Kernel)

**Justification:**

1. **Model Accuracy:** SVR achieves the highest R² (0.69) and lowest MAE ($37,528) and RMSE ($54,921) across all three models. It explains ~69% of variance — a meaningful improvement over the linear baseline.

2. **Generalisation Ability:** The RBF kernel implicitly regularises via the support vector margin, making SVR less prone to overfitting compared to polynomial regression. Its results on held-out test data confirm genuine generalisation.

3. **Non-linearity Handling:** Housing prices in California are strongly non-linear — income in Palo Alto behaves very differently to income in Fresno, even at the same level. SVR's kernel trick captures this without requiring explicit feature engineering of interactions.

4. **Practical Considerations:**
   - SVR is **slower to train** than MLR (O(n²–n³) vs O(n·p)), which is relevant for large-scale deployment.
   - SVR requires careful hyperparameter tuning (C, gamma, epsilon), which adds complexity.
   - For production use, **gradient boosting (e.g., XGBoost, LightGBM)** would outperform all three models tested here, but among the three required models, SVR is the clear winner.

**Recommendation:** Use SVR (RBF) for prediction tasks. If interpretability is required (e.g., to explain which factors drive prices), MLR remains useful as a complementary explainable model alongside SVR.

---

## Part G — Power BI Dashboard

Please refer to the interactive HTML dashboard provided separately, which includes:
- Model performance comparison (R², MAE, RMSE)
- Actual vs Predicted scatter plots for all three models
- Geographic distribution of house values
- Feature importance insights
- Error distribution analysis

---

*Report prepared based on California Housing Census dataset. Models implemented using Python (scikit-learn). All preprocessing steps applied to prevent data leakage.*
