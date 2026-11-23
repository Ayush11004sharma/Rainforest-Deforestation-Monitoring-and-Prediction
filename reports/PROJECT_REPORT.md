# Project Report: Rainforest Deforestation Monitoring and Prediction Using Machine Learning

## 1. Executive Summary

This project develops a machine learning application to monitor historical tree-cover loss across Brazil's 27 subnational regions (2001–2022), predict future annual forest loss, estimate associated carbon emissions, and assign transparent deforestation-risk categories. The application is powered by Global Forest Change (GFW) data derived from Landsat satellite imagery (Hansen et al., 2013) and presents results through an interactive Streamlit dashboard.

**Key Findings:**
- Brazil lost approximately **66.1 million hectares** of tree cover from 2001 to 2022 (at ≥30% canopy density threshold)
- Peak loss year was **2016** with 5.38 million hectares
- **Top 5 most-affected regions:** Pará, Mato Grosso, Maranhão, Rondônia, Amazonas
- 12 of 27 regions show **increasing** deforestation trends
- The Historical Baseline model (previous year's loss) achieved the best test-set MAE of **26,092 hectares**
- **Highest-risk regions:** Maranhão, Rondônia, Pará, Mato Grosso, Bahia

---

## 2. Data Description

### 2.1 Datasets

| Dataset | Rows | Columns | Description |
|---------|------|---------|-------------|
| Subnational 1 tree cover loss | 28,000 | 29 | Annual tree-cover loss (ha) by subnational region × threshold |
| Country tree cover loss | 1,888 | 28 | Annual tree-cover loss (ha) by country × threshold |
| Subnational 1 carbon data | 28,000 | 31 | Annual carbon emissions (Mg CO₂e) by subnational region × threshold |
| Country carbon data | 1,888 | 30 | Annual carbon emissions (Mg CO₂e) by country × threshold |

**Source:** Global Forest Change (Hansen et al., 2013, updated), University of Maryland / GLAD Lab.

### 2.2 Preprocessing

- Filtered to tree-cover density threshold ≥30% (year 2000 baseline)
- Filtered to Brazil (27 subnational regions)
- Melted 22 wide-format annual columns (tc_loss_ha_2001…2022) into long format
- Calculated annual forest-loss rate: `(tc_loss_ha / extent_2000_ha) × 100`
- Merged carbon emissions data (100% match for Brazil at threshold=30)
- Final analytical dataset: **594 rows** (27 regions × 22 years), no missing values

### 2.3 Important Definitions

- **Tree-cover loss** is gross removal of tree canopy detected from Landsat imagery. It is NOT the same as permanent deforestation, net deforestation, or illegal logging.
- **Forest-loss rate** uses the fixed year-2000 baseline extent as denominator.
- **Carbon emissions** are from the GFW forest carbon flux model, NOT computed from hectares by this application.

---

## 3. Exploratory Data Analysis

### 3.1 Temporal Patterns
- Annual tree-cover loss in Brazil increased from ~2.5M ha (2001) to a peak of ~5.4M ha (2016)
- A notable decrease occurred around 2008–2012 (coinciding with enforcement of the Forest Code)
- Loss has remained elevated at 3–5M ha/year in recent years

### 3.2 Regional Patterns
- **Pará** consistently leads in absolute tree-cover loss (>12M ha total)
- **Mato Grosso** had the highest early losses but has declined
- **Maranhão** and **Piauí** show the strongest increasing trends
- Small states (Distrito Federal, Sergipe) have minimal absolute loss but may have high percentage loss

### 3.3 Carbon Emissions
- Total emissions: approximately **30 billion Mg CO₂e** over 2001–2022
- Strong correlation between tree-cover loss and carbon emissions (Pearson r ≈ 0.97)
- Pará, Mato Grosso, and Maranhão dominate emissions

---

## 4. Feature Engineering

15 features were engineered from historical observations, designed to prevent target leakage:

| Category | Features | Count |
|----------|----------|-------|
| Lag features | 1, 2, 3-year lagged loss | 3 |
| Rolling features | 3yr and 5yr trailing mean and std | 4 |
| Trend features | YoY change, acceleration, 3yr slope | 4 |
| Rate features | Lagged loss rate, cumulative loss % | 2 |
| Variability | 5yr coefficient of variation | 1 |
| Region encoding | Label-encoded region ID | 1 |

**Leakage prevention:** All features use `shift(1)` or earlier data only. Rolling windows exclude the current observation. No future values are used.

---

## 5. Model Training and Evaluation

### 5.1 Chronological Split

| Split | Years | Rows | Purpose |
|-------|-------|------|---------|
| Training | ≤2017 | 351 | Model fitting |
| Validation | 2018–2019 | 54 | Hyperparameter tuning |
| Test | 2020–2022 | 81 | Final evaluation |

No random shuffling — all test data is strictly after training data.

### 5.2 Model Comparison (Test Set)

| Model | MAE (ha) | RMSE (ha) | R² |
|-------|----------|-----------|-----|
| **Historical Baseline** | **26,092** | **69,295** | **0.870** |
| Random Forest | 37,386 | 77,939 | 0.835 |
| XGBoost | 52,121 | 113,592 | 0.650 |
| Linear Regression | 53,133 | 101,483 | 0.721 |

### 5.3 Interpretation

The Historical Baseline (previous year's loss as prediction) achieves the lowest MAE, which reflects the strong year-to-year persistence of deforestation patterns. This is common in short time-series settings where recent history is the best predictor.

Random Forest achieves a strong R² of 0.835 and captures nonlinear relationships between features and loss. Its top features by importance are `tc_loss_ha_lag1`, `tc_loss_ha_roll_mean_5yr`, and `region_encoded`.

XGBoost shows signs of overfitting (R² = 1.0 on training, 0.65 on test), suggesting that with only 351 training samples, gradient boosting is too flexible.

**Feature importance does NOT imply causation.** The models learn statistical associations from historical data.

---

## 6. Forecasting

### 6.1 Method
- Forecasts begin at 2023 (after last observed year 2022)
- Features for future years are computed recursively using predicted values
- Default forecast horizon: 5 years (2023–2027)
- Uncertainty intervals use test-set residual standard deviation (±1.96σ for 95% CI)

### 6.2 Limitations
- Uncertainty grows with forecast horizon but our simple interval method assumes constant variance
- Forecasts cannot account for policy changes, fires, climate events, or economic shifts
- These are **estimates**, not certain outcomes or real-time satellite monitoring

---

## 7. Risk Assessment

### 7.1 Method
Composite risk score (0–1) from 4 normalized, weighted indicators:

| Indicator | Weight | Description |
|-----------|--------|-------------|
| Recent loss rate | 30% | Average annual loss rate over last 3 years |
| Loss trend | 25% | Linear slope of loss over last 3 years |
| Cumulative loss % | 25% | Total loss as % of year-2000 baseline |
| Predicted future loss | 20% | Model-predicted average future loss |

### 7.2 Results

| Risk Category | Count | Regions |
|---------------|-------|---------|
| Very High | 7 | Maranhão, Rondônia, Pará, Mato Grosso, Bahia, Tocantins, Piauí |
| High | 7 | Amazonas, Acre, Mato Grosso do Sul, Goiás, Minas Gerais, Paraná, Santa Catarina |
| Moderate | 6 | Roraima, Pernambuco, Paraíba, Ceará, Espírito Santo, Rio Grande do Sul |
| Low | 7 | Amapá, Rio de Janeiro, São Paulo, Alagoas, Sergipe, Rio Grande do Norte, Distrito Federal |

### 7.3 Disclaimer
This is an **analytical risk assessment** for academic purposes. It is NOT a verified prediction of illegal logging or an official conservation classification. Thresholds are configurable and documented.

---

## 8. Dashboard

The Streamlit dashboard (app.py) provides 7 interactive pages:

1. **Overview** — Key metrics, total annual loss bar chart, regional comparison
2. **Historical Monitoring** — Filters, trends, heatmaps, ranking tables, CSV downloads
3. **ML Model Evaluation** — Comparison table, actual vs predicted plots, feature importance
4. **Future Forecasts** — Per-region forecasts with uncertainty, CSV export
5. **Risk Assessment** — Rankings, category distribution, methodology documentation
6. **Carbon Analysis** — Emissions trends, regional comparison, loss-emissions correlation
7. **Data & Methodology** — Definitions, data quality, limitations, feature documentation

No paid APIs or API keys required. Runs locally with `streamlit run app.py`.

---

## 9. Limitations and Future Work

### Limitations
- Only 22 annual observations per region — limits model complexity
- No causal data (roads, population, commodity prices, policy changes)
- Tree-cover loss ≠ permanent deforestation
- Results should not be used for operational decisions without independent validation
- Satellite-based detection may miss small-scale or sub-canopy deforestation

### Future Improvements
1. Incorporate deforestation drivers (socioeconomic, infrastructure, policy data)
2. Cross-validate with PRODES and MapBiomas datasets
3. Add monthly/sub-annual resolution
4. Implement conformal prediction for better uncertainty quantification
5. Apply deep learning time-series models with more data
6. Add spatial analysis with GIS-based deforestation hotspot mapping

---

## 10. Reproducibility

- **All code** is modular and organized in `src/` with clear function signatures
- **25 unit/integration tests** cover loading, preprocessing, features, training, evaluation, forecasting, and risk assessment
- **`run_pipeline.py`** executes the complete pipeline end-to-end
- **`requirements.txt`** pins all dependencies
- **No fabricated data or metrics** — all results are computed from real model evaluations
- **Random seeds** are fixed (42) for reproducible training

---

*Report generated as part of the Rainforest Deforestation ML Project.*
*Data source: Global Forest Change (Hansen et al., 2013, updated).*
