# Rainforest Deforestation Monitoring and Prediction Using Machine Learning

**Monitor Deforestation Rates in Rainforests Using Time-Series Remote Sensing for Conservation Efforts**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This machine learning project analyzes historical tree-cover loss across Brazil's 27 subnational regions (2001–2022) using Global Forest Change data derived from Landsat satellite imagery (Hansen et al.). It trains predictive models to forecast future forest loss, estimates associated carbon emissions, assigns transparent deforestation-risk categories, and presents results through an interactive Streamlit dashboard.

> **Important:** Tree-cover loss ≠ permanent deforestation. Loss includes natural disturbances, plantation harvesting, fire, and legal logging. This analysis measures *gross tree-cover loss* at the ≥30% canopy density threshold.

---

## Key Results

| Model | Test MAE (ha) | Test RMSE (ha) | Test R² |
|-------|---------------|----------------|---------|
| Historical Baseline | 26,092 | 69,295 | 0.870 |
| Random Forest | 37,386 | 77,939 | 0.835 |
| XGBoost | 52,121 | 113,592 | 0.650 |
| Linear Regression | 53,133 | 101,483 | 0.721 |

- **66.1 million hectares** total tree-cover loss (2001–2022)
- **Top affected regions:** Pará, Mato Grosso, Maranhão, Rondônia, Amazonas
- **Highest-risk regions:** Maranhão, Rondônia, Pará, Mato Grosso, Bahia

---

## Project Structure

```
rainforest-deforestation-ml/
├── data/
│   ├── raw/                     # Original CSV files (not committed)
│   └── processed/               # Cleaned analytical datasets
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── src/
│   ├── data_loader.py           # CSV loading and validation
│   ├── preprocessing.py         # Wide→long conversion, loss rate calc
│   ├── eda.py                   # Exploratory data analysis
│   ├── feature_engineering.py   # Lag, rolling, trend features
│   ├── train.py                 # Model training (Baseline, LR, RF, XGB)
│   ├── evaluate.py              # Chronological evaluation
│   ├── forecasting.py           # Recursive future forecasting
│   ├── risk_assessment.py       # Risk scoring and categorization
│   └── visualization.py         # Plotly chart functions
├── models/                      # Saved model artifacts (.joblib)
├── reports/                     # Evaluation results, risk reports
├── tests/
│   └── test_pipeline.py         # 25 unit/integration tests
├── app.py                       # Streamlit dashboard
├── config.py                    # Centralized configuration
├── run_pipeline.py              # End-to-end pipeline script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+ (tested with 3.13)
- pip package manager

### Local Setup (VS Code / Terminal)

```bash
# 1. Clone or navigate to the project directory
cd "Rainforest Deforestation Monitoring and Prediction Using Machine Learning"

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Ensure CSV data files are in data/raw/
#    Required files:
#      - Country tree cover loss.csv
#      - Subnational 1 tree cover loss.csv
#      - Country carbon data.csv
#      - Subnational 1 carbon data.csv
#    (or in the project root directory — the config auto-detects both locations)

# 5. Run the full pipeline
python run_pipeline.py

# 6. Launch the dashboard
streamlit run app.py
```

### Google Colab Setup

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Set up project directory
import os
PROJECT_DIR = '/content/drive/MyDrive/Rainforest Deforestation ML'
os.chdir(PROJECT_DIR)

# 3. Install dependencies
!pip install -q -r requirements.txt

# 4. Copy CSV files to data/raw/ if not already there
!mkdir -p data/raw
!cp "Country tree cover loss.csv" data/raw/ 2>/dev/null || true
!cp "Subnational 1 tree cover loss.csv" data/raw/ 2>/dev/null || true
!cp "Country carbon data.csv" data/raw/ 2>/dev/null || true
!cp "Subnational 1 carbon data.csv" data/raw/ 2>/dev/null || true

# 5. Run pipeline
!python run_pipeline.py

# 6. For the Streamlit dashboard in Colab, use:
#    !pip install pyngrok
#    from pyngrok import ngrok
#    !streamlit run app.py &>/dev/null &
#    public_url = ngrok.connect(8501)
#    print(public_url)
```

---

## Usage

### Run the Complete Pipeline

```bash
python run_pipeline.py
```

This executes all phases in order:
1. Load and validate 4 CSV datasets
2. Preprocess: filter threshold=30, melt to long format, calculate loss rates
3. Exploratory data analysis and summary statistics
4. Feature engineering (15 leakage-safe time-series features)
5. Chronological train/validation/test split
6. Train 4 models (Baseline, Linear Regression, Random Forest, XGBoost)
7. Evaluate on test set (2020–2022)
8. Generate 5-year forecasts (2023–2027) for all 27 regions
9. Risk assessment for all regions

### Launch the Dashboard

```bash
streamlit run app.py
```

The dashboard includes 7 pages:
- **Overview** — Key metrics and top-level findings
- **Historical Monitoring** — Interactive charts, heatmaps, filters
- **ML Model Evaluation** — Comparison table, actual vs predicted plots
- **Future Forecasts** — Per-region forecasts with uncertainty intervals
- **Risk Assessment** — Risk rankings, category distribution, methodology
- **Carbon Analysis** — Emissions trends and correlations
- **Data & Methodology** — Dataset info, definitions, limitations

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Datasets

| File | Rows | Description |
|------|------|-------------|
| Country tree cover loss.csv | 1,888 | Annual tree-cover loss by country × threshold |
| Subnational 1 tree cover loss.csv | 28,000 | Annual tree-cover loss by subnational region × threshold |
| Country carbon data.csv | 1,888 | Annual carbon emissions by country × threshold |
| Subnational 1 carbon data.csv | 28,000 | Annual carbon emissions by subnational region × threshold |

**Source:** Global Forest Change (Hansen et al., 2013, updated) — University of Maryland / GLAD Lab, derived from Landsat satellite imagery.

**Primary analysis dataset:** Subnational tree-cover loss at threshold=30% for Brazil (27 regions, 22 years = 594 observations).

---

## Methodology

### Data Processing
- Filter to ≥30% tree-cover density threshold
- Melt wide-format annual columns (tc_loss_ha_2001..2022) to long format
- Calculate annual loss rate: `(tc_loss_ha / extent_2000_ha) × 100`
- Merge carbon emissions from the GFW carbon dataset

### Feature Engineering (15 features, all leakage-safe)
- **Lag features:** 1, 2, 3-year lagged loss values
- **Rolling features:** 3-year and 5-year trailing means and standard deviations
- **Trend features:** Year-over-year change, acceleration, 3-year linear slope
- **Rate features:** Lagged loss rate and cumulative loss percentage
- **Variability:** 5-year coefficient of variation
- **Region encoding:** Label-encoded region identifier

### Models
| Model | Description |
|-------|-------------|
| Historical Baseline | Previous year's loss as prediction |
| Linear Regression | Scaled linear model with all features |
| Random Forest | 200 trees, max depth 15 |
| XGBoost | Gradient-boosted trees (200 estimators) |

### Evaluation
- **Chronological split:** Train ≤2017, Validate 2018–2019, Test 2020–2022
- **No random shuffling** — prevents temporal leakage
- Metrics: MAE, RMSE, R² computed from real predictions

### Risk Assessment
Composite score (0–1) from 4 normalized, weighted indicators:
- Recent loss rate (30%)
- Loss trend (25%)
- Cumulative loss % (25%)
- Predicted future loss (20%)

Categories assigned by quartile breakpoints (Low / Moderate / High / Very High).

---

## Limitations

- **Short time series:** 22 annual observations per region limits model complexity
- **No causal data:** Dataset does not include deforestation drivers
- **Tree-cover loss ≠ deforestation:** Includes fire, plantation harvest, natural disturbances
- **Forecasts are estimates:** Cannot predict policy changes, climate events, or economic shifts
- **Risk categories are relative rankings**, not scientifically established standards
- Results require independent validation before operational conservation use

---

## Future Improvements

1. Incorporate deforestation drivers (roads, commodity prices, population)
2. Cross-validate with PRODES and MapBiomas datasets
3. Add sub-annual (monthly) resolution if data becomes available
4. Implement conformal prediction for better uncertainty quantification
5. Add spatial analysis with GIS-based visualizations
6. Use deep learning time-series models (LSTM, Transformer) with more data

---

## License

This project is for academic and educational purposes. The underlying data is from the Global Forest Change dataset (Hansen et al., CC-BY 4.0).

---

## Citation

If you use this project, please cite:

> Hansen, M. C., P. V. Potapov, R. Moore, M. Hancher, S. A. Turubanova, A. Tyukavina, D. Thau, S. V. Stehman, S. J. Goetz, T. R. Loveland, A. Kommareddy, A. Egorov, L. Chini, C. O. Justice, and J. R. G. Townshend. 2013. "High-Resolution Global Maps of 21st-Century Forest Cover Change." *Science* 342 (15 November): 850–53. Data available from: https://glad.earthengine.app/view/global-forest-change
