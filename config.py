"""
Configuration module for the Rainforest Deforestation ML Project.
Centralizes all paths, parameters, and configurable thresholds.
"""
import os
from pathlib import Path

# ── Project Root ──
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Data Directories ──
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Raw CSV Filenames ──
# The raw files may have "(2)" in their names depending on the download source.
# We search for files matching patterns to handle both naming conventions.
def _find_csv(directory: Path, pattern: str) -> Path:
    """Find a CSV file matching a pattern in the given directory."""
    candidates = list(directory.glob(pattern))
    if candidates:
        return candidates[0]
    # Fallback: check parent directory
    candidates = list(directory.parent.glob(pattern))
    if candidates:
        return candidates[0]
    # Check project root
    candidates = list(PROJECT_ROOT.glob(pattern))
    if candidates:
        return candidates[0]
    return directory / pattern.replace("*", "")

COUNTRY_TREE_LOSS_FILE = _find_csv(RAW_DATA_DIR, "*ountry tree cover loss*")
SUBNATIONAL_TREE_LOSS_FILE = _find_csv(RAW_DATA_DIR, "*ubnational*tree cover loss*")
COUNTRY_CARBON_FILE = _find_csv(RAW_DATA_DIR, "*ountry carbon data*")
SUBNATIONAL_CARBON_FILE = _find_csv(RAW_DATA_DIR, "*ubnational*carbon data*")

# ── Dataset Configuration ──
# Tree-cover density threshold (%) — the GFW datasets contain rows for
# multiple thresholds (0, 10, 15, 20, 25, 30, 50, 75).
# Threshold=30 means "pixels with ≥30% tree-cover density in year 2000".
TREE_COVER_THRESHOLD = 30

# ── Geographic Scope ──
# Default country for analysis. Set to None to analyse all countries.
DEFAULT_COUNTRY = "Brazil"

# Year range in the supplied datasets
YEAR_START = 2001
YEAR_END = 2022
FORECAST_START = YEAR_END + 1  # 2023

# ── Tree-Cover Loss Column Pattern ──
# In the raw CSV: tc_loss_ha_2001, tc_loss_ha_2002, ..., tc_loss_ha_2022
TC_LOSS_COL_PREFIX = "tc_loss_ha_"

# Carbon emissions column pattern
CARBON_COL_PREFIX = "gfw_forest_carbon_gross_emissions_"
CARBON_COL_SUFFIX = "__Mg_CO2e"

# ── Feature Engineering ──
LAG_YEARS = [1, 2, 3]
ROLLING_WINDOWS = [3, 5]

# ── Model Training ──
# Chronological split configuration
TRAIN_END_YEAR = 2017      # Train on 2001–2017 (after lag features kick in)
VAL_END_YEAR = 2019        # Validate on 2018–2019
TEST_START_YEAR = 2020     # Test on 2020–2022

# ── Forecasting ──
DEFAULT_FORECAST_HORIZON = 5  # years beyond last observed year

# ── Risk Assessment ──
# Risk thresholds based on annual loss rate (%)
# These are configurable and documented — not presented as scientific standards.
RISK_THRESHOLDS = {
    "Low": (0.0, 0.5),
    "Moderate": (0.5, 1.0),
    "High": (1.0, 2.0),
    "Very High": (2.0, float("inf")),
}

# Risk weight configuration
RISK_WEIGHTS = {
    "recent_loss_rate": 0.30,
    "loss_trend": 0.25,
    "cumulative_loss_pct": 0.25,
    "predicted_future_loss": 0.20,
}

# ── Visualization ──
PLOTLY_TEMPLATE = "plotly_white"
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

RISK_COLORS = {
    "Low": "#2ca02c",
    "Moderate": "#ffcc00",
    "High": "#ff7f0e",
    "Very High": "#d62728",
}

# ── Logging ──
import logging
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("deforestation_ml")
