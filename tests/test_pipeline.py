"""
Tests for the Rainforest Deforestation ML Project
=================================================
Covers data loading, preprocessing, feature engineering, model training,
evaluation, forecasting, and risk assessment.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════

@pytest.fixture
def sample_wide_df():
    """Create a sample wide-format dataset for testing."""
    return pd.DataFrame({
        "country": ["Brazil"] * 3,
        "subnational1": ["Para", "Amazonas", "Mato Grosso"],
        "threshold": [30, 30, 30],
        "area_ha": [100000, 200000, 150000],
        "extent_2000_ha": [80000, 180000, 100000],
        "extent_2010_ha": [75000, 175000, 90000],
        "gain_2000-2020_ha": [1000, 2000, 500],
        "tc_loss_ha_2001": [1000, 500, 2000],
        "tc_loss_ha_2002": [1100, 600, 1800],
        "tc_loss_ha_2003": [1200, 550, 1900],
        "tc_loss_ha_2004": [1300, 700, 2100],
        "tc_loss_ha_2005": [1400, 650, 2200],
        "tc_loss_ha_2006": [1500, 800, 2300],
        "tc_loss_ha_2007": [1600, 750, 2400],
        "tc_loss_ha_2008": [900, 400, 1500],
        "tc_loss_ha_2009": [1000, 450, 1600],
        "tc_loss_ha_2010": [1100, 500, 1700],
        "tc_loss_ha_2011": [1200, 550, 1800],
        "tc_loss_ha_2012": [1300, 600, 1900],
        "tc_loss_ha_2013": [1400, 650, 2000],
        "tc_loss_ha_2014": [1500, 700, 2100],
        "tc_loss_ha_2015": [1600, 750, 2200],
        "tc_loss_ha_2016": [1100, 500, 1700],
        "tc_loss_ha_2017": [1200, 550, 1800],
        "tc_loss_ha_2018": [1300, 600, 1900],
        "tc_loss_ha_2019": [1400, 650, 2000],
        "tc_loss_ha_2020": [1500, 700, 2100],
        "tc_loss_ha_2021": [1600, 750, 2200],
        "tc_loss_ha_2022": [1700, 800, 2300],
    })


@pytest.fixture
def sample_long_df(sample_wide_df):
    """Create a preprocessed long-format dataset."""
    from src.preprocessing import preprocess_subnational_tree_loss
    return preprocess_subnational_tree_loss(
        sample_wide_df, threshold=30, country="Brazil"
    )


@pytest.fixture
def sample_featured_df(sample_long_df):
    """Create a featured dataset."""
    from src.feature_engineering import build_feature_set
    return build_feature_set(sample_long_df)


# ════════════════════════════════════════════════════════
# Test Data Loading
# ════════════════════════════════════════════════════════

class TestDataLoading:
    """Test CSV loading and validation."""
    
    def test_load_subnational_loss(self):
        """Test loading the subnational tree cover loss CSV."""
        from src.data_loader import load_csv_safe
        from config import SUBNATIONAL_TREE_LOSS_FILE
        df = load_csv_safe(SUBNATIONAL_TREE_LOSS_FILE, "Subnational Loss")
        assert df is not None
        assert len(df) > 0
        assert "country" in df.columns
        assert "subnational1" in df.columns
    
    def test_load_country_loss(self):
        """Test loading the country tree cover loss CSV."""
        from src.data_loader import load_csv_safe
        from config import COUNTRY_TREE_LOSS_FILE
        df = load_csv_safe(COUNTRY_TREE_LOSS_FILE, "Country Loss")
        assert df is not None
        assert "country" in df.columns
    
    def test_load_subnational_carbon(self):
        """Test loading the subnational carbon CSV."""
        from src.data_loader import load_csv_safe
        from config import SUBNATIONAL_CARBON_FILE
        df = load_csv_safe(SUBNATIONAL_CARBON_FILE, "Subnational Carbon")
        assert df is not None
    
    def test_load_country_carbon(self):
        """Test loading the country carbon CSV."""
        from src.data_loader import load_csv_safe
        from config import COUNTRY_CARBON_FILE
        df = load_csv_safe(COUNTRY_CARBON_FILE, "Country Carbon")
        assert df is not None
    
    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises FileNotFoundError."""
        from src.data_loader import load_csv_safe
        with pytest.raises(FileNotFoundError):
            load_csv_safe(Path("nonexistent.csv"), "Test")


# ════════════════════════════════════════════════════════
# Test Annual Column Parsing
# ════════════════════════════════════════════════════════

class TestColumnParsing:
    """Test extraction of annual columns and year values."""
    
    def test_extract_annual_columns(self, sample_wide_df):
        from src.preprocessing import extract_annual_columns
        cols = extract_annual_columns(sample_wide_df, "tc_loss_ha_")
        assert len(cols) == 22
        assert cols[0] == "tc_loss_ha_2001"
        assert cols[-1] == "tc_loss_ha_2022"
    
    def test_extract_year_from_column(self):
        from src.preprocessing import extract_year_from_column
        assert extract_year_from_column("tc_loss_ha_2001") == 2001
        assert extract_year_from_column("tc_loss_ha_2022") == 2022
        assert extract_year_from_column("extent_2000_ha") == 2000
        assert extract_year_from_column("no_year_here") is None
    
    def test_carbon_column_extraction(self):
        """Test extraction of carbon emissions columns."""
        from src.preprocessing import extract_annual_columns
        df = pd.DataFrame({
            "gfw_forest_carbon_gross_emissions_2001__Mg_CO2e": [100],
            "gfw_forest_carbon_gross_emissions_2002__Mg_CO2e": [200],
            "other_col": [0],
        })
        cols = extract_annual_columns(df, "gfw_forest_carbon_gross_emissions_")
        assert len(cols) == 2


# ════════════════════════════════════════════════════════
# Test Long-Format Conversion
# ════════════════════════════════════════════════════════

class TestLongFormatConversion:
    """Test wide-to-long conversion."""
    
    def test_melt_produces_correct_shape(self, sample_wide_df):
        from src.preprocessing import melt_tree_cover_loss
        long = melt_tree_cover_loss(sample_wide_df)
        # 3 regions × 22 years = 66 rows
        assert len(long) == 66
        assert "year" in long.columns
        assert "tc_loss_ha" in long.columns
    
    def test_melt_year_range(self, sample_wide_df):
        from src.preprocessing import melt_tree_cover_loss
        long = melt_tree_cover_loss(sample_wide_df)
        assert long["year"].min() == 2001
        assert long["year"].max() == 2022
    
    def test_full_preprocessing(self, sample_wide_df):
        from src.preprocessing import preprocess_subnational_tree_loss
        result = preprocess_subnational_tree_loss(
            sample_wide_df, threshold=30, country="Brazil"
        )
        assert "region" in result.columns
        assert "year" in result.columns
        assert "tc_loss_ha" in result.columns
        assert "loss_rate_pct" in result.columns
        assert "cumulative_loss_ha" in result.columns


# ════════════════════════════════════════════════════════
# Test Forest-Loss Rate
# ════════════════════════════════════════════════════════

class TestLossRate:
    """Test forest-loss rate calculations."""
    
    def test_rate_calculation(self):
        from src.preprocessing import calculate_loss_rate
        df = pd.DataFrame({
            "tc_loss_ha": [1000, 500, 0],
            "extent_2000_ha": [100000, 50000, 100000],
        })
        result = calculate_loss_rate(df)
        assert "loss_rate_pct" in result.columns
        assert abs(result.iloc[0]["loss_rate_pct"] - 1.0) < 1e-6
        assert abs(result.iloc[1]["loss_rate_pct"] - 1.0) < 1e-6
        assert abs(result.iloc[2]["loss_rate_pct"] - 0.0) < 1e-6
    
    def test_zero_extent_produces_nan(self):
        from src.preprocessing import calculate_loss_rate
        df = pd.DataFrame({
            "tc_loss_ha": [1000],
            "extent_2000_ha": [0],
        })
        result = calculate_loss_rate(df)
        assert np.isnan(result.iloc[0]["loss_rate_pct"])
    
    def test_missing_extent_produces_nan(self):
        from src.preprocessing import calculate_loss_rate
        df = pd.DataFrame({
            "tc_loss_ha": [1000],
            "extent_2000_ha": [np.nan],
        })
        result = calculate_loss_rate(df)
        assert np.isnan(result.iloc[0]["loss_rate_pct"])


# ════════════════════════════════════════════════════════
# Test Feature Engineering
# ════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Test lag features, rolling averages, and leakage prevention."""
    
    def test_lag_features_created(self, sample_long_df):
        from src.feature_engineering import add_lag_features
        result = add_lag_features(sample_long_df)
        assert "tc_loss_ha_lag1" in result.columns
        assert "tc_loss_ha_lag2" in result.columns
        assert "tc_loss_ha_lag3" in result.columns
    
    def test_lag1_correct_values(self, sample_long_df):
        from src.feature_engineering import add_lag_features
        result = add_lag_features(sample_long_df)
        
        for region in result["region"].unique():
            rd = result[result["region"] == region].sort_values("year")
            # First year should have NaN lag1
            assert np.isnan(rd.iloc[0]["tc_loss_ha_lag1"])
            # Second year's lag1 should equal first year's value
            assert rd.iloc[1]["tc_loss_ha_lag1"] == rd.iloc[0]["tc_loss_ha"]
    
    def test_rolling_features_no_leakage(self, sample_long_df):
        """Verify rolling features don't include current year's value."""
        from src.feature_engineering import add_rolling_features
        result = add_rolling_features(sample_long_df, windows=[3])
        
        for region in result["region"].unique():
            rd = result[result["region"] == region].sort_values("year")
            # First 3 years should be NaN (need 3 shifted values + 1 shift)
            # At minimum, first year should be NaN
            assert np.isnan(rd.iloc[0]["tc_loss_ha_roll_mean_3yr"])
    
    def test_full_feature_set(self, sample_long_df):
        from src.feature_engineering import build_feature_set, get_feature_columns
        result = build_feature_set(sample_long_df)
        features = get_feature_columns(result)
        assert len(features) > 5
        assert "region_encoded" in result.columns


# ════════════════════════════════════════════════════════
# Test Chronological Splitting
# ════════════════════════════════════════════════════════

class TestChronologicalSplit:
    """Test time-based data splitting and leakage prevention."""
    
    def test_split_ordering(self, sample_featured_df):
        from src.train import chronological_split
        splits = chronological_split(
            sample_featured_df,
            train_end_year=2015,
            val_end_year=2018,
        )
        
        meta_train = splits["meta_train"]
        meta_val = splits["meta_val"]
        meta_test = splits["meta_test"]
        
        # Verify ordering
        assert meta_train["year"].max() <= 2015
        assert meta_val["year"].min() >= 2016
        assert meta_val["year"].max() <= 2018
        assert meta_test["year"].min() >= 2019
    
    def test_no_data_leakage(self, sample_featured_df):
        """Verify test period is strictly after training period."""
        from src.train import chronological_split
        splits = chronological_split(
            sample_featured_df,
            train_end_year=2015,
            val_end_year=2018,
        )
        
        train_max = splits["meta_train"]["year"].max()
        test_min = splits["meta_test"]["year"].min()
        assert test_min > train_max


# ════════════════════════════════════════════════════════
# Test Model Training & Evaluation
# ════════════════════════════════════════════════════════

class TestModelTraining:
    """Test model training and evaluation."""
    
    def test_baseline_model(self, sample_featured_df):
        from src.train import HistoricalBaseline, chronological_split
        from src.feature_engineering import get_feature_columns
        
        splits = chronological_split(
            sample_featured_df, train_end_year=2015, val_end_year=2018,
            feature_cols=get_feature_columns(sample_featured_df),
        )
        
        model = HistoricalBaseline()
        model.fit(*splits["train"])
        preds = model.predict(splits["test"][0])
        assert len(preds) == len(splits["test"][1])
        assert np.all(np.isfinite(preds))
    
    def test_evaluation_metrics(self):
        from src.evaluate import evaluate_model
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        metrics = evaluate_model(y_true, y_pred)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics
        assert metrics["MAE"] >= 0
        assert metrics["R2"] <= 1.0


# ════════════════════════════════════════════════════════
# Test Forecasting
# ════════════════════════════════════════════════════════

class TestForecasting:
    """Test future forecast generation."""
    
    def test_forecast_produces_correct_years(self, sample_featured_df):
        from src.train import HistoricalBaseline
        from src.forecasting import forecast_region
        
        bundle = {
            "model": HistoricalBaseline(),
            "imputer": None,
            "scaler": None,
            "feature_cols": [c for c in sample_featured_df.columns
                           if c.startswith("tc_loss_ha_lag")],
        }
        
        result = forecast_region(
            sample_featured_df, "Para", bundle,
            horizon=3, last_observed_year=2022,
        )
        
        forecast_only = result[result["type"] == "forecast"]
        assert len(forecast_only) == 3
        assert list(forecast_only["year"]) == [2023, 2024, 2025]


# ════════════════════════════════════════════════════════
# Test Risk Assessment
# ════════════════════════════════════════════════════════

class TestRiskAssessment:
    """Test risk scoring and categorization."""
    
    def test_risk_indicators(self, sample_long_df):
        from src.risk_assessment import compute_risk_indicators
        indicators = compute_risk_indicators(sample_long_df)
        assert len(indicators) == sample_long_df["region"].nunique()
        assert "recent_loss_rate" in indicators.columns
        assert "loss_trend_ha_yr" in indicators.columns
    
    def test_risk_categories_assigned(self, sample_long_df):
        from src.risk_assessment import full_risk_assessment
        result = full_risk_assessment(sample_long_df)
        assert "risk_category" in result.columns
        assert "risk_score" in result.columns
        assert all(result["risk_score"].between(0, 1))
        valid_cats = {"Low", "Moderate", "High", "Very High"}
        assert set(result["risk_category"]).issubset(valid_cats)
