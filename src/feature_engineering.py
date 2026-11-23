"""
Feature Engineering Module
==========================
Builds time-series features for deforestation prediction.

All features are derived from historical observations ONLY.
No future data or target leakage is permitted.

Feature groups:
  - Lag features: previous 1, 2, 3 year losses
  - Rolling features: 3-year and 5-year moving averages
  - Trend features: linear trend slope, acceleration
  - Rate features: previous-year loss rate
  - Variability: coefficient of variation
  - Region encoding: label-encoded region identifiers
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger("deforestation_ml.feature_engineering")


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    lags: List[int] = None,
    group_col: str = "region",
) -> pd.DataFrame:
    """
    Add lagged versions of the target variable.
    
    Each lag feature contains the target value from N years prior.
    Computed per region to prevent cross-region leakage.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'year', group_col, and target_col.
    target_col : str
        Column to lag.
    lags : list of int
        Lag periods (default: [1, 2, 3]).
    group_col : str
        Group column for per-region computation.
        
    Returns
    -------
    pd.DataFrame
        With added lag columns (e.g., 'tc_loss_ha_lag1').
    """
    if lags is None:
        lags = [1, 2, 3]
    
    df = df.sort_values([group_col, "year"]).copy()
    
    for lag in lags:
        col_name = f"{target_col}_lag{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        logger.debug(f"Added {col_name}: {df[col_name].notna().sum()} valid values")
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    windows: List[int] = None,
    group_col: str = "region",
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features.
    
    Uses a trailing window that does NOT include the current observation.
    For a window of size W at year Y, uses years [Y-W, Y-1].
    
    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    windows : list of int
        Rolling window sizes (default: [3, 5]).
    group_col : str
    
    Returns
    -------
    pd.DataFrame
        With rolling mean and std columns.
    """
    if windows is None:
        windows = [3, 5]
    
    df = df.sort_values([group_col, "year"]).copy()
    
    for w in windows:
        mean_col = f"{target_col}_roll_mean_{w}yr"
        std_col = f"{target_col}_roll_std_{w}yr"
        
        # shift(1) ensures we don't include the current year
        shifted = df.groupby(group_col)[target_col].shift(1)
        
        df[mean_col] = shifted.groupby(df[group_col]).transform(
            lambda x: x.rolling(window=w, min_periods=max(1, w - 1)).mean()
        )
        df[std_col] = shifted.groupby(df[group_col]).transform(
            lambda x: x.rolling(window=w, min_periods=max(2, w - 1)).std()
        )
        
        logger.debug(
            f"Added {mean_col}: {df[mean_col].notna().sum()} valid values"
        )
    
    return df


def add_trend_features(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    group_col: str = "region",
) -> pd.DataFrame:
    """
    Add trend-related features derived from historical patterns.
    
    Features:
    - loss_trend_3yr: slope of linear fit over past 3 years (ha/year)
    - loss_change_1yr: year-over-year change
    - loss_change_pct_1yr: year-over-year percentage change
    - loss_acceleration: change in the year-over-year change
    
    All computed from past data only (shifted to avoid leakage).
    """
    df = df.sort_values([group_col, "year"]).copy()
    
    # Year-over-year change (using lag1 computed from shifted values)
    df["loss_change_1yr"] = (
        df[target_col] -
        df.groupby(group_col)[target_col].shift(1)
    )
    # Shift to avoid leakage: the change feature for year Y uses Y-1 and Y-2
    df["loss_change_1yr_lagged"] = df.groupby(group_col)["loss_change_1yr"].shift(1)
    
    # Percentage change
    prev = df.groupby(group_col)[target_col].shift(1)
    df["loss_change_pct_1yr"] = np.where(
        prev > 0,
        ((df[target_col] - prev) / prev) * 100.0,
        np.nan,
    )
    df["loss_change_pct_1yr_lagged"] = df.groupby(group_col)[
        "loss_change_pct_1yr"
    ].shift(1)
    
    # Acceleration (second difference)
    df["loss_acceleration"] = df.groupby(group_col)[
        "loss_change_1yr_lagged"
    ].diff()
    
    # 3-year trend slope (computed from past observations)
    def _compute_trend_slope(region_name, region_df):
        """Linear regression slope over a trailing 3-year window."""
        slopes = []
        values = region_df[target_col].values
        for i in range(len(values)):
            if i < 3:
                slopes.append(np.nan)
            else:
                window = values[i-3:i]  # past 3 values, not including current
                if np.all(np.isfinite(window)):
                    x = np.arange(3)
                    slope = np.polyfit(x, window, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=region_df.index)
    
    trend_parts = []
    for region_name, region_df in df.groupby(group_col):
        trend_parts.append(_compute_trend_slope(region_name, region_df))
    df["loss_trend_3yr"] = pd.concat(trend_parts)
    
    # Clean up intermediate columns
    df = df.drop(columns=["loss_change_1yr", "loss_change_pct_1yr"], errors="ignore")
    
    return df


def add_rate_features(
    df: pd.DataFrame,
    group_col: str = "region",
) -> pd.DataFrame:
    """
    Add lagged loss-rate features.
    
    - loss_rate_pct_lag1: previous year's loss rate
    - cumulative_loss_pct_lag1: previous year's cumulative loss %
    """
    df = df.sort_values([group_col, "year"]).copy()
    
    if "loss_rate_pct" in df.columns:
        df["loss_rate_pct_lag1"] = df.groupby(group_col)[
            "loss_rate_pct"
        ].shift(1)
    
    if "cumulative_loss_pct" in df.columns:
        df["cumulative_loss_pct_lag1"] = df.groupby(group_col)[
            "cumulative_loss_pct"
        ].shift(1)
    
    return df


def add_region_encoding(
    df: pd.DataFrame,
    region_col: str = "region",
) -> pd.DataFrame:
    """
    Add label-encoded region identifier.
    
    Tree-based models can use this directly. For linear models,
    one-hot encoding may be needed (handled in training).
    """
    df = df.copy()
    
    regions = sorted(df[region_col].unique())
    region_map = {r: i for i, r in enumerate(regions)}
    df["region_encoded"] = df[region_col].map(region_map)
    
    logger.info(f"Encoded {len(regions)} regions")
    return df


def add_variability_features(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    group_col: str = "region",
) -> pd.DataFrame:
    """
    Add coefficient of variation over a trailing window.
    
    CV = std / mean (from past observations only).
    """
    df = df.sort_values([group_col, "year"]).copy()
    
    shifted = df.groupby(group_col)[target_col].shift(1)
    
    roll_mean = shifted.groupby(df[group_col]).transform(
        lambda x: x.rolling(window=5, min_periods=3).mean()
    )
    roll_std = shifted.groupby(df[group_col]).transform(
        lambda x: x.rolling(window=5, min_periods=3).std()
    )
    
    df["loss_cv_5yr"] = np.where(
        roll_mean > 0, roll_std / roll_mean, np.nan
    )
    
    return df


def build_feature_set(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    lags: List[int] = None,
    windows: List[int] = None,
    group_col: str = "region",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Applies all feature groups in order:
      1. Lag features
      2. Rolling features
      3. Trend features
      4. Rate features
      5. Variability features
      6. Region encoding
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format preprocessed dataset.
    target_col : str
    lags : list of int
    windows : list of int
    group_col : str
    
    Returns
    -------
    pd.DataFrame
        Dataset with all engineered features.
    """
    if lags is None:
        lags = [1, 2, 3]
    if windows is None:
        windows = [3, 5]
    
    logger.info("Starting feature engineering...")
    
    df = add_lag_features(df, target_col, lags, group_col)
    df = add_rolling_features(df, target_col, windows, group_col)
    df = add_trend_features(df, target_col, group_col)
    df = add_rate_features(df, group_col)
    df = add_variability_features(df, target_col, group_col)
    df = add_region_encoding(df, group_col)
    
    n_features = len(get_feature_columns(df))
    logger.info(f"Feature engineering complete: {n_features} features created")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature columns suitable for model input.
    
    Excludes identifier columns, target column, and metadata.
    """
    exclude = {
        "country", "region", "year", "tc_loss_ha", "loss_rate_pct",
        "cumulative_loss_ha", "cumulative_loss_pct", "carbon_emissions_Mg_CO2e",
        "total_area_ha", "extent_2000_ha", "extent_2010_ha",
        "gain_2000_2020_ha", "threshold",
        # Intermediate columns that shouldn't be features
        "loss_change_1yr", "loss_change_pct_1yr",
    }
    
    features = [
        c for c in df.columns
        if c not in exclude
        and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]
    
    return features


FEATURE_DOCUMENTATION = """
Feature Documentation
=====================

Target Variable:
  tc_loss_ha — Annual tree-cover loss in hectares for the given region and year.

Lag Features:
  tc_loss_ha_lag1 — Tree-cover loss in the previous year.
    Rationale: Most direct predictor; deforestation tends to persist year-over-year.
    
  tc_loss_ha_lag2 — Tree-cover loss two years prior.
    Rationale: Captures medium-term persistence.
    
  tc_loss_ha_lag3 — Tree-cover loss three years prior.
    Rationale: Captures longer-term baseline deforestation level.

Rolling Features:
  tc_loss_ha_roll_mean_3yr — Mean loss over the prior 3 years.
    Rationale: Smoothed estimate of recent deforestation activity.
    
  tc_loss_ha_roll_std_3yr — Standard deviation of loss over the prior 3 years.
    Rationale: Captures volatility in recent deforestation.
    
  tc_loss_ha_roll_mean_5yr — Mean loss over the prior 5 years.
    Rationale: Longer-term average deforestation level.
    
  tc_loss_ha_roll_std_5yr — Standard deviation of loss over the prior 5 years.
    Rationale: Longer-term volatility measure.

Trend Features:
  loss_change_1yr_lagged — Year-over-year change in loss (previous year).
    Rationale: Direction and magnitude of recent deforestation change.
    
  loss_change_pct_1yr_lagged — Percentage year-over-year change (previous year).
    Rationale: Relative rate of change.
    
  loss_acceleration — Change in the year-over-year change.
    Rationale: Whether deforestation is accelerating or decelerating.
    
  loss_trend_3yr — Linear trend slope over the prior 3 years (ha/year).
    Rationale: Direction and steepness of recent trend.

Rate Features:
  loss_rate_pct_lag1 — Previous year's forest-loss rate (%).
    Rationale: Proportional loss relative to baseline.
    
  cumulative_loss_pct_lag1 — Previous year's cumulative loss as % of baseline.
    Rationale: How much of original forest has been lost.

Variability:
  loss_cv_5yr — Coefficient of variation over trailing 5 years.
    Rationale: Relative variability — identifies regions with erratic deforestation.

Region Encoding:
  region_encoded — Integer-encoded region identifier.
    Rationale: Allows tree-based models to learn region-specific patterns.

Leakage Prevention:
  - All lag and rolling features use shift(1) or earlier data only.
  - Rolling windows exclude the current observation.
  - Trend features are computed from historical values only.
  - No future carbon emissions, future loss values, or test-period statistics
    are used as features.
"""
