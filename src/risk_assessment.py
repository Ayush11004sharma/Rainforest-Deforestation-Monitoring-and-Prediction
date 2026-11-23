"""
Risk Assessment Module
======================
Assigns transparent deforestation-risk categories to regions.

IMPORTANT DISCLAIMERS:
  - This is an analytical risk assessment for academic purposes.
  - It is NOT a verified prediction of illegal logging or an official
    conservation classification.
  - Thresholds are configurable and documented — not presented as
    scientifically established standards.
  - The assessment is based on available GFW tree-cover-loss data only.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger("deforestation_ml.risk_assessment")


def compute_risk_indicators(
    df: pd.DataFrame,
    last_n_years: int = 3,
    target_col: str = "tc_loss_ha",
) -> pd.DataFrame:
    """
    Compute risk indicators for each region.
    
    Indicators:
      1. recent_loss_rate: Average annual loss rate over the last N years (%)
      2. loss_trend: Slope of linear fit over the last N years (ha/year)
      3. cumulative_loss_pct: Total cumulative loss as % of baseline extent
      4. recent_loss_ha: Average annual loss over the last N years (ha)
    
    Parameters
    ----------
    df : pd.DataFrame
        Long-format preprocessed dataset with loss rate and cumulative loss.
    last_n_years : int
        Number of recent years to consider.
    target_col : str
    
    Returns
    -------
    pd.DataFrame
        One row per region with risk indicators.
    """
    if df.empty:
        return pd.DataFrame()
    
    max_year = df["year"].max()
    recent_years = list(range(max_year - last_n_years + 1, max_year + 1))
    
    indicators = []
    
    for region in df["region"].unique():
        region_data = df[df["region"] == region].sort_values("year")
        recent = region_data[region_data["year"].isin(recent_years)]
        
        if recent.empty:
            continue
        
        # Recent average loss rate (%)
        recent_loss_rate = (
            recent["loss_rate_pct"].mean()
            if "loss_rate_pct" in recent.columns
            else np.nan
        )
        
        # Recent average loss (ha)
        recent_loss_ha = recent[target_col].mean()
        
        # Loss trend (slope over recent years)
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = recent[target_col].values
            valid = np.isfinite(y)
            if valid.sum() >= 2:
                slope = np.polyfit(x[valid], y[valid], 1)[0]
            else:
                slope = 0.0
        else:
            slope = 0.0
        
        # Normalized trend: positive = increasing deforestation
        loss_trend = slope
        
        # Cumulative loss %
        cumulative_loss_pct = (
            region_data["cumulative_loss_pct"].iloc[-1]
            if "cumulative_loss_pct" in region_data.columns
            else np.nan
        )
        
        # Baseline extent
        extent = (
            region_data["extent_2000_ha"].iloc[0]
            if "extent_2000_ha" in region_data.columns
            else np.nan
        )
        
        indicators.append({
            "region": region,
            "recent_loss_rate": round(recent_loss_rate, 4) if np.isfinite(recent_loss_rate) else 0.0,
            "recent_loss_ha": round(recent_loss_ha, 0),
            "loss_trend_ha_yr": round(loss_trend, 0),
            "cumulative_loss_pct": round(cumulative_loss_pct, 2) if np.isfinite(cumulative_loss_pct) else 0.0,
            "extent_2000_ha": extent,
            "last_year": int(max_year),
            "n_recent_years": len(recent),
        })
    
    return pd.DataFrame(indicators)


def normalize_indicator(
    series: pd.Series, higher_is_riskier: bool = True
) -> pd.Series:
    """
    Min-max normalize an indicator to [0, 1].
    
    Parameters
    ----------
    series : pd.Series
    higher_is_riskier : bool
        If True, higher raw values = higher risk score.
    """
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    
    normalized = (series - min_val) / (max_val - min_val)
    
    if not higher_is_riskier:
        normalized = 1 - normalized
    
    return normalized


def compute_risk_scores(
    indicators_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    predicted_future_loss: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute composite risk scores for each region.
    
    The risk score is a weighted average of normalized indicators.
    
    Parameters
    ----------
    indicators_df : pd.DataFrame
        Output from compute_risk_indicators().
    weights : dict, optional
        Weights for each indicator. Uses config defaults if not provided.
    predicted_future_loss : dict, optional
        Dict mapping region → predicted future loss (ha). Added as an indicator.
    
    Returns
    -------
    pd.DataFrame
        With added 'risk_score' (0–1) and 'risk_category' columns.
    """
    from config import RISK_WEIGHTS
    
    if weights is None:
        weights = RISK_WEIGHTS
    
    df = indicators_df.copy()
    
    # Add predicted future loss if available
    if predicted_future_loss:
        df["predicted_future_loss"] = df["region"].map(predicted_future_loss).fillna(0)
    else:
        df["predicted_future_loss"] = 0.0
        # Reduce weight for missing predictor
        weights = weights.copy()
        other_weight = weights.pop("predicted_future_loss", 0)
        remaining_sum = sum(weights.values())
        if remaining_sum > 0:
            for k in weights:
                weights[k] += other_weight * weights[k] / remaining_sum
    
    # Normalize each indicator
    score_components = {}
    
    if "recent_loss_rate" in df.columns:
        score_components["recent_loss_rate"] = normalize_indicator(
            df["recent_loss_rate"], higher_is_riskier=True
        )
    
    if "loss_trend_ha_yr" in df.columns:
        score_components["loss_trend"] = normalize_indicator(
            df["loss_trend_ha_yr"], higher_is_riskier=True
        )
    
    if "cumulative_loss_pct" in df.columns:
        score_components["cumulative_loss_pct"] = normalize_indicator(
            df["cumulative_loss_pct"], higher_is_riskier=True
        )
    
    if "predicted_future_loss" in df.columns:
        score_components["predicted_future_loss"] = normalize_indicator(
            df["predicted_future_loss"], higher_is_riskier=True
        )
    
    # Weighted average
    total_weight = 0
    risk_score = pd.Series(0.0, index=df.index)
    
    for key, norm_values in score_components.items():
        w = weights.get(key, 0.0)
        risk_score += norm_values * w
        total_weight += w
    
    if total_weight > 0:
        risk_score /= total_weight
    
    df["risk_score"] = risk_score.round(4)
    
    # Store normalized components for transparency
    for key, norm_values in score_components.items():
        df[f"norm_{key}"] = norm_values.round(4)
    
    return df


def assign_risk_categories(
    df: pd.DataFrame,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Assign risk categories based on the composite risk score.
    
    Default thresholds (on the 0–1 normalized score):
      - Low: [0.0, 0.25)
      - Moderate: [0.25, 0.50)
      - High: [0.50, 0.75)
      - Very High: [0.75, 1.0]
    
    These are CONFIGURABLE thresholds using percentile-based breakpoints
    from the score distribution. They are NOT scientifically established
    standards.
    """
    df = df.copy()
    
    if thresholds is None:
        # Use quartile-based thresholds from actual data
        q25 = df["risk_score"].quantile(0.25)
        q50 = df["risk_score"].quantile(0.50)
        q75 = df["risk_score"].quantile(0.75)
        
        thresholds = {
            "Low": (0.0, q25),
            "Moderate": (q25, q50),
            "High": (q50, q75),
            "Very High": (q75, 1.01),
        }
    
    def categorize(score):
        for category, (low, high) in thresholds.items():
            if low <= score < high:
                return category
        return "Very High"  # Default for edge case
    
    df["risk_category"] = df["risk_score"].apply(categorize)
    
    # Store thresholds for documentation
    df.attrs["risk_thresholds"] = thresholds
    
    return df


def generate_risk_report(
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a formatted risk report sorted by risk score (highest first).
    
    Returns
    -------
    pd.DataFrame
        Risk report with all indicators and categories.
    """
    report_cols = [
        "region", "risk_category", "risk_score",
        "recent_loss_rate", "recent_loss_ha", "loss_trend_ha_yr",
        "cumulative_loss_pct", "extent_2000_ha",
    ]
    
    available_cols = [c for c in report_cols if c in risk_df.columns]
    
    report = risk_df[available_cols].sort_values(
        "risk_score", ascending=False
    ).reset_index(drop=True)
    
    return report


def full_risk_assessment(
    processed_df: pd.DataFrame,
    predicted_future: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    last_n_years: int = 3,
) -> pd.DataFrame:
    """
    Run the complete risk assessment pipeline.
    
    Parameters
    ----------
    processed_df : pd.DataFrame
        Long-format preprocessed dataset.
    predicted_future : dict, optional
        Region → predicted future loss.
    weights : dict, optional
        Risk indicator weights.
    last_n_years : int
    
    Returns
    -------
    pd.DataFrame
        Complete risk report with scores and categories.
    """
    # Step 1: Compute indicators
    indicators = compute_risk_indicators(processed_df, last_n_years)
    
    # Step 2: Compute scores
    scored = compute_risk_scores(indicators, weights, predicted_future)
    
    # Step 3: Assign categories
    categorized = assign_risk_categories(scored)
    
    # Step 4: Generate report
    report = generate_risk_report(categorized)
    
    logger.info(
        f"Risk assessment complete: {len(report)} regions. "
        f"Distribution: {categorized['risk_category'].value_counts().to_dict()}"
    )
    
    return categorized
