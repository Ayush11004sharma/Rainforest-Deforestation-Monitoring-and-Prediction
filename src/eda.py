"""
Exploratory Data Analysis Module
=================================
Generates numerical summaries and visualizations for deforestation data.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("deforestation_ml.eda")


def summary_statistics(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
) -> Dict:
    """
    Compute key summary statistics for the preprocessed dataset.
    
    Returns
    -------
    dict with overview metrics, annual summaries, and regional summaries.
    """
    stats = {}
    
    # Overview
    stats["total_loss_ha"] = float(df[target_col].sum())
    stats["n_regions"] = int(df["region"].nunique())
    stats["year_range"] = (int(df["year"].min()), int(df["year"].max()))
    stats["n_years"] = int(df["year"].nunique())
    stats["total_observations"] = len(df)
    
    # Annual totals
    annual = df.groupby("year")[target_col].sum().reset_index()
    annual.columns = ["year", "total_loss_ha"]
    stats["annual_totals"] = annual
    stats["avg_annual_loss"] = float(annual["total_loss_ha"].mean())
    stats["peak_year"] = int(annual.loc[annual["total_loss_ha"].idxmax(), "year"])
    stats["peak_loss"] = float(annual["total_loss_ha"].max())
    
    # Regional totals
    regional = (
        df.groupby("region")[target_col]
        .agg(["sum", "mean", "std", "min", "max"])
        .reset_index()
    )
    regional.columns = [
        "region", "total_loss_ha", "avg_annual_loss_ha",
        "std_annual_loss_ha", "min_annual_loss_ha", "max_annual_loss_ha",
    ]
    regional = regional.sort_values("total_loss_ha", ascending=False)
    stats["regional_summary"] = regional
    
    # Top/bottom regions
    stats["top_5_regions"] = regional.head(5)["region"].tolist()
    stats["bottom_5_regions"] = regional.tail(5)["region"].tolist()
    
    # Loss rate statistics
    if "loss_rate_pct" in df.columns:
        stats["avg_loss_rate"] = float(df["loss_rate_pct"].mean())
        stats["max_loss_rate"] = float(df["loss_rate_pct"].max())
        rate_by_region = df.groupby("region")["loss_rate_pct"].mean()
        stats["highest_rate_region"] = rate_by_region.idxmax()
        stats["highest_avg_rate"] = float(rate_by_region.max())
    
    # Trend analysis: regions with increasing/decreasing trends
    trends = compute_regional_trends(df, target_col)
    stats["increasing_regions"] = trends[trends["trend_slope"] > 0]["region"].tolist()
    stats["decreasing_regions"] = trends[trends["trend_slope"] < 0]["region"].tolist()
    
    return stats


def compute_regional_trends(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
) -> pd.DataFrame:
    """
    Compute linear trend (slope) for each region over the full period.
    
    Positive slope = increasing deforestation over time.
    Negative slope = decreasing deforestation over time.
    """
    trends = []
    for region in df["region"].unique():
        region_data = df[df["region"] == region].sort_values("year")
        if len(region_data) < 3:
            continue
        
        x = region_data["year"].values.astype(float)
        y = region_data[target_col].values.astype(float)
        valid = np.isfinite(y)
        
        if valid.sum() >= 3:
            coeffs = np.polyfit(x[valid], y[valid], 1)
            slope = coeffs[0]
        else:
            slope = 0.0
        
        trends.append({
            "region": region,
            "trend_slope": round(slope, 1),
            "trend_direction": "Increasing" if slope > 0 else "Decreasing",
            "total_loss": float(y.sum()),
        })
    
    return pd.DataFrame(trends)


def correlation_analysis(
    df: pd.DataFrame,
    cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute pairwise correlations between numeric columns.
    """
    if cols is None:
        cols = ["tc_loss_ha", "loss_rate_pct", "cumulative_loss_ha"]
        if "carbon_emissions_Mg_CO2e" in df.columns:
            cols.append("carbon_emissions_Mg_CO2e")
    
    available = [c for c in cols if c in df.columns]
    return df[available].corr()


def missing_data_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Summary of missing values across the dataset."""
    missing = df.isnull().sum()
    total = len(df)
    
    summary = pd.DataFrame({
        "column": missing.index,
        "missing_count": missing.values,
        "missing_pct": (missing.values / total * 100).round(2),
    })
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    return summary


def ranking_table(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Create a ranking table of regions by forest loss.
    
    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    year : int, optional
        If specified, rank for a single year. Otherwise, rank by total.
    """
    if year is not None:
        data = df[df["year"] == year].copy()
        value_col = target_col
        title_suffix = f"in {year}"
    else:
        data = df.groupby("region").agg(
            total_loss=(target_col, "sum"),
            avg_loss=(target_col, "mean"),
        ).reset_index()
        value_col = "total_loss"
        title_suffix = "Total (2001–2022)"
    
    data = data.sort_values(
        value_col if value_col in data.columns else target_col,
        ascending=False,
    ).reset_index(drop=True)
    data.index += 1
    data.index.name = "Rank"
    
    return data
