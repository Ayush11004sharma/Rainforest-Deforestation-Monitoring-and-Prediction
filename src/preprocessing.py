"""
Preprocessing Module
====================
Transforms raw wide-format GFW datasets into clean long-format analytical
datasets ready for EDA, feature engineering, and modeling.

Key transformations:
  - Filter to configured tree-cover threshold (default: 30%).
  - Melt annual columns into long-format (region × year rows).
  - Calculate annual forest-loss rate.
  - Merge carbon emissions data where available.
  - Handle missing values and zero denominators.
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger("deforestation_ml.preprocessing")


# ════════════════════════════════════════════════════════
# Column Name Extraction
# ════════════════════════════════════════════════════════

def extract_annual_columns(
    df: pd.DataFrame, prefix: str = "tc_loss_ha_"
) -> List[str]:
    """
    Find all annual data columns matching a prefix pattern.
    
    Parameters
    ----------
    df : pd.DataFrame
    prefix : str
        Column name prefix (e.g. 'tc_loss_ha_' or 'gfw_forest_carbon_gross_emissions_')
    
    Returns
    -------
    list of str
        Matching column names sorted by year.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d{{4}})")
    cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def extract_year_from_column(col_name: str) -> Optional[int]:
    """Extract a 4-digit year from a column name."""
    m = re.search(r"(\d{4})", col_name)
    if m:
        year = int(m.group(1))
        if 2000 <= year <= 2030:
            return year
    return None


# ════════════════════════════════════════════════════════
# Threshold Filtering
# ════════════════════════════════════════════════════════

def filter_by_threshold(
    df: pd.DataFrame,
    threshold: int = 30,
    threshold_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Filter dataset to rows matching the specified tree-cover threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
    threshold : int
        Tree-cover density threshold (%).
    threshold_col : str, optional
        Name of the threshold column. Auto-detected if not provided.
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    if threshold_col is None:
        for candidate in ["threshold", "umd_tree_cover_density_2000__threshold"]:
            if candidate in df.columns:
                threshold_col = candidate
                break
    
    if threshold_col is None or threshold_col not in df.columns:
        logger.warning("No threshold column found. Returning data unfiltered.")
        return df.copy()
    
    available = sorted(df[threshold_col].unique())
    if threshold not in available:
        logger.warning(
            f"Threshold {threshold} not in available values {available}. "
            f"Using closest available."
        )
        threshold = min(available, key=lambda x: abs(x - threshold))
    
    filtered = df[df[threshold_col] == threshold].copy()
    logger.info(
        f"Filtered to threshold={threshold}: {len(filtered)} rows "
        f"(from {len(df)})"
    )
    return filtered


# ════════════════════════════════════════════════════════
# Wide-to-Long Conversion
# ════════════════════════════════════════════════════════

def melt_tree_cover_loss(
    df: pd.DataFrame,
    id_vars: Optional[List[str]] = None,
    prefix: str = "tc_loss_ha_",
    value_name: str = "tc_loss_ha",
) -> pd.DataFrame:
    """
    Melt wide-format annual tree-cover-loss columns into long format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataset with columns like tc_loss_ha_2001, tc_loss_ha_2002, ...
    id_vars : list of str, optional
        Identifier columns to keep. Auto-detected if not provided.
    prefix : str
        Column name prefix for annual values.
    value_name : str
        Name for the melted value column.
        
    Returns
    -------
    pd.DataFrame
        Long-format with columns: [id_vars..., 'year', value_name]
    """
    annual_cols = extract_annual_columns(df, prefix)
    
    if not annual_cols:
        raise ValueError(f"No annual columns found with prefix '{prefix}'")
    
    if id_vars is None:
        id_vars = [c for c in df.columns if c not in annual_cols]
    
    # Melt
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=annual_cols,
        var_name="year_col",
        value_name=value_name,
    )
    
    # Extract year
    long_df["year"] = long_df["year_col"].apply(extract_year_from_column)
    long_df = long_df.drop(columns=["year_col"])
    
    # Convert to numeric
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    
    logger.info(
        f"Melted {len(annual_cols)} annual columns → {len(long_df)} rows, "
        f"years {long_df['year'].min()}–{long_df['year'].max()}"
    )
    return long_df


def melt_carbon_emissions(
    df: pd.DataFrame,
    id_vars: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Melt wide-format annual carbon emissions columns into long format.
    
    Carbon columns follow pattern:
        gfw_forest_carbon_gross_emissions_YYYY__Mg_CO2e
    """
    # Find annual carbon columns
    pattern = re.compile(
        r"gfw_forest_carbon_gross_emissions_(\d{4})__Mg_CO2e"
    )
    annual_cols = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            annual_cols.append((int(m.group(1)), c))
    annual_cols.sort()
    annual_col_names = [c for _, c in annual_cols]
    
    if not annual_col_names:
        logger.warning("No annual carbon emission columns found.")
        return pd.DataFrame()
    
    # Identify static/metadata columns
    static_cols = [
        "country",
        "subnational1",
        "umd_tree_cover_density_2000__threshold",
        "umd_tree_cover_extent_2000__ha",
        "gfw_aboveground_carbon_stocks_2000__Mg_C",
        "avg_gfw_aboveground_carbon_stocks_2000__Mg_C_ha-1",
        "gfw_forest_carbon_gross_emissions__Mg_CO2e_yr-1",
        "gfw_forest_carbon_gross_removals__Mg_CO2_yr-1",
        "gfw_forest_carbon_net_flux__Mg_CO2e_yr-1",
    ]
    if id_vars is None:
        id_vars = [c for c in static_cols if c in df.columns]
    
    long_df = df.melt(
        id_vars=id_vars,
        value_vars=annual_col_names,
        var_name="year_col",
        value_name="carbon_emissions_Mg_CO2e",
    )
    
    long_df["year"] = long_df["year_col"].apply(extract_year_from_column)
    long_df = long_df.drop(columns=["year_col"])
    long_df["carbon_emissions_Mg_CO2e"] = pd.to_numeric(
        long_df["carbon_emissions_Mg_CO2e"], errors="coerce"
    )
    
    return long_df


# ════════════════════════════════════════════════════════
# Forest-Loss Rate Calculation
# ════════════════════════════════════════════════════════

def calculate_loss_rate(
    df: pd.DataFrame,
    loss_col: str = "tc_loss_ha",
    extent_col: str = "extent_2000_ha",
    rate_col: str = "loss_rate_pct",
) -> pd.DataFrame:
    """
    Calculate annual forest-loss rate (%).
    
    Formula:
        loss_rate_pct = (tc_loss_ha / extent_2000_ha) × 100
    
    Notes
    -----
    - extent_2000_ha is the baseline tree-cover extent in hectares at the
      year-2000 reference date for the selected threshold.
    - This rate is computed relative to the fixed 2000 baseline, NOT the
      remaining forest in each year. It measures what fraction of the
      original forest was lost in a given year.
    - Zero or missing extent values produce NaN rates (not infinity).
    """
    df = df.copy()
    
    # Ensure numeric
    df[loss_col] = pd.to_numeric(df[loss_col], errors="coerce")
    if extent_col in df.columns:
        df[extent_col] = pd.to_numeric(df[extent_col], errors="coerce")
        # Safe division: avoid division by zero
        df[rate_col] = np.where(
            (df[extent_col] > 0) & df[extent_col].notna(),
            (df[loss_col] / df[extent_col]) * 100.0,
            np.nan,
        )
    else:
        logger.warning(
            f"Extent column '{extent_col}' not found. Cannot compute loss rate."
        )
        df[rate_col] = np.nan
    
    return df


# ════════════════════════════════════════════════════════
# Full Preprocessing Pipeline
# ════════════════════════════════════════════════════════

def preprocess_subnational_tree_loss(
    df: pd.DataFrame,
    threshold: int = 30,
    country: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for subnational tree-cover loss data.
    
    Steps:
      1. Filter to specified threshold.
      2. Optionally filter to a single country.
      3. Melt annual columns to long format.
      4. Calculate annual loss rate.
      5. Add cumulative loss.
      6. Clean and validate.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw subnational tree-cover loss dataset (wide format).
    threshold : int
        Tree-cover density threshold (default: 30).
    country : str, optional
        Filter to a single country (e.g., 'Brazil').
    
    Returns
    -------
    pd.DataFrame
        Long-format analytical dataset.
    """
    # Step 1: Filter by threshold
    df_filtered = filter_by_threshold(df, threshold)
    
    # Step 2: Filter by country
    if country is not None:
        if "country" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["country"] == country].copy()
            logger.info(f"Filtered to country='{country}': {len(df_filtered)} rows")
        else:
            logger.warning("No 'country' column found for country filtering.")
    
    # Step 3: Identify id_vars (non-annual columns)
    annual_cols = extract_annual_columns(df_filtered, "tc_loss_ha_")
    id_vars = [c for c in df_filtered.columns if c not in annual_cols]
    
    # Step 4: Melt to long format
    long_df = melt_tree_cover_loss(df_filtered, id_vars=id_vars)
    
    # Step 5: Rename/standardize columns
    rename_map = {
        "subnational1": "region",
        "extent_2000_ha": "extent_2000_ha",
        "extent_2010_ha": "extent_2010_ha",
        "area_ha": "total_area_ha",
        "gain_2000-2020_ha": "gain_2000_2020_ha",
    }
    for old, new in rename_map.items():
        if old in long_df.columns and old != new:
            long_df = long_df.rename(columns={old: new})
    
    # Step 6: Calculate loss rate
    long_df = calculate_loss_rate(long_df)
    
    # Step 7: Add cumulative loss per region
    long_df = long_df.sort_values(["region", "year"]).reset_index(drop=True)
    long_df["cumulative_loss_ha"] = long_df.groupby("region")["tc_loss_ha"].cumsum()
    
    # Step 8: Cumulative loss as % of baseline
    if "extent_2000_ha" in long_df.columns:
        long_df["cumulative_loss_pct"] = np.where(
            long_df["extent_2000_ha"] > 0,
            (long_df["cumulative_loss_ha"] / long_df["extent_2000_ha"]) * 100.0,
            np.nan,
        )
    
    # Step 9: Remove threshold column (no longer needed)
    for col in ["threshold", "umd_tree_cover_density_2000__threshold"]:
        if col in long_df.columns:
            long_df = long_df.drop(columns=[col])
    
    logger.info(
        f"Preprocessing complete: {long_df.shape[0]} rows, "
        f"{long_df['region'].nunique()} regions, "
        f"years {long_df['year'].min()}–{long_df['year'].max()}"
    )
    
    return long_df


def preprocess_carbon_data(
    df: pd.DataFrame,
    threshold: int = 30,
    country: Optional[str] = None,
) -> pd.DataFrame:
    """
    Preprocess carbon emissions data to long format.
    
    Returns
    -------
    pd.DataFrame
        Long-format with columns including 'country', 'region' (if subnational),
        'year', 'carbon_emissions_Mg_CO2e'.
    """
    # Determine threshold column
    threshold_col = None
    for c in ["threshold", "umd_tree_cover_density_2000__threshold"]:
        if c in df.columns:
            threshold_col = c
            break
    
    if threshold_col:
        df_filtered = filter_by_threshold(df, threshold, threshold_col)
    else:
        df_filtered = df.copy()
    
    if country and "country" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["country"] == country].copy()
    
    long_df = melt_carbon_emissions(df_filtered)
    
    # Rename subnational1 to region if present
    if "subnational1" in long_df.columns:
        long_df = long_df.rename(columns={"subnational1": "region"})
    
    # Drop threshold column
    for col in ["umd_tree_cover_density_2000__threshold"]:
        if col in long_df.columns:
            long_df = long_df.drop(columns=[col])
    
    return long_df


def merge_carbon_with_loss(
    loss_df: pd.DataFrame,
    carbon_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge carbon emissions into the tree-cover loss dataset.
    
    Matches on country + region (if subnational) + year.
    """
    if carbon_df is None or carbon_df.empty:
        logger.warning("No carbon data available for merging.")
        loss_df["carbon_emissions_Mg_CO2e"] = np.nan
        return loss_df
    
    # Determine merge keys
    merge_keys = ["year"]
    if "country" in loss_df.columns and "country" in carbon_df.columns:
        merge_keys.append("country")
    if "region" in loss_df.columns and "region" in carbon_df.columns:
        merge_keys.append("region")
    
    # Select only the key columns and emissions from carbon data
    carbon_cols = merge_keys + ["carbon_emissions_Mg_CO2e"]
    carbon_merge = carbon_df[
        [c for c in carbon_cols if c in carbon_df.columns]
    ].copy()
    
    # Drop duplicates on merge keys
    carbon_merge = carbon_merge.drop_duplicates(subset=merge_keys)
    
    merged = loss_df.merge(carbon_merge, on=merge_keys, how="left")
    
    n_matched = merged["carbon_emissions_Mg_CO2e"].notna().sum()
    logger.info(
        f"Merged carbon data: {n_matched}/{len(merged)} rows matched "
        f"({n_matched/len(merged)*100:.1f}%)"
    )
    
    return merged


def save_processed_data(
    df: pd.DataFrame,
    filename: str = "subnational_loss_long.csv",
    output_dir: Optional[Path] = None,
) -> Path:
    """Save processed dataset to CSV."""
    from config import PROCESSED_DATA_DIR
    
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data: {filepath} ({len(df)} rows)")
    return filepath


def generate_data_quality_report(
    df: pd.DataFrame,
    name: str = "dataset",
) -> Dict:
    """
    Generate a comprehensive data-quality report.
    
    Returns
    -------
    dict
        Report with missing values, duplicates, value ranges, and more.
    """
    report = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
    
    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    report["numeric_stats"] = {}
    for col in numeric_cols:
        report["numeric_stats"][col] = {
            "min": float(df[col].min()) if df[col].notna().any() else None,
            "max": float(df[col].max()) if df[col].notna().any() else None,
            "mean": float(df[col].mean()) if df[col].notna().any() else None,
            "zeros": int((df[col] == 0).sum()),
            "negatives": int((df[col] < 0).sum()),
        }
    
    # Categorical column summaries
    cat_cols = df.select_dtypes(include=["object"]).columns
    report["categorical_stats"] = {}
    for col in cat_cols:
        report["categorical_stats"][col] = {
            "unique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(5).to_dict(),
        }
    
    if "year" in df.columns:
        report["year_range"] = (int(df["year"].min()), int(df["year"].max()))
        report["years_available"] = sorted(df["year"].unique().tolist())
    
    if "region" in df.columns:
        report["regions"] = sorted(df["region"].unique().tolist())
        report["n_regions"] = int(df["region"].nunique())
    
    return report
