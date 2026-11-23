"""
Data Loader Module
==================
Loads and validates the four raw CSV datasets from the Global Forest Change
(Hansen et al.) / GFW data portal.

Datasets:
  1. Country tree cover loss — wide-format annual tree-cover loss by country
  2. Subnational 1 tree cover loss — same at subnational (state/province) level
  3. Country carbon data — annual carbon emissions by country
  4. Subnational 1 carbon data — annual carbon emissions by subnational region
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger("deforestation_ml.data_loader")


def load_csv_safe(filepath: Path, description: str = "") -> pd.DataFrame:
    """
    Load a CSV file with validation.
    
    Parameters
    ----------
    filepath : Path
        Path to the CSV file.
    description : str
        Human-readable name for logging.
        
    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
        
    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset '{description}' not found at: {filepath}\n"
            f"Please ensure the CSV files are placed in the data/raw/ directory."
        )
    
    df = pd.read_csv(filepath, encoding="utf-8")
    logger.info(
        f"Loaded {description}: {df.shape[0]} rows × {df.shape[1]} columns "
        f"from {filepath.name}"
    )
    return df


def load_all_datasets(
    raw_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all four raw CSV datasets.
    
    Parameters
    ----------
    raw_dir : Path, optional
        Directory containing raw CSV files. Uses config default if not provided.
    
    Returns
    -------
    dict
        Keys: 'country_loss', 'subnational_loss', 'country_carbon', 'subnational_carbon'
    """
    from config import (
        COUNTRY_TREE_LOSS_FILE,
        SUBNATIONAL_TREE_LOSS_FILE,
        COUNTRY_CARBON_FILE,
        SUBNATIONAL_CARBON_FILE,
    )
    
    datasets = {}
    
    file_map = {
        "country_loss": (COUNTRY_TREE_LOSS_FILE, "Country Tree Cover Loss"),
        "subnational_loss": (SUBNATIONAL_TREE_LOSS_FILE, "Subnational Tree Cover Loss"),
        "country_carbon": (COUNTRY_CARBON_FILE, "Country Carbon Data"),
        "subnational_carbon": (SUBNATIONAL_CARBON_FILE, "Subnational Carbon Data"),
    }
    
    for key, (filepath, desc) in file_map.items():
        try:
            datasets[key] = load_csv_safe(filepath, desc)
        except FileNotFoundError as e:
            logger.warning(str(e))
            datasets[key] = None
    
    return datasets


def validate_dataset(df: pd.DataFrame, name: str) -> Dict:
    """
    Produce a data-quality report for a loaded dataset.
    
    Returns a dict with shape, columns, dtypes, missing counts,
    duplicate count, and sample rows.
    """
    if df is None:
        return {"status": "not_loaded", "name": name}
    
    report = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_column": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(3).to_dict(),
    }
    
    # Identify year columns
    year_cols = [c for c in df.columns if any(
        str(y) in c for y in range(2001, 2023)
    )]
    report["year_columns"] = year_cols
    report["year_range"] = _extract_year_range(year_cols)
    
    return report


def _extract_year_range(columns: list) -> Tuple[Optional[int], Optional[int]]:
    """Extract min and max year from column names."""
    import re
    years = []
    for col in columns:
        matches = re.findall(r"(\d{4})", col)
        years.extend(int(m) for m in matches if 2000 <= int(m) <= 2030)
    if years:
        return (min(years), max(years))
    return (None, None)


def get_threshold_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the tree-cover threshold column name."""
    candidates = ["threshold", "umd_tree_cover_density_2000__threshold"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def get_extent_column(df: pd.DataFrame) -> Optional[str]:
    """Identify the year-2000 extent column."""
    candidates = ["extent_2000_ha", "umd_tree_cover_extent_2000__ha"]
    for c in candidates:
        if c in df.columns:
            return c
    return None
