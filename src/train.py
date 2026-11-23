"""
Model Training Module
=====================
Trains and compares multiple ML models for deforestation prediction.

Models:
  A. Historical Baseline — uses previous year's loss as prediction
  B. Linear Regression — interpretable linear model
  C. Random Forest Regressor — nonlinear ensemble
  D. XGBoost Regressor — gradient boosting (optional)

Uses chronological (time-based) train/val/test splits.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger("deforestation_ml.train")


# ════════════════════════════════════════════════════════
# Chronological Split
# ════════════════════════════════════════════════════════

def chronological_split(
    df: pd.DataFrame,
    train_end_year: int = 2017,
    val_end_year: int = 2019,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "tc_loss_ha",
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data chronologically for time-series evaluation.
    
    Train: years ≤ train_end_year
    Validation: train_end_year < year ≤ val_end_year
    Test: year > val_end_year
    
    Only rows with complete feature sets (non-NaN in all features) are included.
    
    Parameters
    ----------
    df : pd.DataFrame
    train_end_year : int
    val_end_year : int
    feature_cols : list of str
    target_col : str
    
    Returns
    -------
    dict with keys 'train', 'val', 'test', each containing (X, y) tuple,
    plus 'meta' with the full dataframe rows for each split.
    """
    if feature_cols is None:
        from src.feature_engineering import get_feature_columns
        feature_cols = get_feature_columns(df)
    
    # Ensure features exist
    available_features = [c for c in feature_cols if c in df.columns]
    if len(available_features) < len(feature_cols):
        missing = set(feature_cols) - set(available_features)
        logger.warning(f"Missing features: {missing}")
    
    # Drop rows where features or target are NaN
    valid_mask = df[available_features + [target_col]].notna().all(axis=1)
    df_valid = df[valid_mask].copy()
    
    train = df_valid[df_valid["year"] <= train_end_year]
    val = df_valid[
        (df_valid["year"] > train_end_year) & (df_valid["year"] <= val_end_year)
    ]
    test = df_valid[df_valid["year"] > val_end_year]
    
    splits = {
        "train": (train[available_features], train[target_col]),
        "val": (val[available_features], val[target_col]),
        "test": (test[available_features], test[target_col]),
        "meta_train": train[["region", "year", target_col]].copy(),
        "meta_val": val[["region", "year", target_col]].copy(),
        "meta_test": test[["region", "year", target_col]].copy(),
        "feature_cols": available_features,
    }
    
    logger.info(
        f"Chronological split — "
        f"Train: {len(train)} rows (≤{train_end_year}), "
        f"Val: {len(val)} rows ({train_end_year+1}–{val_end_year}), "
        f"Test: {len(test)} rows (>{val_end_year})"
    )
    
    return splits


# ════════════════════════════════════════════════════════
# Baseline Model
# ════════════════════════════════════════════════════════

class HistoricalBaseline:
    """
    Model A: Naive baseline that predicts the previous year's loss.
    
    This is equivalent to saying 'next year's deforestation will be the
    same as this year's'. Provides a lower bound for model comparison.
    """
    
    def __init__(self):
        self.name = "Historical Baseline"
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalBaseline":
        """No fitting needed — this model uses lag1 directly."""
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the lag1 feature."""
        if "tc_loss_ha_lag1" in X.columns:
            preds = X["tc_loss_ha_lag1"].values.copy()
        else:
            # Fallback: use first lag-like column
            lag_cols = [c for c in X.columns if "lag1" in c]
            if lag_cols:
                preds = X[lag_cols[0]].values.copy()
            else:
                preds = np.zeros(len(X))
                logger.warning("No lag1 feature found for baseline model.")
        
        # Replace NaN with 0
        preds = np.nan_to_num(preds, nan=0.0)
        return preds


# ════════════════════════════════════════════════════════
# Model Training Functions
# ════════════════════════════════════════════════════════

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[LinearRegression, SimpleImputer, StandardScaler]:
    """
    Train Model B: Linear Regression with imputation and scaling.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)
    
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    
    logger.info(f"Linear Regression trained. R² (train): {model.score(X_scaled, y_train):.4f}")
    return model, imputer, scaler


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = 15,
    random_state: int = 42,
) -> Tuple[RandomForestRegressor, SimpleImputer]:
    """
    Train Model C: Random Forest Regressor.
    """
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=5,
    )
    model.fit(X_imputed, y_train)
    
    logger.info(
        f"Random Forest trained. R² (train): {model.score(X_imputed, y_train):.4f}"
    )
    return model, imputer


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Optional[Tuple[Any, SimpleImputer]]:
    """
    Train Model D: XGBoost Regressor (optional).
    
    Returns None if XGBoost is not installed.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        logger.warning(
            "XGBoost is not installed. Skipping Model D. "
            "Install with: pip install xgboost"
        )
        return None
    
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)
    
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_imputed, y_train)
    
    logger.info(
        f"XGBoost trained. R² (train): {model.score(X_imputed, y_train):.4f}"
    )
    return model, imputer


# ════════════════════════════════════════════════════════
# Train All Models
# ════════════════════════════════════════════════════════

def train_all_models(
    splits: Dict,
    models_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """
    Train all models and return a dictionary of trained model bundles.
    
    Each bundle contains the model, preprocessors, and metadata.
    
    Returns
    -------
    dict
        Keys: model names, Values: dict with 'model', 'imputer', 'scaler' (optional),
        'feature_cols'.
    """
    from config import MODELS_DIR
    
    if models_dir is None:
        models_dir = MODELS_DIR
    
    X_train, y_train = splits["train"]
    feature_cols = splits["feature_cols"]
    
    trained_models = {}
    
    # Model A: Baseline
    logger.info("Training Model A: Historical Baseline...")
    baseline = HistoricalBaseline()
    baseline.fit(X_train, y_train)
    trained_models["Historical Baseline"] = {
        "model": baseline,
        "imputer": None,
        "scaler": None,
        "feature_cols": feature_cols,
    }
    
    # Model B: Linear Regression
    logger.info("Training Model B: Linear Regression...")
    lr_model, lr_imputer, lr_scaler = train_linear_regression(X_train, y_train)
    trained_models["Linear Regression"] = {
        "model": lr_model,
        "imputer": lr_imputer,
        "scaler": lr_scaler,
        "feature_cols": feature_cols,
    }
    
    # Model C: Random Forest
    logger.info("Training Model C: Random Forest...")
    rf_model, rf_imputer = train_random_forest(X_train, y_train)
    trained_models["Random Forest"] = {
        "model": rf_model,
        "imputer": rf_imputer,
        "scaler": None,
        "feature_cols": feature_cols,
    }
    
    # Model D: XGBoost (optional)
    logger.info("Training Model D: XGBoost...")
    xgb_result = train_xgboost(X_train, y_train)
    if xgb_result is not None:
        xgb_model, xgb_imputer = xgb_result
        trained_models["XGBoost"] = {
            "model": xgb_model,
            "imputer": xgb_imputer,
            "scaler": None,
            "feature_cols": feature_cols,
        }
    
    # Save models
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, bundle in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        filepath = models_dir / f"{safe_name}_bundle.joblib"
        joblib.dump(bundle, filepath)
        logger.info(f"Saved {name} to {filepath}")
    
    return trained_models


def predict_with_model(
    model_bundle: Dict,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions using a trained model bundle.
    
    Handles imputation and scaling as needed.
    """
    model = model_bundle["model"]
    imputer = model_bundle.get("imputer")
    scaler = model_bundle.get("scaler")
    
    X_processed = X.copy()
    
    if isinstance(model, HistoricalBaseline):
        return model.predict(X_processed)
    
    if imputer is not None:
        X_processed = imputer.transform(X_processed)
    
    if scaler is not None:
        X_processed = scaler.transform(X_processed)
    
    preds = model.predict(X_processed)
    
    # Ensure non-negative predictions (forest loss can't be negative)
    preds = np.maximum(preds, 0)
    
    return preds


def load_trained_models(
    models_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Load all saved model bundles from disk."""
    from config import MODELS_DIR
    
    if models_dir is None:
        models_dir = MODELS_DIR
    
    trained_models = {}
    
    for filepath in models_dir.glob("*_bundle.joblib"):
        name = filepath.stem.replace("_bundle", "").replace("_", " ").title()
        bundle = joblib.load(filepath)
        trained_models[name] = bundle
        logger.info(f"Loaded model: {name}")
    
    return trained_models
