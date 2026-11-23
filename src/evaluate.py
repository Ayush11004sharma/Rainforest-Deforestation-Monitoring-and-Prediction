"""
Model Evaluation Module
=======================
Evaluates trained models using chronological splits.

Metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² (Coefficient of Determination)

All metrics are calculated on real model predictions — never fabricated.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

logger = logging.getLogger("deforestation_ml.evaluate")


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a single model.
    
    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    model_name : str
    
    Returns
    -------
    dict with MAE, RMSE, R² values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    # Remove NaN pairs
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        logger.warning(f"{model_name}: No valid predictions to evaluate.")
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "n_samples": 0}
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "n_samples": int(len(y_true)),
    }
    
    logger.info(
        f"{model_name} — MAE: {metrics['MAE']:,.0f} ha, "
        f"RMSE: {metrics['RMSE']:,.0f} ha, "
        f"R²: {metrics['R2']:.4f} "
        f"(n={metrics['n_samples']})"
    )
    
    return metrics


def evaluate_all_models(
    trained_models: Dict,
    splits: Dict,
    split_name: str = "test",
) -> pd.DataFrame:
    """
    Evaluate all trained models on a specified split.
    
    Parameters
    ----------
    trained_models : dict
        Model bundles from train_all_models().
    splits : dict
        Data splits from chronological_split().
    split_name : str
        Which split to evaluate ('val' or 'test').
    
    Returns
    -------
    pd.DataFrame
        Comparison table with one row per model.
    """
    from src.train import predict_with_model
    
    X = splits[split_name][0]
    y = splits[split_name][1]
    
    results = []
    predictions = {}
    
    for name, bundle in trained_models.items():
        y_pred = predict_with_model(bundle, X)
        metrics = evaluate_model(y.values, y_pred, name)
        metrics["Model"] = name
        results.append(metrics)
        predictions[name] = y_pred
    
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df[["Model", "MAE", "RMSE", "R2", "n_samples"]]
    comparison_df = comparison_df.sort_values("MAE").reset_index(drop=True)
    
    return comparison_df, predictions


def evaluate_by_region(
    trained_models: Dict,
    splits: Dict,
    split_name: str = "test",
) -> pd.DataFrame:
    """
    Evaluate models per region for the specified split.
    
    Returns a DataFrame with region-level metrics for each model.
    """
    from src.train import predict_with_model
    
    X = splits[split_name][0]
    y = splits[split_name][1]
    meta = splits[f"meta_{split_name}"]
    
    region_results = []
    
    for model_name, bundle in trained_models.items():
        y_pred = predict_with_model(bundle, X)
        
        # Create evaluation dataframe
        eval_df = meta.copy()
        eval_df["y_pred"] = y_pred
        eval_df["y_true"] = y.values
        
        for region in eval_df["region"].unique():
            region_data = eval_df[eval_df["region"] == region]
            if len(region_data) == 0:
                continue
            
            metrics = evaluate_model(
                region_data["y_true"].values,
                region_data["y_pred"].values,
                f"{model_name} — {region}",
            )
            metrics["Model"] = model_name
            metrics["Region"] = region
            region_results.append(metrics)
    
    return pd.DataFrame(region_results)


def build_prediction_table(
    trained_models: Dict,
    splits: Dict,
    split_name: str = "test",
    best_model_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a table of actual vs predicted values for visualization.
    
    Returns
    -------
    pd.DataFrame
        Columns: region, year, actual, predicted_{model_name}, ...
    """
    from src.train import predict_with_model
    
    X = splits[split_name][0]
    y = splits[split_name][1]
    meta = splits[f"meta_{split_name}"]
    
    result = meta[["region", "year"]].copy()
    result["actual"] = y.values
    
    for name, bundle in trained_models.items():
        y_pred = predict_with_model(bundle, X)
        safe_name = name.lower().replace(" ", "_")
        result[f"pred_{safe_name}"] = y_pred
    
    return result.sort_values(["region", "year"]).reset_index(drop=True)
