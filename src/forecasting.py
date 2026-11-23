"""
Forecasting Module
==================
Generates future deforestation forecasts using trained models.

Key design decisions:
  - Forecasts begin after the last observed year (default: 2022).
  - Future features are generated recursively: each forecast year uses
    predictions from prior years as inputs.
  - Forecasts are labeled as ESTIMATES, not certain outcomes.
  - Uncertainty intervals are provided via a simple residual-based method.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("deforestation_ml.forecasting")


def generate_future_features(
    historical_df: pd.DataFrame,
    region: str,
    forecast_years: List[int],
    model_bundle: Dict,
    target_col: str = "tc_loss_ha",
) -> pd.DataFrame:
    """
    Recursively generate features and predictions for future years.
    
    For each future year:
      1. Build feature vector using available historical + previously predicted values.
      2. Predict target using the trained model.
      3. Append prediction to history for next iteration.
    
    Parameters
    ----------
    historical_df : pd.DataFrame
        Historical data for this region with all features.
    region : str
        Region name.
    forecast_years : list of int
        Years to forecast (e.g., [2023, 2024, 2025, 2026, 2027]).
    model_bundle : dict
        Trained model bundle.
    target_col : str
    
    Returns
    -------
    pd.DataFrame
        Forecast rows with year, predicted loss, and feature values.
    """
    from src.train import predict_with_model
    from src.feature_engineering import get_feature_columns
    
    region_data = historical_df[historical_df["region"] == region].copy()
    region_data = region_data.sort_values("year").reset_index(drop=True)
    
    if region_data.empty:
        logger.warning(f"No historical data for region '{region}'")
        return pd.DataFrame()
    
    feature_cols = model_bundle.get("feature_cols", get_feature_columns(region_data))
    forecasts = []
    
    # Working copy of historical values for recursive feature computation
    loss_history = region_data[target_col].tolist()
    rate_history = (
        region_data["loss_rate_pct"].tolist()
        if "loss_rate_pct" in region_data.columns
        else [0.0] * len(loss_history)
    )
    cumulative = (
        region_data["cumulative_loss_pct"].iloc[-1]
        if "cumulative_loss_pct" in region_data.columns
        else 0.0
    )
    
    extent = (
        region_data["extent_2000_ha"].iloc[0]
        if "extent_2000_ha" in region_data.columns
        else 0.0
    )
    region_encoded = (
        region_data["region_encoded"].iloc[0]
        if "region_encoded" in region_data.columns
        else 0
    )
    
    for year in forecast_years:
        features = {}
        
        # Lag features
        for lag in [1, 2, 3]:
            col = f"{target_col}_lag{lag}"
            if col in feature_cols:
                idx = -lag
                features[col] = (
                    loss_history[idx] if abs(idx) <= len(loss_history) else 0.0
                )
        
        # Rolling features
        for w in [3, 5]:
            mean_col = f"{target_col}_roll_mean_{w}yr"
            std_col = f"{target_col}_roll_std_{w}yr"
            
            window_data = loss_history[-w:] if len(loss_history) >= w else loss_history
            
            if mean_col in feature_cols:
                features[mean_col] = np.mean(window_data) if window_data else 0.0
            if std_col in feature_cols:
                features[std_col] = np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0
        
        # Trend features
        if "loss_change_1yr_lagged" in feature_cols:
            if len(loss_history) >= 2:
                features["loss_change_1yr_lagged"] = loss_history[-1] - loss_history[-2]
            else:
                features["loss_change_1yr_lagged"] = 0.0
        
        if "loss_change_pct_1yr_lagged" in feature_cols:
            if len(loss_history) >= 2 and loss_history[-2] > 0:
                features["loss_change_pct_1yr_lagged"] = (
                    (loss_history[-1] - loss_history[-2]) / loss_history[-2] * 100
                )
            else:
                features["loss_change_pct_1yr_lagged"] = 0.0
        
        if "loss_acceleration" in feature_cols:
            if len(loss_history) >= 3:
                change1 = loss_history[-1] - loss_history[-2]
                change2 = loss_history[-2] - loss_history[-3]
                features["loss_acceleration"] = change1 - change2
            else:
                features["loss_acceleration"] = 0.0
        
        if "loss_trend_3yr" in feature_cols:
            if len(loss_history) >= 3:
                recent = loss_history[-3:]
                x = np.arange(3)
                features["loss_trend_3yr"] = np.polyfit(x, recent, 1)[0]
            else:
                features["loss_trend_3yr"] = 0.0
        
        # Rate features
        if "loss_rate_pct_lag1" in feature_cols:
            features["loss_rate_pct_lag1"] = rate_history[-1] if rate_history else 0.0
        
        if "cumulative_loss_pct_lag1" in feature_cols:
            features["cumulative_loss_pct_lag1"] = cumulative
        
        # Variability
        if "loss_cv_5yr" in feature_cols:
            window_data = loss_history[-5:] if len(loss_history) >= 5 else loss_history
            mean_val = np.mean(window_data)
            features["loss_cv_5yr"] = (
                np.std(window_data, ddof=1) / mean_val if mean_val > 0 else 0.0
            )
        
        # Region encoding
        if "region_encoded" in feature_cols:
            features["region_encoded"] = region_encoded
        
        # Build feature DataFrame for prediction
        X_future = pd.DataFrame([features])
        
        # Ensure all feature columns are present
        for col in feature_cols:
            if col not in X_future.columns:
                X_future[col] = 0.0
        X_future = X_future[feature_cols]
        
        # Predict
        prediction = predict_with_model(model_bundle, X_future)[0]
        prediction = max(0, prediction)  # Can't have negative loss
        
        # Update history for next iteration
        loss_history.append(prediction)
        
        if extent > 0:
            rate = (prediction / extent) * 100.0
        else:
            rate = 0.0
        rate_history.append(rate)
        
        cumulative += rate
        
        forecasts.append({
            "region": region,
            "year": year,
            "predicted_loss_ha": round(prediction, 0),
            "predicted_rate_pct": round(rate, 4),
            "cumulative_loss_pct": round(cumulative, 4),
            "type": "forecast",
        })
    
    return pd.DataFrame(forecasts)


def forecast_region(
    historical_df: pd.DataFrame,
    region: str,
    model_bundle: Dict,
    horizon: int = 5,
    last_observed_year: int = 2022,
) -> pd.DataFrame:
    """
    Generate forecasts for a single region.
    
    Parameters
    ----------
    historical_df : pd.DataFrame
    region : str
    model_bundle : dict
    horizon : int
        Number of years to forecast.
    last_observed_year : int
    
    Returns
    -------
    pd.DataFrame
        Combined historical observations and future forecasts.
    """
    forecast_years = list(range(last_observed_year + 1, last_observed_year + horizon + 1))
    
    future_df = generate_future_features(
        historical_df, region, forecast_years, model_bundle
    )
    
    # Build historical reference
    region_hist = historical_df[historical_df["region"] == region].copy()
    hist_summary = region_hist[["region", "year", "tc_loss_ha", "loss_rate_pct"]].copy()
    hist_summary = hist_summary.rename(columns={
        "tc_loss_ha": "predicted_loss_ha",
        "loss_rate_pct": "predicted_rate_pct",
    })
    hist_summary["type"] = "observed"
    if "cumulative_loss_pct" in region_hist.columns:
        hist_summary["cumulative_loss_pct"] = region_hist["cumulative_loss_pct"].values
    
    combined = pd.concat([hist_summary, future_df], ignore_index=True)
    combined = combined.sort_values("year").reset_index(drop=True)
    
    return combined


def compute_forecast_uncertainty(
    trained_models: Dict,
    splits: Dict,
    model_name: str,
    split_name: str = "test",
) -> Dict:
    """
    Estimate forecast uncertainty from historical residuals.
    
    Method: Use the standard deviation of test-set residuals as a measure
    of forecast uncertainty. This provides a rough confidence interval
    but does NOT account for increasing uncertainty with forecast horizon.
    
    This is a simple, defensible method. For production use, consider
    bootstrapping, conformal prediction, or quantile regression.
    
    Returns
    -------
    dict with 'residual_std', 'mean_residual', and 'ci_multiplier' (1.96 for 95%).
    """
    from src.train import predict_with_model
    
    X = splits[split_name][0]
    y = splits[split_name][1]
    
    bundle = trained_models[model_name]
    y_pred = predict_with_model(bundle, X)
    
    residuals = y.values - y_pred
    
    uncertainty = {
        "residual_std": float(np.std(residuals)),
        "mean_residual": float(np.mean(residuals)),
        "ci_multiplier": 1.96,  # 95% confidence interval
        "ci_lower_factor": float(np.mean(residuals) - 1.96 * np.std(residuals)),
        "ci_upper_factor": float(np.mean(residuals) + 1.96 * np.std(residuals)),
        "method": (
            "Residual-based 95% interval: prediction ± 1.96 × σ(test residuals). "
            "This assumes stationary error variance and does not account for "
            "increasing uncertainty with forecast horizon. Treat as approximate."
        ),
    }
    
    return uncertainty


def apply_uncertainty_intervals(
    forecast_df: pd.DataFrame,
    uncertainty: Dict,
    loss_col: str = "predicted_loss_ha",
) -> pd.DataFrame:
    """
    Add uncertainty intervals to forecast DataFrame.
    
    Only applies to 'forecast' type rows.
    """
    df = forecast_df.copy()
    
    std = uncertainty["residual_std"]
    multiplier = uncertainty["ci_multiplier"]
    
    is_forecast = df["type"] == "forecast"
    
    df.loc[is_forecast, "lower_bound"] = np.maximum(
        0, df.loc[is_forecast, loss_col] - multiplier * std
    )
    df.loc[is_forecast, "upper_bound"] = (
        df.loc[is_forecast, loss_col] + multiplier * std
    )
    
    return df
