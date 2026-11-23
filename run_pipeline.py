"""
run_pipeline.py — Full End-to-End Pipeline
===========================================
Runs all phases: data loading → preprocessing → EDA → feature engineering →
model training → evaluation → forecasting → risk assessment → save artifacts.

Usage:
    python run_pipeline.py
"""
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import config
from config import logger

from src.data_loader import load_all_datasets, validate_dataset
from src.preprocessing import (
    preprocess_subnational_tree_loss,
    preprocess_carbon_data,
    merge_carbon_with_loss,
    save_processed_data,
    generate_data_quality_report,
)
from src.feature_engineering import build_feature_set, get_feature_columns, FEATURE_DOCUMENTATION
from src.train import train_all_models, chronological_split, predict_with_model
from src.evaluate import evaluate_all_models, evaluate_by_region, build_prediction_table
from src.forecasting import forecast_region, compute_forecast_uncertainty, apply_uncertainty_intervals
from src.risk_assessment import full_risk_assessment, generate_risk_report
from src.eda import summary_statistics, compute_regional_trends


def main():
    print("=" * 70)
    print("RAINFOREST DEFORESTATION ML PIPELINE")
    print("=" * 70)

    # ─── Phase 1: Load & Validate ────────────────────────────
    print("\n📦 Phase 1: Loading datasets...")
    datasets = load_all_datasets()

    for key, df in datasets.items():
        if df is not None:
            report = validate_dataset(df, key)
            print(f"  ✅ {key}: {report['shape']}")
        else:
            print(f"  ❌ {key}: NOT LOADED")

    # ─── Phase 2: Preprocessing ──────────────────────────────
    print("\n🔧 Phase 2: Preprocessing...")
    processed = preprocess_subnational_tree_loss(
        datasets["subnational_loss"],
        threshold=config.TREE_COVER_THRESHOLD,
        country=config.DEFAULT_COUNTRY,
    )
    print(f"  Processed shape: {processed.shape}")
    print(f"  Regions: {processed['region'].nunique()}")
    print(f"  Years: {processed['year'].min()}–{processed['year'].max()}")

    # Carbon data
    carbon_long = preprocess_carbon_data(
        datasets["subnational_carbon"],
        threshold=config.TREE_COVER_THRESHOLD,
        country=config.DEFAULT_COUNTRY,
    )
    processed = merge_carbon_with_loss(processed, carbon_long)
    print(f"  Carbon data merged: {processed['carbon_emissions_Mg_CO2e'].notna().sum()} rows")

    # Save processed data
    save_processed_data(processed, "subnational_loss_long.csv")

    # Data quality report
    quality_report = generate_data_quality_report(processed, "processed_subnational")
    with open(config.REPORTS_DIR / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2, default=str)
    print("  ✅ Processed data and quality report saved.")

    # ─── Phase 3: EDA ────────────────────────────────────────
    print("\n📊 Phase 3: Exploratory Data Analysis...")
    stats = summary_statistics(processed)
    print(f"  Total forest loss: {stats['total_loss_ha']:,.0f} ha")
    print(f"  Peak year: {stats['peak_year']} ({stats['peak_loss']:,.0f} ha)")
    print(f"  Top 5 regions: {', '.join(stats['top_5_regions'])}")
    print(f"  Regions with increasing trends: {len(stats['increasing_regions'])}")
    print(f"  Regions with decreasing trends: {len(stats['decreasing_regions'])}")

    trends = compute_regional_trends(processed)
    trends.to_csv(config.REPORTS_DIR / "regional_trends.csv", index=False)

    # Save stats summary
    stats_save = {k: v for k, v in stats.items()
                  if not isinstance(v, pd.DataFrame)}
    with open(config.REPORTS_DIR / "eda_summary.json", "w") as f:
        json.dump(stats_save, f, indent=2, default=str)
    print("  ✅ EDA reports saved.")

    # ─── Phase 4: Feature Engineering ────────────────────────
    print("\n⚙️ Phase 4: Feature Engineering...")
    featured = build_feature_set(processed)
    feature_cols = get_feature_columns(featured)
    print(f"  Features created: {len(feature_cols)}")
    print(f"  Feature names: {feature_cols}")

    save_processed_data(featured, "featured_dataset.csv")

    # Save feature documentation
    with open(config.REPORTS_DIR / "feature_documentation.txt", "w") as f:
        f.write(FEATURE_DOCUMENTATION)
    print("  ✅ Feature set saved.")

    # ─── Phase 5: Chronological Split ────────────────────────
    print("\n✂️ Phase 5: Chronological train/val/test split...")
    splits = chronological_split(
        featured,
        train_end_year=config.TRAIN_END_YEAR,
        val_end_year=config.VAL_END_YEAR,
        feature_cols=feature_cols,
    )
    print(f"  Train: {len(splits['train'][0])} rows (≤{config.TRAIN_END_YEAR})")
    print(f"  Val:   {len(splits['val'][0])} rows ({config.TRAIN_END_YEAR+1}–{config.VAL_END_YEAR})")
    print(f"  Test:  {len(splits['test'][0])} rows (>{config.VAL_END_YEAR})")

    # ─── Phase 6: Model Training ─────────────────────────────
    print("\n🤖 Phase 6: Training models...")
    trained_models = train_all_models(splits)
    print(f"  Models trained: {list(trained_models.keys())}")

    # ─── Phase 7: Evaluation ─────────────────────────────────
    print("\n📈 Phase 7: Model evaluation...")

    # Validation set
    val_comparison, val_preds = evaluate_all_models(trained_models, splits, "val")
    print("\n  === Validation Set Results ===")
    print(val_comparison.to_string(index=False))

    # Test set
    test_comparison, test_preds = evaluate_all_models(trained_models, splits, "test")
    print("\n  === Test Set Results ===")
    print(test_comparison.to_string(index=False))

    # Save comparison tables
    val_comparison.to_csv(config.REPORTS_DIR / "model_comparison_val.csv", index=False)
    test_comparison.to_csv(config.REPORTS_DIR / "model_comparison_test.csv", index=False)

    # Region-level evaluation
    region_metrics = evaluate_by_region(trained_models, splits, "test")
    region_metrics.to_csv(config.REPORTS_DIR / "region_level_metrics.csv", index=False)

    # Prediction table
    pred_table = build_prediction_table(trained_models, splits, "test")
    pred_table.to_csv(config.REPORTS_DIR / "predictions_test.csv", index=False)
    print("  ✅ Evaluation results saved.")

    # ─── Phase 8: Forecasting ────────────────────────────────
    print("\n🔮 Phase 8: Forecasting...")

    # Use best model (by MAE on test)
    best_model_name = test_comparison.iloc[0]["Model"]
    print(f"  Best model: {best_model_name}")

    # Compute uncertainty
    uncertainty = compute_forecast_uncertainty(
        trained_models, splits, best_model_name, "test"
    )
    print(f"  Forecast uncertainty (residual std): {uncertainty['residual_std']:,.0f} ha")

    # Forecast for all regions
    all_forecasts = []
    for region in processed["region"].unique():
        fc = forecast_region(
            featured, region,
            trained_models[best_model_name],
            horizon=config.DEFAULT_FORECAST_HORIZON,
            last_observed_year=config.YEAR_END,
        )
        fc = apply_uncertainty_intervals(fc, uncertainty)
        all_forecasts.append(fc)

    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    forecasts_df.to_csv(config.REPORTS_DIR / "forecasts_all_regions.csv", index=False)

    future_only = forecasts_df[forecasts_df["type"] == "forecast"]
    print(f"  Forecasted {future_only['region'].nunique()} regions × {config.DEFAULT_FORECAST_HORIZON} years")
    print(f"  Total future rows: {len(future_only)}")
    print("  ✅ Forecasts saved.")

    # ─── Phase 9: Risk Assessment ────────────────────────────
    print("\n⚠️ Phase 9: Risk Assessment...")

    predicted_future = {}
    for region in processed["region"].unique():
        region_fc = future_only[future_only["region"] == region]
        if not region_fc.empty:
            predicted_future[region] = region_fc["predicted_loss_ha"].mean()

    risk_df = full_risk_assessment(
        processed,
        predicted_future=predicted_future,
        last_n_years=3,
    )
    risk_report = generate_risk_report(risk_df)
    risk_report.to_csv(config.REPORTS_DIR / "risk_assessment_report.csv", index=False)

    print("\n  Risk Assessment Results:")
    print(f"  {risk_df['risk_category'].value_counts().to_dict()}")
    print("\n  Top 5 highest-risk regions:")
    for _, row in risk_report.head(5).iterrows():
        print(f"    {row['region']}: {row['risk_category']} (score={row['risk_score']:.3f})")

    print("  ✅ Risk assessment saved.")

    # ─── Summary ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE — All artifacts saved to:")
    print(f"   Data:    {config.PROCESSED_DATA_DIR}")
    print(f"   Models:  {config.MODELS_DIR}")
    print(f"   Reports: {config.REPORTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
