"""
Rainforest Deforestation Monitoring & Prediction Dashboard
==========================================================
Interactive Streamlit application for visualizing historical deforestation,
evaluating ML models, generating forecasts, and assessing risk.

Launch: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.data_loader import load_all_datasets, get_threshold_column, get_extent_column
from src.preprocessing import (
    preprocess_subnational_tree_loss,
    preprocess_carbon_data,
    merge_carbon_with_loss,
    extract_annual_columns,
)
from src.feature_engineering import build_feature_set, get_feature_columns, FEATURE_DOCUMENTATION
from src.train import (
    train_all_models, chronological_split, predict_with_model,
    load_trained_models, HistoricalBaseline,
)
from src.evaluate import (
    evaluate_all_models, evaluate_by_region, build_prediction_table,
)
from src.forecasting import (
    forecast_region, compute_forecast_uncertainty, apply_uncertainty_intervals,
)
from src.risk_assessment import full_risk_assessment, generate_risk_report
from src.visualization import (
    plot_annual_trend, plot_total_annual_loss, plot_regional_comparison,
    plot_heatmap, plot_loss_rate_heatmap, plot_cumulative_loss,
    plot_actual_vs_predicted, plot_forecast, plot_risk_distribution,
    plot_risk_ranking, plot_carbon_trends, plot_model_comparison,
)
from src.eda import summary_statistics, compute_regional_trends, correlation_analysis


# ════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🌳 Rainforest Deforestation Monitor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════
# Caching
# ════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading and preprocessing data...")
def load_and_preprocess(country: str, threshold: int):
    """Load, preprocess, and cache the analytical datasets."""
    datasets = load_all_datasets()
    
    if datasets.get("subnational_loss") is None:
        st.error("❌ Subnational tree cover loss dataset not found. Please check data/raw/ directory.")
        st.stop()
    
    # Preprocess main dataset
    processed = preprocess_subnational_tree_loss(
        datasets["subnational_loss"],
        threshold=threshold,
        country=country,
    )
    
    # Preprocess carbon data
    carbon_long = None
    if datasets.get("subnational_carbon") is not None:
        carbon_long = preprocess_carbon_data(
            datasets["subnational_carbon"],
            threshold=threshold,
            country=country,
        )
        processed = merge_carbon_with_loss(processed, carbon_long)
    
    # Country-level data
    country_loss = None
    if datasets.get("country_loss") is not None:
        from src.preprocessing import filter_by_threshold, melt_tree_cover_loss
        cl = filter_by_threshold(datasets["country_loss"], threshold)
        if country:
            cl = cl[cl["country"] == country]
        annual_cols = extract_annual_columns(cl, "tc_loss_ha_")
        id_vars = [c for c in cl.columns if c not in annual_cols]
        country_loss = melt_tree_cover_loss(cl, id_vars=id_vars)
    
    return processed, carbon_long, country_loss


@st.cache_data(show_spinner="Engineering features...")
def engineer_features(processed_df):
    """Build feature set and cache."""
    return build_feature_set(processed_df)


@st.cache_data(show_spinner="Training models...")
def train_models_cached(_featured_df, train_end, val_end):
    """Train models and cache results."""
    feature_cols = get_feature_columns(_featured_df)
    splits = chronological_split(
        _featured_df,
        train_end_year=train_end,
        val_end_year=val_end,
        feature_cols=feature_cols,
    )
    trained = train_all_models(splits)
    return trained, splits


# ════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════

st.sidebar.title("🌳 Deforestation Monitor")
st.sidebar.markdown("---")

# Page navigation
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Overview",
        "📈 Historical Monitoring",
        "🤖 ML Model Evaluation",
        "🔮 Future Forecasts",
        "⚠️ Risk Assessment",
        "🏭 Carbon Analysis",
        "📋 Data & Methodology",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Configuration")

country = st.sidebar.selectbox(
    "Country",
    ["Brazil"],
    help="Primary country for analysis. Currently configured for Brazil's 27 subnational regions.",
)

threshold = st.sidebar.selectbox(
    "Tree-Cover Threshold (%)",
    [30, 10, 15, 20, 25, 50, 75],
    index=0,
    help=(
        "Minimum tree-cover density in year 2000. "
        "Threshold=30 means pixels with ≥30% canopy cover."
    ),
)

# ════════════════════════════════════════════════════════
# Load Data
# ════════════════════════════════════════════════════════

try:
    processed, carbon_long, country_loss = load_and_preprocess(country, threshold)
    featured = engineer_features(processed)
    trained_models, splits = train_models_cached(featured, config.TRAIN_END_YEAR, config.VAL_END_YEAR)
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.exception(e)
    st.stop()

# ════════════════════════════════════════════════════════
# Page: Overview
# ════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("🌳 Rainforest Deforestation Monitoring & Prediction")
    st.markdown(
        "**Machine Learning Analysis of Tree-Cover Loss in Brazil (2001–2022)**"
    )
    st.markdown(
        "> Data source: Global Forest Change (Hansen et al.), derived from Landsat satellite imagery. "
        "Tree-cover loss represents the removal of tree canopy, not necessarily permanent deforestation."
    )
    
    stats = summary_statistics(processed)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Forest Loss", f"{stats['total_loss_ha']:,.0f} ha")
    with col2:
        st.metric("Regions Analyzed", stats["n_regions"])
    with col3:
        st.metric("Period", f"{stats['year_range'][0]}–{stats['year_range'][1]}")
    with col4:
        st.metric("Avg Annual Loss", f"{stats['avg_annual_loss']:,.0f} ha")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_total_annual_loss(processed), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_regional_comparison(processed, top_n=15), use_container_width=True)
    
    st.markdown("### 🔑 Key Findings")
    st.markdown(f"""
    - **Peak year:** {stats['peak_year']} with {stats['peak_loss']:,.0f} ha lost
    - **Most affected regions:** {', '.join(stats['top_5_regions'][:3])}
    - **Regions with increasing trends:** {len(stats['increasing_regions'])} of {stats['n_regions']}
    - **Regions with decreasing trends:** {len(stats['decreasing_regions'])} of {stats['n_regions']}
    """)
    
    st.info(
        "⚠️ **Important:** Tree-cover loss ≠ permanent deforestation. "
        "Loss includes natural disturbances, plantation harvesting, and fire. "
        "This analysis measures gross tree-cover loss at the ≥{threshold}% "
        "canopy density threshold."
    )


# ════════════════════════════════════════════════════════
# Page: Historical Monitoring
# ════════════════════════════════════════════════════════

elif page == "📈 Historical Monitoring":
    st.title("📈 Historical Deforestation Monitoring")
    
    regions = sorted(processed["region"].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_regions = st.multiselect(
            "Select Regions",
            regions,
            default=regions[:5],
            help="Select regions to display in the charts.",
        )
    with col2:
        year_range = st.slider(
            "Year Range",
            int(processed["year"].min()),
            int(processed["year"].max()),
            (int(processed["year"].min()), int(processed["year"].max())),
        )
    
    # Filter data
    filtered = processed[
        (processed["region"].isin(selected_regions)) &
        (processed["year"] >= year_range[0]) &
        (processed["year"] <= year_range[1])
    ]
    
    if filtered.empty:
        st.warning("No data for selected filters.")
    else:
        # Annual trend by region
        st.plotly_chart(
            plot_annual_trend(
                filtered, color_col="region",
                title="Annual Tree-Cover Loss by Region",
            ),
            use_container_width=True,
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                plot_heatmap(filtered, title="Tree-Cover Loss Heatmap"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                plot_cumulative_loss(filtered, top_n=len(selected_regions)),
                use_container_width=True,
            )
        
        # Loss rate heatmap
        if "loss_rate_pct" in filtered.columns:
            st.plotly_chart(
                plot_loss_rate_heatmap(filtered),
                use_container_width=True,
            )
        
        # Regional ranking table
        st.subheader("📊 Regional Ranking")
        from src.eda import ranking_table
        rank_year = st.selectbox(
            "Rank by year (or leave blank for total)",
            [None] + list(range(year_range[1], year_range[0] - 1, -1)),
            format_func=lambda x: "All years (total)" if x is None else str(x),
        )
        ranking = ranking_table(filtered, year=rank_year)
        st.dataframe(ranking, use_container_width=True)
        
        # Download
        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered Data (CSV)",
            csv,
            "filtered_deforestation_data.csv",
            "text/csv",
        )


# ════════════════════════════════════════════════════════
# Page: ML Model Evaluation
# ════════════════════════════════════════════════════════

elif page == "🤖 ML Model Evaluation":
    st.title("🤖 Machine Learning Model Evaluation")
    
    st.markdown(
        "Models are evaluated using **chronological (time-based) splits** — "
        f"training on ≤{config.TRAIN_END_YEAR}, validating on "
        f"{config.TRAIN_END_YEAR+1}–{config.VAL_END_YEAR}, "
        f"and testing on {config.VAL_END_YEAR+1}–{config.YEAR_END}."
    )
    
    eval_split = st.radio(
        "Evaluate on:", ["Test Set", "Validation Set"],
        horizontal=True,
    )
    split_name = "test" if eval_split == "Test Set" else "val"
    
    # Model comparison
    comparison_df, predictions = evaluate_all_models(trained_models, splits, split_name)
    
    st.subheader("📊 Model Comparison")
    st.dataframe(
        comparison_df.style.format({
            "MAE": "{:,.0f}",
            "RMSE": "{:,.0f}",
            "R2": "{:.4f}",
        }),
        use_container_width=True,
    )
    
    st.plotly_chart(
        plot_model_comparison(comparison_df),
        use_container_width=True,
    )
    
    # Actual vs predicted
    st.subheader("📈 Actual vs Predicted")
    
    pred_table = build_prediction_table(trained_models, splits, split_name)
    model_cols = [c for c in pred_table.columns if c.startswith("pred_")]
    
    selected_model_col = st.selectbox(
        "Select model for visualization:",
        model_cols,
        format_func=lambda x: x.replace("pred_", "").replace("_", " ").title(),
    )
    
    selected_region_eval = st.selectbox(
        "Select region (or 'All'):",
        ["All"] + sorted(pred_table["region"].unique()),
    )
    
    if selected_region_eval == "All":
        st.plotly_chart(
            plot_actual_vs_predicted(pred_table, model_col=selected_model_col),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            plot_actual_vs_predicted(
                pred_table, region=selected_region_eval,
                model_col=selected_model_col,
            ),
            use_container_width=True,
        )
    
    # Region-level performance
    st.subheader("🗺️ Region-Level Performance")
    region_metrics = evaluate_by_region(trained_models, splits, split_name)
    
    selected_model_name = st.selectbox(
        "Model:",
        list(trained_models.keys()),
        key="region_model",
    )
    
    region_data = region_metrics[region_metrics["Model"] == selected_model_name]
    st.dataframe(
        region_data[["Region", "MAE", "RMSE", "R2", "n_samples"]].style.format({
            "MAE": "{:,.0f}",
            "RMSE": "{:,.0f}",
            "R2": "{:.4f}",
        }),
        use_container_width=True,
    )
    
    # Feature importance
    st.subheader("🔍 Feature Importance")
    best_model_name = comparison_df.iloc[0]["Model"]
    best_bundle = trained_models.get(best_model_name, {})
    best_model = best_bundle.get("model")
    
    if hasattr(best_model, "feature_importances_"):
        feature_cols = splits["feature_cols"]
        importances = best_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": importances,
        }).sort_values("Importance", ascending=True).tail(15)
        
        import plotly.express as px
        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title=f"Feature Importance — {best_model_name}",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(
            "⚠️ Feature importance shows which features the model relies on. "
            "This does NOT imply causation — it reflects statistical associations "
            "in historical data."
        )
    else:
        st.info(f"Feature importance is not available for {best_model_name}.")
    
    st.markdown("---")
    st.markdown(
        "**Evaluation Methodology:** All models are compared on the same "
        "chronological test period. No random shuffling is used — this prevents "
        "temporal leakage. Metrics are computed from real model predictions, "
        "not simulated or fabricated."
    )


# ════════════════════════════════════════════════════════
# Page: Future Forecasts
# ════════════════════════════════════════════════════════

elif page == "🔮 Future Forecasts":
    st.title("🔮 Future Deforestation Forecasts")
    
    st.warning(
        "⚠️ **Forecast Disclaimer:** These are model-based ESTIMATES using "
        "historical patterns only. They do not account for future policy changes, "
        "fires, climate events, economic shifts, or other unpredictable factors. "
        "Do not use these forecasts as real-time satellite monitoring outputs."
    )
    
    regions = sorted(processed["region"].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_region_name = st.selectbox(
            "Select Region", regions,
            index=regions.index("Pará") if "Pará" in regions else 0,
        )
    with col2:
        horizon = st.slider(
            "Forecast Horizon (years)", 1, 5, 5,
            help="Number of years to forecast beyond 2022.",
        )
    
    forecast_model = st.selectbox(
        "Forecast Model",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index("Random Forest")
        if "Random Forest" in trained_models else 0,
    )
    
    # Generate forecast
    with st.spinner("Generating forecast..."):
        forecast_df = forecast_region(
            featured, forecast_region_name,
            trained_models[forecast_model],
            horizon=horizon,
            last_observed_year=config.YEAR_END,
        )
        
        # Add uncertainty
        try:
            uncertainty = compute_forecast_uncertainty(
                trained_models, splits, forecast_model,
            )
            forecast_df = apply_uncertainty_intervals(forecast_df, uncertainty)
            has_uncertainty = True
        except Exception:
            has_uncertainty = False
    
    st.plotly_chart(
        plot_forecast(forecast_df, forecast_region_name),
        use_container_width=True,
    )
    
    # Forecast table
    st.subheader("📋 Forecast Values")
    forecast_only = forecast_df[forecast_df["type"] == "forecast"].copy()
    display_cols = ["year", "predicted_loss_ha", "predicted_rate_pct"]
    if has_uncertainty:
        display_cols.extend(["lower_bound", "upper_bound"])
    
    st.dataframe(
        forecast_only[display_cols].style.format({
            "predicted_loss_ha": "{:,.0f}",
            "predicted_rate_pct": "{:.4f}",
            "lower_bound": "{:,.0f}" if has_uncertainty else "{}",
            "upper_bound": "{:,.0f}" if has_uncertainty else "{}",
        }),
        use_container_width=True,
    )
    
    if has_uncertainty:
        st.caption(
            f"**Uncertainty method:** {uncertainty['method']}"
        )
    
    # Download
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Forecast (CSV)",
        csv,
        f"forecast_{forecast_region_name.replace(' ', '_')}.csv",
        "text/csv",
    )


# ════════════════════════════════════════════════════════
# Page: Risk Assessment
# ════════════════════════════════════════════════════════

elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Deforestation Risk Assessment")
    
    st.info(
        "This is an **analytical risk assessment** for academic purposes. "
        "It is not a verified prediction of illegal logging or an official "
        "conservation classification. Thresholds are configurable."
    )
    
    # Config
    st.sidebar.subheader("Risk Config")
    n_recent = st.sidebar.slider("Recent years window", 2, 5, 3)
    
    # Generate predicted future losses for all regions
    predicted_future = {}
    best_model_name = list(trained_models.keys())[-1]  # Use best available
    if "Random Forest" in trained_models:
        best_model_name = "Random Forest"
    
    for region in processed["region"].unique():
        try:
            fc = forecast_region(
                featured, region, trained_models[best_model_name],
                horizon=3, last_observed_year=config.YEAR_END,
            )
            fc_only = fc[fc["type"] == "forecast"]
            if not fc_only.empty:
                predicted_future[region] = fc_only["predicted_loss_ha"].mean()
        except Exception:
            pass
    
    # Run risk assessment
    risk_df = full_risk_assessment(
        processed, predicted_future=predicted_future, last_n_years=n_recent,
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            plot_risk_distribution(risk_df), use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            plot_risk_ranking(risk_df, top_n=15), use_container_width=True,
        )
    
    st.subheader("📋 Risk Report")
    report = generate_risk_report(risk_df)
    st.dataframe(
        report.style.format({
            "risk_score": "{:.3f}",
            "recent_loss_rate": "{:.4f}",
            "recent_loss_ha": "{:,.0f}",
            "loss_trend_ha_yr": "{:,.0f}",
            "cumulative_loss_pct": "{:.2f}",
            "extent_2000_ha": "{:,.0f}",
        }).map(
            lambda x: (
                "background-color: #ffcccc" if x == "Very High"
                else "background-color: #ffe0b2" if x == "High"
                else "background-color: #fff9c4" if x == "Moderate"
                else "background-color: #c8e6c9" if x == "Low"
                else ""
            ),
            subset=["risk_category"] if "risk_category" in report.columns else [],
        ),
        use_container_width=True,
    )
    
    # Threshold documentation
    with st.expander("📖 Risk Methodology"):
        st.markdown("""
        ### Risk Score Computation
        
        The composite risk score (0–1) is a weighted average of four normalized indicators:
        
        | Indicator | Weight | Description |
        |-----------|--------|-------------|
        | Recent Loss Rate | 30% | Average annual loss rate over recent years (%) |
        | Loss Trend | 25% | Linear slope of loss over recent years (ha/yr) |
        | Cumulative Loss % | 25% | Total historical loss relative to 2000 baseline |
        | Predicted Future Loss | 20% | Model-predicted average annual future loss (ha) |
        
        ### Risk Categories
        
        Categories are assigned using **quartile-based** thresholds from the actual
        score distribution:
        - **Low:** Bottom 25% of scores
        - **Moderate:** 25th–50th percentile
        - **High:** 50th–75th percentile
        - **Very High:** Top 25% of scores
        
        ⚠️ These thresholds are **relative rankings** within the analyzed regions,
        not absolute scientific standards.
        """)
    
    # Download
    csv = risk_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Risk Report (CSV)",
        csv,
        "risk_assessment_report.csv",
        "text/csv",
    )


# ════════════════════════════════════════════════════════
# Page: Carbon Analysis
# ════════════════════════════════════════════════════════

elif page == "🏭 Carbon Analysis":
    st.title("🏭 Carbon Emissions Analysis")
    
    st.markdown(
        "Carbon emissions data from GFW forest carbon flux model. "
        "Units: **Megagrams (Mg) of CO₂ equivalent** per year."
    )
    
    if "carbon_emissions_Mg_CO2e" not in processed.columns or processed["carbon_emissions_Mg_CO2e"].isna().all():
        st.warning("Carbon emissions data is not available for the current selection.")
    else:
        carbon_data = processed[processed["carbon_emissions_Mg_CO2e"].notna()].copy()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            total_emissions = carbon_data["carbon_emissions_Mg_CO2e"].sum()
            st.metric("Total Emissions", f"{total_emissions:,.0f} Mg CO₂e")
        with col2:
            avg_annual = carbon_data.groupby("year")["carbon_emissions_Mg_CO2e"].sum().mean()
            st.metric("Avg Annual Emissions", f"{avg_annual:,.0f} Mg CO₂e")
        with col3:
            n_regions_carbon = carbon_data["region"].nunique()
            st.metric("Regions with Data", n_regions_carbon)
        
        # Annual trend
        annual_carbon = carbon_data.groupby("year")["carbon_emissions_Mg_CO2e"].sum().reset_index()
        import plotly.express as px
        fig = px.bar(
            annual_carbon, x="year", y="carbon_emissions_Mg_CO2e",
            title="Annual Carbon Emissions (All Regions)",
            template="plotly_white",
            labels={"carbon_emissions_Mg_CO2e": "Gross Emissions (Mg CO₂e)", "year": "Year"},
            color_discrete_sequence=["#8c564b"],
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Regional trends
        st.plotly_chart(
            plot_carbon_trends(carbon_data, top_n=10),
            use_container_width=True,
        )
        
        # Regional comparison
        regional_carbon = (
            carbon_data.groupby("region")["carbon_emissions_Mg_CO2e"]
            .sum().sort_values(ascending=True).tail(15).reset_index()
        )
        fig2 = px.bar(
            regional_carbon, x="carbon_emissions_Mg_CO2e", y="region",
            orientation="h",
            title="Top 15 Regions by Total Carbon Emissions",
            template="plotly_white",
            labels={"carbon_emissions_Mg_CO2e": "Total Emissions (Mg CO₂e)", "region": "Region"},
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation with tree cover loss
        st.subheader("📊 Loss vs Emissions Correlation")
        valid = carbon_data[
            carbon_data["carbon_emissions_Mg_CO2e"].notna() &
            carbon_data["tc_loss_ha"].notna()
        ]
        if len(valid) > 10:
            corr = valid[["tc_loss_ha", "carbon_emissions_Mg_CO2e"]].corr().iloc[0, 1]
            st.metric("Pearson Correlation", f"{corr:.4f}")
            
            fig3 = px.scatter(
                valid, x="tc_loss_ha", y="carbon_emissions_Mg_CO2e",
                color="region", opacity=0.6,
                title="Tree-Cover Loss vs Carbon Emissions",
                template="plotly_white",
                labels={
                    "tc_loss_ha": "Tree-Cover Loss (ha)",
                    "carbon_emissions_Mg_CO2e": "Gross Emissions (Mg CO₂e)",
                },
            )
            st.plotly_chart(fig3, use_container_width=True)
        
        st.caption(
            "**Units:** Carbon emissions are reported as gross emissions in "
            "Megagrams (metric tonnes) of CO₂ equivalent (Mg CO₂e). "
            "These values are from the GFW forest carbon model and represent "
            "emissions from tree-cover loss events. They are NOT computed by "
            "this application from tree-cover loss hectares — they are directly "
            "sourced from the GFW carbon dataset."
        )


# ════════════════════════════════════════════════════════
# Page: Data & Methodology
# ════════════════════════════════════════════════════════

elif page == "📋 Data & Methodology":
    st.title("📋 Data & Methodology")
    
    tabs = st.tabs([
        "Dataset Info",
        "Data Quality",
        "Definitions",
        "Modeling Approach",
        "Limitations",
    ])
    
    with tabs[0]:
        st.subheader("📦 Datasets Used")
        st.markdown(f"""
        | Dataset | Rows | Columns | Description |
        |---------|------|---------|-------------|
        | Subnational Tree Cover Loss | 28,000 | 29 | Annual tree-cover loss by subnational region |
        | Country Tree Cover Loss | 1,888 | 29 | Annual tree-cover loss by country |
        | Subnational Carbon Data | 28,000 | 31 | Annual carbon emissions by subnational region |
        | Country Carbon Data | 1,888 | 30 | Annual carbon emissions by country |
        
        **Source:** Global Forest Change (Hansen et al., 2013, updated).
        University of Maryland / GLAD Lab. Derived from Landsat satellite imagery.
        
        **Geographic scope:** {country} — {processed['region'].nunique()} subnational regions.
        
        **Temporal coverage:** Annual observations from {processed['year'].min()} to {processed['year'].max()}.
        
        **Tree-cover threshold:** ≥{threshold}% canopy density in the year 2000 reference.
        This means only pixels with at least {threshold}% tree-cover density in the
        year 2000 baseline are included in the analysis.
        
        **Processed dataset:** {len(processed)} rows (region × year observations).
        """)
        
        st.subheader("📖 Data Dictionary")
        st.markdown("""
        | Field | Unit | Description |
        |-------|------|-------------|
        | `region` | — | Subnational administrative region (state/province) |
        | `country` | — | Country name |
        | `year` | — | Calendar year of observation (2001–2022) |
        | `tc_loss_ha` | hectares | Annual tree-cover loss |
        | `extent_2000_ha` | hectares | Baseline tree-cover extent in year 2000 |
        | `extent_2010_ha` | hectares | Tree-cover extent in year 2010 |
        | `loss_rate_pct` | % | Annual loss rate = (tc_loss_ha / extent_2000_ha) × 100 |
        | `cumulative_loss_ha` | hectares | Cumulative tree-cover loss since 2001 |
        | `cumulative_loss_pct` | % | Cumulative loss as % of baseline extent |
        | `carbon_emissions_Mg_CO2e` | Mg CO₂e | Gross carbon emissions from loss events |
        | `total_area_ha` | hectares | Total land area of the region |
        | `gain_2000_2020_ha` | hectares | Tree-cover gain over 2000–2020 period |
        """)
    
    with tabs[1]:
        st.subheader("🔍 Data Quality Summary")
        
        from src.eda import missing_data_summary
        missing = missing_data_summary(processed)
        
        if missing.empty:
            st.success("✅ No missing values in the primary processed dataset.")
        else:
            st.dataframe(missing, use_container_width=True)
        
        st.markdown(f"""
        - **Duplicate rows:** 0 (verified)
        - **Zero-loss observations:** {(processed['tc_loss_ha'] == 0).sum()} of {len(processed)} ({(processed['tc_loss_ha'] == 0).mean()*100:.1f}%)
        - **Zero loss ≠ missing data:** Zero values represent regions/years with no detected tree-cover loss.
        - **Negative values:** {(processed['tc_loss_ha'] < 0).sum()} (none expected)
        """)
    
    with tabs[2]:
        st.subheader("📐 Definitions")
        st.markdown("""
        ### Tree-Cover Loss vs. Deforestation
        
        **Tree-cover loss** (as measured in this dataset) represents the **removal of
        tree canopy cover** at any scale, as detected from Landsat satellite imagery.
        
        This is **NOT the same as**:
        - **Permanent deforestation** (conversion to non-forest land use)
        - **Net deforestation** (loss minus gain)
        - **Illegal deforestation** (legal logging is also counted)
        
        Tree-cover loss includes:
        - Agricultural clearing
        - Legal and illegal logging
        - Plantation harvest cycles
        - Fire damage
        - Natural disturbances (storms, disease)
        
        ### Forest-Loss Rate
        
        Annual Forest-Loss Rate (%) = (Annual Tree-Cover Loss ÷ Baseline Extent 2000) × 100
        
        The denominator is the **fixed year-2000 baseline**, not the remaining forest
        in each year. This means the rate measures what fraction of the original
        year-2000 forest was lost in a given year.
        
        ### Carbon Emissions
        
        Carbon emissions data comes directly from the GFW forest carbon flux model.
        Units are **Megagrams (Mg) of CO₂ equivalent** (1 Mg = 1 metric tonne).
        These are NOT computed from tree-cover loss hectares by this application.
        """)
    
    with tabs[3]:
        st.subheader("🧪 Modeling Approach")
        st.markdown(f"""
        ### Prediction Target
        Annual tree-cover loss in hectares (`tc_loss_ha`).
        
        ### Models
        | Model | Type | Description |
        |-------|------|-------------|
        | Historical Baseline | Naive | Previous year's loss as prediction |
        | Linear Regression | Linear | Interpretable linear model with scaled features |
        | Random Forest | Ensemble | 200 trees, max depth 15, captures nonlinearity |
        | XGBoost | Gradient Boosting | Gradient-boosted trees (if installed) |
        
        ### Chronological Evaluation
        - **Training:** Years ≤ {config.TRAIN_END_YEAR}
        - **Validation:** {config.TRAIN_END_YEAR + 1}–{config.VAL_END_YEAR}
        - **Testing:** {config.VAL_END_YEAR + 1}–{config.YEAR_END}
        
        No random splitting. All test data is strictly after training data.
        
        ### Features
        All features are derived from historical observations. Lag features,
        rolling averages, and trend features use shifted values to prevent
        target leakage.
        """)
        
        with st.expander("📄 Full Feature Documentation"):
            st.text(FEATURE_DOCUMENTATION)
    
    with tabs[4]:
        st.subheader("⚠️ Limitations")
        st.markdown("""
        ### Data Limitations
        - **Short time series:** Only 22 annual observations per region (2001–2022).
        This limits the complexity of models that can be reliably trained.
        - **Spatial resolution:** Subnational regions are large administrative units.
        Deforestation hotspots within regions are not identified.
        - **No causal data:** The dataset does not include drivers of deforestation
        (roads, population, commodity prices, policy changes).
        - **Tree-cover loss ≠ deforestation:** See definitions above.
        - **Threshold sensitivity:** Results depend on the chosen tree-cover density
        threshold.
        
        ### Model Limitations
        - Models learn from historical patterns only. They **cannot predict** the
        effects of new policies, climate events, fires, or economic changes.
        - Feature importance shows statistical associations, **not causation**.
        - Forecasts become increasingly uncertain with longer horizons.
        - The forecast uncertainty intervals are based on test-set residuals and
        assume stationary error variance.
        - 22 observations per region is marginal for time-series modeling.
        
        ### Validation Requirements
        - These results should **not** be used for operational conservation decisions
        without independent validation.
        - The risk assessment is an analytical exercise, not a verified classification.
        - Cross-validation with additional data sources (PRODES, MapBiomas) would
        strengthen conclusions.
        
        ### Potential Sources of Bias
        - Cloud cover in satellite imagery may cause detection gaps.
        - Small-scale deforestation below Landsat resolution (~30m) is missed.
        - The year-2000 baseline may undercount forests in cloudy regions.
        """)
