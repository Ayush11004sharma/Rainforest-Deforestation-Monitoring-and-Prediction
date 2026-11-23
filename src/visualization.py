"""
Visualization Module
====================
Creates charts and visual analysis for the deforestation project.
Used by both notebooks and the Streamlit dashboard.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict
import logging

logger = logging.getLogger("deforestation_ml.visualization")

TEMPLATE = "plotly_white"


def plot_annual_trend(
    df: pd.DataFrame,
    value_col: str = "tc_loss_ha",
    title: str = "Annual Tree-Cover Loss",
    y_label: str = "Tree-Cover Loss (hectares)",
    color_col: Optional[str] = None,
) -> go.Figure:
    """Line chart of annual forest loss, optionally by region."""
    if color_col:
        fig = px.line(
            df, x="year", y=value_col, color=color_col,
            title=title, template=TEMPLATE,
            labels={"year": "Year", value_col: y_label},
        )
    else:
        annual = df.groupby("year")[value_col].sum().reset_index()
        fig = px.line(
            annual, x="year", y=value_col,
            title=title, template=TEMPLATE,
            labels={"year": "Year", value_col: y_label},
            markers=True,
        )
    
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=y_label,
        hovermode="x unified",
    )
    return fig


def plot_total_annual_loss(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
) -> go.Figure:
    """Bar chart of total annual forest loss across all regions."""
    annual = df.groupby("year")[target_col].sum().reset_index()
    
    fig = px.bar(
        annual, x="year", y=target_col,
        title="Total Annual Tree-Cover Loss (All Regions)",
        template=TEMPLATE,
        labels={"year": "Year", target_col: "Tree-Cover Loss (ha)"},
        color_discrete_sequence=["#d62728"],
    )
    fig.update_layout(xaxis_title="Year", yaxis_title="Tree-Cover Loss (hectares)")
    return fig


def plot_regional_comparison(
    df: pd.DataFrame,
    target_col: str = "tc_loss_ha",
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of total loss by region."""
    regional = (
        df.groupby("region")[target_col]
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
        .reset_index()
    )
    
    fig = px.bar(
        regional, x=target_col, y="region",
        orientation="h",
        title=f"Top {top_n} Regions by Total Tree-Cover Loss (2001–2022)",
        template=TEMPLATE,
        labels={target_col: "Total Tree-Cover Loss (ha)", "region": "Region"},
        color_discrete_sequence=["#ff7f0e"],
    )
    fig.update_layout(yaxis_title="Region", xaxis_title="Total Tree-Cover Loss (hectares)")
    return fig


def plot_heatmap(
    df: pd.DataFrame,
    value_col: str = "tc_loss_ha",
    title: str = "Tree-Cover Loss Heatmap (Region × Year)",
) -> go.Figure:
    """Heatmap of loss values across regions and years."""
    pivot = df.pivot_table(
        index="region", columns="year", values=value_col, aggfunc="sum"
    )
    pivot = pivot.fillna(0)
    
    # Sort by total loss
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index,
        colorscale="YlOrRd",
        colorbar_title="Loss (ha)",
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Region",
        template=TEMPLATE,
        height=max(400, len(pivot) * 25),
    )
    return fig


def plot_loss_rate_heatmap(
    df: pd.DataFrame,
) -> go.Figure:
    """Heatmap of annual loss rate (%)."""
    return plot_heatmap(
        df,
        value_col="loss_rate_pct",
        title="Annual Forest-Loss Rate (%) — Region × Year",
    )


def plot_cumulative_loss(
    df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Line chart of cumulative forest loss over time."""
    if "cumulative_loss_ha" not in df.columns:
        return go.Figure().add_annotation(text="Cumulative loss data not available")
    
    top_regions = (
        df.groupby("region")["tc_loss_ha"].sum()
        .nlargest(top_n).index
    )
    plot_data = df[df["region"].isin(top_regions)]
    
    fig = px.line(
        plot_data, x="year", y="cumulative_loss_ha", color="region",
        title=f"Cumulative Tree-Cover Loss — Top {top_n} Regions",
        template=TEMPLATE,
        labels={
            "year": "Year",
            "cumulative_loss_ha": "Cumulative Loss (ha)",
            "region": "Region",
        },
    )
    return fig


def plot_actual_vs_predicted(
    prediction_df: pd.DataFrame,
    region: Optional[str] = None,
    model_col: str = "pred_random_forest",
) -> go.Figure:
    """Scatter/line plot of actual vs predicted values."""
    if region:
        data = prediction_df[prediction_df["region"] == region]
        title = f"Actual vs Predicted — {region}"
    else:
        data = prediction_df
        title = "Actual vs Predicted Tree-Cover Loss"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["actual"], y=data[model_col],
        mode="markers",
        name="Predictions",
        marker=dict(color="#1f77b4", size=8, opacity=0.7),
    ))
    
    # Perfect prediction line
    min_val = min(data["actual"].min(), data[model_col].min())
    max_val = max(data["actual"].max(), data[model_col].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect Prediction",
        line=dict(color="red", dash="dash"),
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Loss (ha)",
        yaxis_title="Predicted Loss (ha)",
        template=TEMPLATE,
    )
    return fig


def plot_forecast(
    forecast_df: pd.DataFrame,
    region: str,
    title: Optional[str] = None,
) -> go.Figure:
    """Plot historical observations and future forecasts."""
    data = forecast_df[forecast_df["region"] == region].copy()
    
    if title is None:
        title = f"Deforestation Forecast — {region}"
    
    fig = go.Figure()
    
    # Observed values
    observed = data[data["type"] == "observed"]
    fig.add_trace(go.Scatter(
        x=observed["year"], y=observed["predicted_loss_ha"],
        mode="lines+markers",
        name="Observed",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6),
    ))
    
    # Forecast values
    forecast = data[data["type"] == "forecast"]
    fig.add_trace(go.Scatter(
        x=forecast["year"], y=forecast["predicted_loss_ha"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#d62728", width=2, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    
    # Uncertainty bounds
    if "lower_bound" in forecast.columns and "upper_bound" in forecast.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast["year"], forecast["year"][::-1]]),
            y=pd.concat([forecast["upper_bound"], forecast["lower_bound"][::-1]]),
            fill="toself",
            fillcolor="rgba(214, 39, 40, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval",
            showlegend=True,
        ))
    
    # Vertical line at last observed year
    if not observed.empty and not forecast.empty:
        last_year = observed["year"].max()
        fig.add_vline(
            x=last_year + 0.5, line_dash="dot", line_color="gray",
            annotation_text="Forecast starts",
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Tree-Cover Loss (hectares)",
        template=TEMPLATE,
        hovermode="x unified",
    )
    return fig


def plot_risk_distribution(
    risk_df: pd.DataFrame,
) -> go.Figure:
    """Bar chart of risk category distribution."""
    from config import RISK_COLORS
    
    counts = risk_df["risk_category"].value_counts().reindex(
        ["Low", "Moderate", "High", "Very High"], fill_value=0
    ).reset_index()
    counts.columns = ["Risk Category", "Count"]
    
    colors = [RISK_COLORS.get(cat, "#999") for cat in counts["Risk Category"]]
    
    fig = go.Figure(go.Bar(
        x=counts["Risk Category"],
        y=counts["Count"],
        marker_color=colors,
        text=counts["Count"],
        textposition="auto",
    ))
    
    fig.update_layout(
        title="Distribution of Deforestation Risk Categories",
        xaxis_title="Risk Category",
        yaxis_title="Number of Regions",
        template=TEMPLATE,
    )
    return fig


def plot_risk_ranking(
    risk_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of regions ranked by risk score."""
    from config import RISK_COLORS
    
    data = risk_df.nlargest(top_n, "risk_score")
    colors = [RISK_COLORS.get(cat, "#999") for cat in data["risk_category"]]
    
    fig = go.Figure(go.Bar(
        x=data["risk_score"],
        y=data["region"],
        orientation="h",
        marker_color=colors,
        text=data["risk_category"],
        textposition="auto",
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Regions by Deforestation Risk Score",
        xaxis_title="Risk Score (0–1)",
        yaxis_title="Region",
        template=TEMPLATE,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_carbon_trends(
    carbon_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Line chart of annual carbon emissions by region."""
    if carbon_df.empty or "carbon_emissions_Mg_CO2e" not in carbon_df.columns:
        fig = go.Figure()
        fig.add_annotation(text="Carbon emissions data not available")
        return fig
    
    top_regions = (
        carbon_df.groupby("region")["carbon_emissions_Mg_CO2e"]
        .sum().nlargest(top_n).index
    )
    plot_data = carbon_df[carbon_df["region"].isin(top_regions)]
    
    fig = px.line(
        plot_data, x="year", y="carbon_emissions_Mg_CO2e", color="region",
        title=f"Annual Carbon Emissions — Top {top_n} Regions",
        template=TEMPLATE,
        labels={
            "year": "Year",
            "carbon_emissions_Mg_CO2e": "Gross Emissions (Mg CO₂e)",
            "region": "Region",
        },
    )
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
) -> go.Figure:
    """Bar chart comparing model performance metrics."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["MAE (hectares)", "RMSE (hectares)", "R²"],
    )
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    for i, metric in enumerate(["MAE", "RMSE", "R2"]):
        fig.add_trace(
            go.Bar(
                x=comparison_df["Model"],
                y=comparison_df[metric],
                marker_color=colors[:len(comparison_df)],
                name=metric,
                showlegend=False,
            ),
            row=1, col=i+1,
        )
    
    fig.update_layout(
        title="Model Performance Comparison",
        template=TEMPLATE,
        height=400,
    )
    return fig
