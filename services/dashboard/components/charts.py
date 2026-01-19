"""Reusable chart components for the dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List
import structlog

logger = structlog.get_logger("charts")


def create_metric_card(title: str, value: float, unit: str, status: str = "normal") -> Dict:
    """Create metric card data structure.

    Args:
        title: Metric title
        value: Current value
        unit: Unit of measurement
        status: Status indicator (normal, warning, critical)

    Returns:
        Dictionary with metric data
    """
    return {
        "title": title,
        "value": value,
        "unit": unit,
        "status": status
    }


def create_time_series_chart(df: pd.DataFrame, y_column: str, title: str, y_label: str) -> go.Figure:
    """Create a time series line chart.

    Args:
        df: DataFrame with timestamp and sensor data
        y_column: Column name for y-axis
        title: Chart title
        y_label: Y-axis label

    Returns:
        Plotly figure object
    """
    logger.debug("creating_time_series_chart", column=y_column, rows=len(df), title=title)

    if df.empty:
        logger.warning("empty_dataframe_for_chart", column=y_column, title=title)
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title=title, height=300)
        return fig

    fig = px.line(
        df,
        x="timestamp",
        y=y_column,
        title=title,
        labels={"timestamp": "Time", y_column: y_label},
        template="plotly_white"
    )

    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        height=300,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig


def create_gauge_chart(value: float, title: str, min_val: float, max_val: float,
                       thresholds: Dict[str, float]) -> go.Figure:
    """Create a gauge chart with color zones.

    Args:
        value: Current value
        title: Chart title
        min_val: Minimum value
        max_val: Maximum value
        thresholds: Dictionary with warning and critical thresholds

    Returns:
        Plotly figure object
    """
    # Determine color based on thresholds
    if value >= thresholds.get("critical", max_val):
        color = "#ef4444"  # red
    elif value >= thresholds.get("warning", max_val * 0.8):
        color = "#f59e0b"  # amber
    else:
        color = "#10b981"  # green

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": color},
            "steps": [
                {"range": [min_val, thresholds.get("warning", max_val * 0.7)], "color": "#e0f2fe"},
                {"range": [thresholds.get("warning", max_val * 0.7), thresholds.get("critical", max_val * 0.9)], "color": "#fef3c7"},
                {"range": [thresholds.get("critical", max_val * 0.9), max_val], "color": "#fee2e2"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": thresholds.get("critical", max_val)
            }
        }
    ))

    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_multi_sensor_chart(df: pd.DataFrame, title: str = "All Sensors") -> go.Figure:
    """Create a chart with all sensor readings on separate y-axes.

    Args:
        df: DataFrame with timestamp and all sensor columns
        title: Chart title

    Returns:
        Plotly figure object
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title=title, height=400)
        return fig

    fig = go.Figure()

    # Temperature
    if "temperature" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["temperature"],
            name="Temperature (°C)",
            line=dict(color="#ef4444", width=2),
            yaxis="y1"
        ))

    # Vibration
    if "vibration" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["vibration"],
            name="Vibration (mm/s)",
            line=dict(color="#3b82f6", width=2),
            yaxis="y2"
        ))

    # Pressure
    if "pressure" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["pressure"],
            name="Pressure (PSI)",
            line=dict(color="#10b981", width=2),
            yaxis="y3"
        ))

    # Power
    if "power" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["power"],
            name="Power (W)",
            line=dict(color="#f59e0b", width=2),
            yaxis="y4"
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(domain=[0.1, 0.9]),
        yaxis=dict(title="Temp (°C)", titlefont=dict(color="#ef4444"), tickfont=dict(color="#ef4444"), anchor="free", side="left", position=0.0),
        yaxis2=dict(title="Vib (mm/s)", titlefont=dict(color="#3b82f6"), tickfont=dict(color="#3b82f6"), anchor="free", overlaying="y", side="left", position=0.05),
        yaxis3=dict(title="Press (PSI)", titlefont=dict(color="#10b981"), tickfont=dict(color="#10b981"), anchor="free", overlaying="y", side="right", position=0.95),
        yaxis4=dict(title="Power (W)", titlefont=dict(color="#f59e0b"), tickfont=dict(color="#f59e0b"), anchor="free", overlaying="y", side="right", position=1.0),
        height=400,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=80, r=80, t=40, b=40)
    )

    return fig


def create_histogram(data: List[float], title: str, x_label: str) -> go.Figure:
    """Create a histogram for sensor value distribution.

    Args:
        data: List of sensor values
        title: Chart title
        x_label: X-axis label

    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        x=data,
        title=title,
        labels={"x": x_label, "y": "Count"},
        template="plotly_white",
        nbins=30
    )

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )

    return fig


def create_health_score_chart(machines_df: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing health scores for all machines.

    Args:
        machines_df: DataFrame with machine_id and sensor readings

    Returns:
        Plotly figure object
    """
    if machines_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray")
        )
        fig.update_layout(title="Machine Health Scores", height=400)
        return fig

    # Simple health score calculation (can be improved with ML model)
    def calculate_health_score(row):
        score = 100
        # Temperature penalty (normal range: 40-70°C)
        if row.get("temperature", 55) > 70:
            score -= min((row["temperature"] - 70) * 2, 30)
        # Vibration penalty (normal range: 0.5-1.5 mm/s)
        if row.get("vibration", 1.0) > 1.5:
            score -= min((row["vibration"] - 1.5) * 20, 30)
        # Pressure penalty (normal range: 40-60 PSI)
        if row.get("pressure", 50) < 40 or row.get("pressure", 50) > 60:
            score -= min(abs(row["pressure"] - 50) * 1.5, 20)
        # Power penalty (normal range: 200-300W)
        if row.get("power", 250) > 300:
            score -= min((row["power"] - 300) * 0.1, 20)

        return max(0, score)

    machines_df["health_score"] = machines_df.apply(calculate_health_score, axis=1)

    # Color based on health score
    colors = ["#10b981" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
              for score in machines_df["health_score"]]

    fig = go.Figure(data=[
        go.Bar(
            x=machines_df["machine_id"],
            y=machines_df["health_score"],
            marker_color=colors,
            text=machines_df["health_score"].round(1),
            textposition="outside"
        )
    ])

    fig.update_layout(
        title="Machine Health Scores",
        xaxis_title="Machine ID",
        yaxis_title="Health Score (%)",
        yaxis=dict(range=[0, 105]),
        height=400,
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig
