"""
TRKR Visuals Utilities
======================
Plotly-based visualization wrappers for F1 race data, inspired by F1 Dash design.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional


def position_gap_chart(laps_df: pd.DataFrame, race_name: str = "Race") -> go.Figure:
    """
    Create a position-gap chart showing how driver gaps change over laps.
    
    Args:
        laps_df: DataFrame with columns [lap, driver, gap_to_leader]
        race_name: Name of the race for title
    
    Returns:
        Plotly figure
    """
    if laps_df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    fig = px.line(
        laps_df,
        x="lap",
        y="gap_to_leader",
        color="driver",
        title=f"Gap to Leader Over Laps — {race_name}",
        labels={"lap": "Lap", "gap_to_leader": "Gap (s)"},
        height=500
    )
    
    fig.update_layout(
        hovermode="x unified",
        template="plotly_dark",
        font=dict(size=12),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def driver_telemetry_chart(telemetry_df: pd.DataFrame, driver: str) -> go.Figure:
    """
    Create speed/throttle/brake profile for a driver on a lap.
    
    Args:
        telemetry_df: DataFrame with columns [distance, Speed, Throttle, Brake]
        driver: Driver name for title
    
    Returns:
        Plotly figure with subplots
    """
    if telemetry_df.empty:
        return go.Figure().add_annotation(text="No telemetry available")
    
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Speed", "Throttle", "Brake"),
        shared_xaxes=True,
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=telemetry_df['Distance'], y=telemetry_df['Speed'], 
                   name="Speed", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=telemetry_df['Distance'], y=telemetry_df['Throttle'], 
                   name="Throttle", line=dict(color="green")),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=telemetry_df['Distance'], y=telemetry_df['Brake'], 
                   name="Brake", line=dict(color="red")),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Distance (m)", row=3, col=1)
    fig.update_layout(
        title_text=f"Telemetry — {driver}",
        height=600,
        template="plotly_dark",
        hovermode="x unified"
    )
    
    return fig


def leaderboard_chart(results_df: pd.DataFrame, title: str = "Leaderboard") -> go.Figure:
    """
    Create a horizontal bar chart of final positions with points.
    
    Args:
        results_df: DataFrame with columns [Position, Driver, Team, Points]
        title: Chart title
    
    Returns:
        Plotly figure
    """
    if results_df.empty:
        return go.Figure().add_annotation(text="No results available")
    
    fig = px.barh(
        results_df.sort_values("Points", ascending=True),
        x="Points",
        y="Driver",
        color="Team",
        title=title,
        labels={"Points": "Points Earned"},
        height=600
    )
    
    fig.update_layout(
        template="plotly_dark",
        showlegend=True,
        hovermode="y"
    )
    
    return fig


def prediction_confidence_gauge(confidence_score: float, confidence_level: str) -> go.Figure:
    """
    Create a gauge chart for prediction confidence.
    
    Args:
        confidence_score: 0-100 score
        confidence_level: "HIGH", "MEDIUM", or "LOW"
    
    Returns:
        Plotly gauge figure
    """
    color_map = {
        "HIGH": "#2ecc71",
        "MEDIUM": "#f39c12",
        "LOW": "#e74c3c"
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_score,
        title={'text': f"Prediction Confidence — {confidence_level}"},
        delta={'reference': 85, 'suffix': " from target"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color_map.get(confidence_level, "#95a5a6")},
            'steps': [
                {'range': [0, 50], 'color': "#e74c3c"},
                {'range': [50, 85], 'color': "#f39c12"},
                {'range': [85, 100], 'color': "#2ecc71"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        font=dict(size=14),
        height=400
    )
    
    return fig


def team_strength_chart(team_data: Dict[str, float], title: str = "Team Strength Index") -> go.Figure:
    """
    Create horizontal bar chart of team strength scores.
    
    Args:
        team_data: Dict of {team_name: strength_score}
        title: Chart title
    
    Returns:
        Plotly figure
    """
    df = pd.DataFrame(list(team_data.items()), columns=["Team", "Strength"])
    df = df.sort_values("Strength", ascending=True)
    
    fig = px.barh(
        df,
        x="Strength",
        y="Team",
        title=title,
        labels={"Strength": "Strength Score (0-100)"},
        color="Strength",
        color_continuous_scale="RdYlGn",
        height=400
    )
    
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig


def prediction_vs_reality_heatmap(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap comparing predicted vs actual positions.
    
    Args:
        comparison_df: DataFrame with [Driver, Predicted, Actual] columns
    
    Returns:
        Plotly heatmap figure
    """
    pivot = comparison_df.pivot_table(
        values="Delta",
        index="Driver",
        aggfunc="first"
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.flatten(),
        x=["Δ (Predicted vs Actual)"],
        y=pivot.index,
        colorscale="RdYlGn_r",
        text=pivot.values.flatten(),
        texttemplate="%{text:.1f}",
        hovertemplate="<b>%{y}</b><br>Δ: %{text:.1f}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Prediction Accuracy Heatmap",
        template="plotly_dark",
        height=500
    )
    
    return fig
