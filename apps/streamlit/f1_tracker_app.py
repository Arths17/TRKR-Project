"""
F1 Prediction Tracker - Production-Ready Streamlit Application
================================================================
A comprehensive F1 race prediction tracking system with full visualization,
validation, exports, and analysis capabilities.

Features:
- Race selection with metadata validation
- Frozen predictions with confidence scoring
- Official results comparison
- Team strength index analysis
- Fastest lap tracking
- Data quality validation
- Interactive visualizations
- CSV/Excel exports
"""

import sys
from pathlib import Path

# Add project root to path (app/ is 2 directories up from apps/streamlit/)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import io

# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Database imports
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app import models
from app.models import Race, Prediction, PredictionEntry, RaceResult, EvaluationMetric

# FastF1 for schedule data
try:
    import fastf1
except ImportError:
    fastf1 = None

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="F1 Prediction Tracker",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 0.25rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Status badges */
    .badge-success {
        background-color: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-warning {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-danger {
        background-color: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-info {
        background-color: #3b82f6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Data quality indicators */
    .quality-excellent {
        color: #10b981;
        font-weight: 600;
    }
    
    .quality-good {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .quality-poor {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .error-box {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .info-box {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .success-box {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE & CACHING
# =============================================================================

@st.cache_resource
def get_db_session() -> Session:
    """Get a database session (cached)."""
    return SessionLocal()

@st.cache_data(ttl=3600)
def load_all_races() -> List[Dict]:
    """Load all races from database with caching."""
    db = SessionLocal()
    try:
        races = db.query(Race).order_by(Race.year.desc(), Race.round.desc()).all()
        return [
            {
                'id': r.id,
                'year': r.year,
                'round': r.round,
                'name': r.name,
                'circuit': r.circuit,
                'event_date': r.event_date
            }
            for r in races
        ]
    finally:
        db.close()

@st.cache_data(ttl=3600)
def fetch_fastf1_schedule(year: int) -> Optional[pd.DataFrame]:
    """Fetch FastF1 schedule for a given year."""
    if fastf1 is None:
        return None
    try:
        fastf1.Cache.enable_cache('cache')
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        st.warning(f"Could not fetch FastF1 schedule for {year}: {e}")
        return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_confidence(level: str, score: float) -> str:
    """Format confidence level with color and score."""
    score_int = int(round(score))
    if level == "HIGH":
        return f"🟢 **HIGH** ({score_int}/100)"
    elif level == "MEDIUM":
        return f"🟡 **MEDIUM** ({score_int}/100)"
    else:
        return f"🔴 **LOW** ({score_int}/100)"

def format_coverage(coverage: float) -> str:
    """Format feature coverage percentage."""
    if coverage >= 0.85:
        return f"✅ **Excellent** ({coverage*100:.1f}%)"
    elif coverage >= 0.70:
        return f"⚠️ **Good** ({coverage*100:.1f}%)"
    else:
        return f"❌ **Below Target** ({coverage*100:.1f}%)"

def format_time(seconds: Optional[float]) -> str:
    """Format seconds to MM:SS.mmm format."""
    if seconds is None or pd.isna(seconds):
        return "N/A"
    if seconds < 0:
        return "N/A"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"

def format_gap(gap: Optional[float]) -> str:
    """Format gap in seconds."""
    if gap is None or pd.isna(gap) or gap < 0:
        return "—"
    if gap == 0:
        return "—"
    return f"+{gap:.3f}s"

def get_position_emoji(position: int) -> str:
    """Get emoji for podium positions."""
    if position == 1:
        return "🥇"
    elif position == 2:
        return "🥈"
    elif position == 3:
        return "🥉"
    return ""

def calculate_team_strength(entries: List[PredictionEntry]) -> Dict[str, float]:
    """
    Calculate team strength index (0-100 scale).
    Formula: 100 * (21 - avg_position) / 20
    """
    team_positions = {}
    for entry in entries:
        if entry.team not in team_positions:
            team_positions[entry.team] = []
        team_positions[entry.team].append(entry.predicted_position)
    
    team_strength = {}
    for team, positions in team_positions.items():
        avg_pos = np.mean(positions)
        strength = 100 * (21 - avg_pos) / 20
        team_strength[team] = max(0, min(100, strength))
    
    return team_strength

def detect_dark_horses(entries: List[PredictionEntry]) -> List[str]:
    """
    Detect dark horse candidates (P4-P8 within 15s of P3).
    """
    sorted_entries = sorted(entries, key=lambda e: e.predicted_position)
    if len(sorted_entries) < 4:
        return []
    
    p3_time = sorted_entries[2].predicted_race_time if sorted_entries[2].predicted_race_time else 0
    dark_horses = []
    
    for entry in sorted_entries[3:8]:  # P4 to P8
        if entry.predicted_race_time and p3_time:
            gap = abs(entry.predicted_race_time - p3_time)
            if gap <= 15:
                dark_horses.append(entry.driver)
    
    return dark_horses

def check_extreme_gaps(entries: List[PredictionEntry], threshold: float = 120.0) -> List[Tuple[str, float]]:
    """Check for extreme gaps (potential DNF candidates)."""
    extreme_gaps = []
    for entry in entries:
        if entry.gap and entry.gap > threshold:
            extreme_gaps.append((entry.driver, entry.gap))
    return extreme_gaps

def export_to_excel(dataframes: Dict[str, pd.DataFrame], filename: str) -> bytes:
    """Export multiple DataFrames to Excel with multiple sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output.read()

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_race_data(race_id: int) -> Dict:
    """Load comprehensive race data including predictions, results, and metrics."""
    db = SessionLocal()
    try:
        race = db.query(Race).filter(Race.id == race_id).first()
        if not race:
            return None
        
        prediction = db.query(Prediction).filter(Prediction.race_id == race_id).first()
        results = db.query(RaceResult).filter(RaceResult.race_id == race_id).all()
        metrics = db.query(EvaluationMetric).filter(EvaluationMetric.race_id == race_id).first()
        
        prediction_entries = []
        if prediction:
            prediction_entries = db.query(PredictionEntry).filter(
                PredictionEntry.prediction_id == prediction.id
            ).order_by(PredictionEntry.predicted_position).all()
        
        return {
            'race': race,
            'prediction': prediction,
            'prediction_entries': prediction_entries,
            'results': results,
            'metrics': metrics
        }
    finally:
        db.close()

def get_historical_stats() -> Dict:
    """Get historical statistics across all races."""
    db = SessionLocal()
    try:
        total_races = db.query(Race).count()
        total_predictions = db.query(Prediction).count()
        
        metrics = db.query(EvaluationMetric).all()
        if metrics:
            avg_position_mae = np.mean([m.position_mae for m in metrics if m.position_mae])
            avg_time_mae = np.mean([m.time_mae_seconds for m in metrics if m.time_mae_seconds])
            winner_accuracy = np.mean([m.winner_correct for m in metrics if m.winner_correct is not None])
            podium_accuracy = np.mean([m.podium_accuracy for m in metrics if m.podium_accuracy is not None])
        else:
            avg_position_mae = avg_time_mae = winner_accuracy = podium_accuracy = 0
        
        return {
            'total_races': total_races,
            'total_predictions': total_predictions,
            'avg_position_mae': avg_position_mae,
            'avg_time_mae': avg_time_mae,
            'winner_accuracy': winner_accuracy * 100,
            'podium_accuracy': podium_accuracy * 100
        }
    finally:
        db.close()

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_confidence_gauge(confidence_score: float) -> go.Figure:
    """Create a gauge chart for confidence score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 70], 'color': "#fee2e2"},
                {'range': [70, 85], 'color': "#fef3c7"},
                {'range': [85, 100], 'color': "#d1fae5"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 85
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_team_strength_chart(team_strength: Dict[str, float]) -> go.Figure:
    """Create horizontal bar chart for team strength index."""
    teams = list(team_strength.keys())
    strengths = list(team_strength.values())
    
    # Sort by strength descending
    sorted_data = sorted(zip(teams, strengths), key=lambda x: x[1], reverse=True)
    teams, strengths = zip(*sorted_data) if sorted_data else ([], [])
    
    # Color gradient based on strength
    colors = ['#10b981' if s >= 70 else '#f59e0b' if s >= 50 else '#ef4444' for s in strengths]
    
    fig = go.Figure(go.Bar(
        y=teams,
        x=strengths,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{s:.1f}" for s in strengths],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Team Strength Index (0-100)",
        xaxis_title="Strength Score",
        yaxis_title="Team",
        height=max(300, len(teams) * 40),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig

def create_prediction_vs_actual_chart(prediction_entries: List[PredictionEntry], 
                                     results: List[RaceResult]) -> go.Figure:
    """Create comparison chart of predicted vs actual positions."""
    if not results:
        return None
    
    # Create mapping
    result_map = {r.driver: r.position for r in results}
    
    drivers = []
    predicted_positions = []
    actual_positions = []
    
    for entry in sorted(prediction_entries, key=lambda e: e.predicted_position):
        if entry.driver in result_map:
            drivers.append(entry.driver)
            predicted_positions.append(entry.predicted_position)
            actual_positions.append(result_map[entry.driver])
    
    fig = go.Figure()
    
    # Predicted positions
    fig.add_trace(go.Scatter(
        x=drivers,
        y=predicted_positions,
        mode='markers+lines',
        name='Predicted',
        marker=dict(size=10, color='#667eea'),
        line=dict(color='#667eea', dash='dash')
    ))
    
    # Actual positions
    fig.add_trace(go.Scatter(
        x=drivers,
        y=actual_positions,
        mode='markers+lines',
        name='Actual',
        marker=dict(size=10, color='#10b981'),
        line=dict(color='#10b981')
    ))
    
    fig.update_layout(
        title="Predicted vs Actual Positions",
        xaxis_title="Driver",
        yaxis_title="Position",
        yaxis=dict(autorange='reversed'),
        height=400,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_fastest_lap_chart(results: List[RaceResult]) -> Optional[go.Figure]:
    """Create chart for fastest lap analysis."""
    # Extract lap times (if available in results)
    # Note: Current schema doesn't have lap times, but this is a placeholder
    # for when that data becomes available
    
    drivers_with_times = [(r.driver, r.time) for r in results if r.time and r.time > 0]
    if not drivers_with_times:
        return None
    
    # Sort by time
    drivers_with_times.sort(key=lambda x: x[1])
    top_10 = drivers_with_times[:10]
    
    drivers, times = zip(*top_10)
    
    # Color top 3
    colors = ['#ffd700' if i == 0 else '#c0c0c0' if i == 1 else '#cd7f32' 
              if i == 2 else '#3b82f6' for i in range(len(drivers))]
    
    fig = go.Figure(go.Bar(
        x=drivers,
        y=times,
        marker=dict(color=colors),
        text=[format_time(t) for t in times],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Race Time Comparison (Top 10)",
        xaxis_title="Driver",
        yaxis_title="Time (seconds)",
        height=400,
        showlegend=False
    )
    
    return fig

def create_position_delta_heatmap(prediction_entries: List[PredictionEntry],
                                  results: List[RaceResult]) -> Optional[go.Figure]:
    """Create heatmap showing position deltas."""
    if not results:
        return None
    
    result_map = {r.driver: r.position for r in results}
    
    drivers = []
    deltas = []
    
    for entry in sorted(prediction_entries, key=lambda e: e.predicted_position):
        if entry.driver in result_map:
            delta = result_map[entry.driver] - entry.predicted_position
            drivers.append(entry.driver)
            deltas.append(delta)
    
    # Create color scale: green for exact, yellow for close, red for far
    colors = []
    for delta in deltas:
        if delta == 0:
            colors.append('#10b981')  # Green
        elif abs(delta) <= 2:
            colors.append('#f59e0b')  # Yellow
        else:
            colors.append('#ef4444')  # Red
    
    fig = go.Figure(go.Bar(
        x=drivers,
        y=deltas,
        marker=dict(color=colors),
        text=[f"{d:+d}" if d != 0 else "✓" for d in deltas],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Position Accuracy (Predicted - Actual)",
        xaxis_title="Driver",
        yaxis_title="Position Delta",
        height=400,
        shapes=[dict(
            type='line',
            x0=-0.5,
            x1=len(drivers)-0.5,
            y0=0,
            y1=0,
            line=dict(color='black', width=2, dash='dash')
        )]
    )
    
    return fig

# =============================================================================
# TABLE BUILDING FUNCTIONS
# =============================================================================

def build_prediction_table(entries: List[PredictionEntry], 
                          dark_horses: List[str],
                          extreme_gaps: List[Tuple[str, float]]) -> pd.DataFrame:
    """Build formatted prediction table with styling indicators."""
    extreme_gap_drivers = {driver for driver, _ in extreme_gaps}
    
    data = []
    for entry in sorted(entries, key=lambda e: e.predicted_position):
        emoji = get_position_emoji(entry.predicted_position)
        
        # Status indicators
        indicators = []
        if entry.driver in dark_horses:
            indicators.append("🌟 Dark Horse")
        if entry.driver in extreme_gap_drivers:
            indicators.append("⚠️ Extreme Gap")
        
        data.append({
            'Pos': f"{emoji} {entry.predicted_position}" if emoji else str(entry.predicted_position),
            'Driver': entry.driver,
            'Team': entry.team,
            'Race Time': format_time(entry.predicted_race_time),
            'Gap': format_gap(entry.gap),
            'Uncertainty': f"±{entry.uncertainty:.1f}s" if entry.uncertainty else "N/A",
            'Notes': " | ".join(indicators) if indicators else ""
        })
    
    return pd.DataFrame(data)

def build_results_table(results: List[RaceResult]) -> pd.DataFrame:
    """Build official results table."""
    data = []
    for result in sorted(results, key=lambda r: r.position):
        emoji = get_position_emoji(result.position)
        
        data.append({
            'Pos': f"{emoji} {result.position}" if emoji else str(result.position),
            'Driver': result.driver,
            'Team': result.team,
            'Time': format_time(result.time) if result.time else "N/A",
            'Status': result.status or "Finished",
            'Points': result.points or 0
        })
    
    return pd.DataFrame(data)

def build_comparison_table(prediction_entries: List[PredictionEntry],
                          results: List[RaceResult]) -> pd.DataFrame:
    """Build prediction vs actual comparison table."""
    result_map = {r.driver: r for r in results}
    
    data = []
    for entry in sorted(prediction_entries, key=lambda e: e.predicted_position):
        if entry.driver not in result_map:
            continue
        
        actual_result = result_map[entry.driver]
        position_delta = actual_result.position - entry.predicted_position
        
        # Accuracy assessment
        if position_delta == 0:
            accuracy = "✅ Exact"
            accuracy_color = "🟢"
        elif abs(position_delta) <= 2:
            accuracy = "✓ Close"
            accuracy_color = "🟡"
        elif abs(position_delta) <= 5:
            accuracy = "~ Off"
            accuracy_color = "🟠"
        else:
            accuracy = "❌ Miss"
            accuracy_color = "🔴"
        
        data.append({
            'Driver': entry.driver,
            'Predicted': entry.predicted_position,
            'Actual': actual_result.position,
            'Delta': f"{position_delta:+d}",
            'Accuracy': f"{accuracy_color} {accuracy}",
            'Pred Time': format_time(entry.predicted_race_time),
            'Actual Time': format_time(actual_result.time) if actual_result.time else "N/A"
        })
    
    return pd.DataFrame(data)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Header
    st.title("🏎️ F1 Prediction Tracker")
    st.markdown("**Production-Ready Race Prediction System** | FastF1 + XGBoost ML")
    st.markdown("---")
    
    # =============================================================================
    # SIDEBAR - RACE SELECTION & SUMMARY STATS
    # =============================================================================
    
    with st.sidebar:
        # Quick stats banner
        stats = get_historical_stats()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center; 
                    margin-bottom: 1rem;'>
            <h3 style='margin: 0; font-size: 1.2rem;'>🏎️ F1 Prediction Tracker</h3>
            <p style='margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.9;'>
                {stats['total_races']} Races • {stats['podium_accuracy']:.0f}% Podium Accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.header("📊 Race Selection")
        
        # Load all races
        races = load_all_races()
        
        if not races:
            st.error("No races found in database. Please run predictions first.")
            st.stop()
        
        # Race selection dropdown
        race_options = {
            f"{r['year']} - Round {r['round']}: {r['name']} ({r['circuit']})": r['id']
            for r in races
        }
        
        selected_race_label = st.selectbox(
            "Select Race",
            options=list(race_options.keys()),
            index=0
        )
        
        selected_race_id = race_options[selected_race_label]
        
        st.markdown("---")
        
        # Summary statistics
        st.header("📈 System Stats")
        
        stats = get_historical_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Races", stats['total_races'])
            st.metric("Avg Position MAE", f"{stats['avg_position_mae']:.2f}")
        with col2:
            st.metric("Predictions", stats['total_predictions'])
            st.metric("Winner Accuracy", f"{stats['winner_accuracy']:.1f}%")
        
        st.metric("Podium Accuracy", f"{stats['podium_accuracy']:.1f}%")
        
        st.markdown("---")
        
        # Data quality section
        st.header("🔍 Data Quality")
        
        # Load current race data for quality checks
        race_data = load_race_data(selected_race_id)
        
        if race_data and race_data['prediction']:
            pred = race_data['prediction']
            
            # Confidence
            if pred.confidence_level == "HIGH":
                st.success(f"✅ Confidence: {pred.confidence_level}")
            elif pred.confidence_level == "MEDIUM":
                st.warning(f"⚠️ Confidence: {pred.confidence_level}")
            else:
                st.error(f"🔴 Confidence: {pred.confidence_level}")
            
            # Coverage (handle both decimal 0-1 and percentage values)
            coverage = pred.feature_coverage or 0
            coverage_pct = coverage if coverage <= 1.0 else coverage / 100
            display_pct = coverage_pct * 100
            
            if coverage_pct >= 0.85:
                st.success(f"✅ Coverage: {display_pct:.1f}%")
            elif coverage_pct >= 0.70:
                st.warning(f"⚠️ Coverage: {display_pct:.1f}%")
            else:
                st.error(f"🔴 Coverage: {display_pct:.1f}%")
            
            # Extreme gaps
            extreme_gaps = check_extreme_gaps(race_data['prediction_entries'])
            if extreme_gaps:
                st.warning(f"⚠️ {len(extreme_gaps)} extreme gap(s)")
            else:
                st.success("✅ No extreme gaps")
            
            # Add explanation expander
            with st.expander("ℹ️ What do these metrics mean?", expanded=False):
                st.markdown("""
                **🎯 Confidence Score**
                - **HIGH (85-100)**: Excellent data quality, ≥85% of features available
                - **MEDIUM (70-84)**: Good data quality, some features imputed
                - **LOW (<70)**: Limited data, many features imputed with fallback values
                
                Higher confidence = more reliable predictions.
                
                ---
                
                **📊 Feature Coverage**
                - Percentage of required ML features successfully retrieved from FastF1
                - **Target: ≥85%** for high confidence predictions
                - Missing features are filled with historical averages
                
                Low coverage (<70%) triggers LOW confidence warnings.
                
                ---
                
                **⚠️ Extreme Gaps**
                - Drivers predicted >120s behind the leader
                - Often indicates potential DNF (mechanical failure, crash, etc.)
                - Normal race gaps are typically <90s for full field
                
                Extreme gaps help identify at-risk predictions.
                """)
    
    # =============================================================================
    # MAIN CONTENT AREA
    # =============================================================================
    
    # Load selected race data
    race_data = load_race_data(selected_race_id)
    
    if not race_data:
        st.error("Race data not found.")
        st.stop()
    
    race = race_data['race']
    prediction = race_data['prediction']
    prediction_entries = race_data['prediction_entries']
    results = race_data['results']
    metrics = race_data['metrics']
    
    # =============================================================================
    # TAB NAVIGATION
    # =============================================================================
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "ℹ️ Race Info",
        "🔮 Predictions",
        "🏁 Results",
        "📊 Analysis",
        "🏆 Team Strength",
        "💾 Export"
    ])
    
    # =============================================================================
    # TAB 1: RACE INFORMATION
    # =============================================================================
    
    with tab1:
        st.header("Race Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Year", race.year if race.year else "⚠️ N/A")
        with col2:
            st.metric("Round", race.round if race.round else "⚠️ N/A")
        with col3:
            st.metric("Circuit", race.circuit if race.circuit else "⚠️ N/A")
        with col4:
            if race.event_date:
                event_date = race.event_date.strftime("%Y-%m-%d")
                # Calculate days until/since race
                today = datetime.now().date()
                race_date = race.event_date.date() if hasattr(race.event_date, 'date') else race.event_date
                days_diff = (race_date - today).days
                
                if days_diff > 0:
                    delta_label = f"🕐 in {days_diff} days"
                elif days_diff == 0:
                    delta_label = "🏁 Today!"
                else:
                    delta_label = f"✓ {abs(days_diff)} days ago"
                
                st.metric("Date", event_date, delta=delta_label)
            else:
                st.metric("Date", "⚠️ N/A")
        
        st.markdown("---")
        
        # Prediction metadata
        if prediction:
            st.subheader("🔮 Prediction Snapshot")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Confidence Score**")
                st.markdown(format_confidence(prediction.confidence_level, prediction.confidence_score))
                
                # Confidence gauge
                st.plotly_chart(
                    create_confidence_gauge(prediction.confidence_score),
                    width='stretch'
                )
            
            with col2:
                st.markdown("**Feature Coverage**")
                st.markdown(format_coverage(prediction.feature_coverage))
                
                # Coverage progress bar (ensure value is between 0.0 and 1.0)
                coverage_value = prediction.feature_coverage if prediction.feature_coverage <= 1.0 else prediction.feature_coverage / 100
                st.progress(min(1.0, max(0.0, coverage_value)))
                
                if prediction.num_imputed:
                    st.caption(f"⚠️ {prediction.num_imputed} features imputed")
            
            with col3:
                st.markdown("**Freeze Policy**")
                st.info(f"📸 {prediction.freeze_policy}")
                
                st.markdown("**Snapshot Time**")
                snapshot_time = prediction.snapshot_ts.strftime("%Y-%m-%d %H:%M UTC") if prediction.snapshot_ts else "N/A"
                st.caption(snapshot_time)
                
                st.markdown("**FastF1 Version**")
                st.caption(prediction.fastf1_version or "N/A")
            
            # Confidence explanation (for all levels)
            with st.expander("ℹ️ Understanding Prediction Confidence", expanded=(prediction.confidence_level == "LOW")):
                coverage_pct = prediction.feature_coverage if prediction.feature_coverage <= 1.0 else prediction.feature_coverage / 100
                
                if prediction.confidence_level == "LOW":
                    st.error(
                        f"**⚠️ LOW Confidence ({prediction.confidence_score:.0f}/100)**\n\n"
                        f"Feature coverage is {coverage_pct*100:.1f}%, below the 85% threshold. "
                        f"This means **{prediction.num_imputed or 'several'} features** were imputed with "
                        f"historical averages, increasing prediction uncertainty.\n\n"
                        f"**Impact:** Predictions are less reliable, especially for mid-field positions."
                    )
                elif prediction.confidence_level == "MEDIUM":
                    st.warning(
                        f"**⚠️ MEDIUM Confidence ({prediction.confidence_score:.0f}/100)**\n\n"
                        f"Feature coverage is {coverage_pct*100:.1f}%, which is good but not optimal. "
                        f"Some features ({prediction.num_imputed or 'a few'}) were imputed.\n\n"
                        f"**Impact:** Predictions are generally reliable, with slight uncertainty."
                    )
                else:
                    st.success(
                        f"**✅ HIGH Confidence ({prediction.confidence_score:.0f}/100)**\n\n"
                        f"Feature coverage is {coverage_pct*100:.1f}%, exceeding the 85% threshold. "
                        f"All or nearly all features were successfully retrieved from FastF1.\n\n"
                        f"**Impact:** Predictions are highly reliable with minimal uncertainty."
                    )
        else:
            st.warning("⏳ No prediction available for this race.")
        
        # Circuit information (from FastF1)
        st.markdown("---")
        st.subheader("🏎️ Circuit Details")
        
        if race.year:
            schedule = fetch_fastf1_schedule(race.year)
            if schedule is not None:
                try:
                    event = schedule[schedule['RoundNumber'] == race.round].iloc[0]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Location", event.get('Location', 'N/A'))
                    with col2:
                        st.metric("Country", event.get('Country', 'N/A'))
                    with col3:
                        st.metric("Event Format", event.get('EventFormat', 'N/A'))
                    
                    st.info(f"**Official Name:** {event.get('EventName', 'N/A')}")
                    
                except Exception as e:
                    st.caption(f"Circuit details not available: {e}")
            else:
                st.caption("FastF1 schedule not available.")
        else:
            st.caption("Race year not set - cannot fetch circuit details.")
    
    # =============================================================================
    # TAB 2: PREDICTIONS
    # =============================================================================
    
    with tab2:
        st.header("🔮 Race Predictions")
        
        if not prediction or not prediction_entries:
            st.warning("⏳ No predictions available for this race.")
        else:
            # Dark horses and extreme gaps
            dark_horses = detect_dark_horses(prediction_entries)
            extreme_gaps = check_extreme_gaps(prediction_entries)
            
            # Alerts
            if dark_horses:
                st.success(f"🌟 **Dark Horse Candidates:** {', '.join(dark_horses)}")
            
            if extreme_gaps:
                gap_text = ", ".join([f"{driver} (+{gap:.1f}s)" for driver, gap in extreme_gaps])
                st.error(f"⚠️ **Extreme Gaps Detected:** {gap_text}")
            
            # Key insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                winner = prediction_entries[0] if prediction_entries else None
                if winner:
                    st.metric("🥇 Predicted Winner", winner.driver)
                    st.caption(f"Team: {winner.team}")
            
            with col2:
                if len(prediction_entries) >= 3:
                    podium = [e.driver for e in prediction_entries[:3]]
                    st.metric("🏆 Podium", " / ".join(podium))
            
            with col3:
                st.metric("🌟 Dark Horses", len(dark_horses))
                st.caption(f"Within 15s of podium")
            
            st.markdown("---")
            
            # Prediction table
            st.subheader("📋 Full Prediction Breakdown")
            
            pred_df = build_prediction_table(prediction_entries, dark_horses, extreme_gaps)
            
            # Interactive filtering
            teams = sorted(list(set([e.team for e in prediction_entries])))
            
            selected_teams = st.multiselect("🏁 Filter by Team", teams, default=teams, key="team_filter")
            
            # Driver search
            search_driver = st.text_input("🔍 Search Driver", placeholder="Type driver name...", key="driver_search")
            
            filtered_df = pred_df.copy()
            if selected_teams:
                filtered_df = filtered_df[filtered_df['Team'].isin(selected_teams)]
            if search_driver:
                filtered_df = filtered_df[filtered_df['Driver'].str.contains(search_driver, case=False, na=False)]
            
            st.caption(f"📊 Showing **{len(filtered_df)}** of **{len(pred_df)}** drivers")
            
            st.dataframe(filtered_df, width='stretch', height=600)
            
            # Download button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{race.year}_R{race.round}_{race.name}.csv",
                mime="text/csv"
            )
    
    # =============================================================================
    # TAB 3: RESULTS
    # =============================================================================
    
    with tab3:
        st.header("🏁 Official Race Results")
        
        if not results:
            st.info("⏳ Race results not yet available. Results will appear after the race is complete.")
        else:
            # Results summary
            col1, col2, col3 = st.columns(3)
            
            winner = next((r for r in results if r.position == 1), None)
            
            with col1:
                if winner:
                    st.metric("🥇 Winner", winner.driver)
                    st.caption(f"Team: {winner.team}")
            
            with col2:
                podium = [r.driver for r in sorted(results, key=lambda x: x.position)[:3]]
                if len(podium) == 3:
                    st.metric("🏆 Podium", " / ".join(podium))
            
            with col3:
                dnf_count = sum(1 for r in results if r.status and r.status != "Finished")
                st.metric("⚠️ DNFs", dnf_count)
            
            st.markdown("---")
            
            # Results table
            results_df = build_results_table(results)
            st.dataframe(results_df, width='stretch', height=600)
            
            # Fastest lap chart
            fastest_lap_fig = create_fastest_lap_chart(results)
            if fastest_lap_fig:
                st.plotly_chart(fastest_lap_fig, width='stretch')
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"results_{race.year}_R{race.round}_{race.name}.csv",
                mime="text/csv"
            )
    
    # =============================================================================
    # TAB 4: ANALYSIS
    # =============================================================================
    
    with tab4:
        st.header("📊 Prediction Analysis")
        
        if not results or not prediction_entries:
            st.info("⏳ Analysis available after race results are synced.")
        else:
            # Comparison summary
            comparison_df = build_comparison_table(prediction_entries, results)
            
            # Accuracy breakdown
            exact_count = len(comparison_df[comparison_df['Accuracy'].str.contains('Exact')])
            close_count = len(comparison_df[comparison_df['Accuracy'].str.contains('Close')])
            off_count = len(comparison_df[comparison_df['Accuracy'].str.contains('Off')])
            miss_count = len(comparison_df[comparison_df['Accuracy'].str.contains('Miss')])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("✅ Exact", exact_count)
            with col2:
                st.metric("✓ Close (±2)", close_count)
            with col3:
                st.metric("~ Off (±5)", off_count)
            with col4:
                st.metric("❌ Miss (>5)", miss_count)
            
            st.markdown("---")
            
            # Comparison table
            st.subheader("🔄 Prediction vs Reality")
            st.dataframe(comparison_df, width='stretch', height=500)
            
            st.markdown("---")
            
            # Metrics
            if metrics:
                st.subheader("📈 Accuracy Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Position MAE", f"{metrics.position_mae:.2f}")
                with col2:
                    st.metric("Time MAE", f"{metrics.time_mae_seconds:.1f}s")
                with col3:
                    winner_icon = "✅" if metrics.winner_correct else "❌"
                    st.metric("Winner Correct", winner_icon)
                with col4:
                    st.metric("Podium Accuracy", f"{metrics.podium_accuracy*100:.0f}%")
            
            st.markdown("---")
            
            # Performance insights
            if metrics:
                st.subheader("💡 Key Insights")
                
                insights = []
                
                # Position accuracy insight
                if metrics.position_mae <= 2.0:
                    insights.append("🎯 **Excellent position accuracy** - Average error ≤2 positions")
                elif metrics.position_mae <= 3.5:
                    insights.append("✅ **Good position accuracy** - Average error ≤3.5 positions")
                else:
                    insights.append("⚠️ **Challenging predictions** - Higher than usual position error")
                
                # Winner prediction
                if metrics.winner_correct:
                    insights.append("🏆 **Winner correctly predicted!**")
                else:
                    insights.append("❌ Winner prediction missed")
                
                # Podium accuracy
                if metrics.podium_accuracy >= 0.67:
                    insights.append(f"🥇 **Strong podium prediction** - {metrics.podium_accuracy*100:.0f}% accuracy")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                st.markdown("---")
            
            # Performance insights
            if metrics:
                st.subheader("💡 Prediction Performance Insights")
                
                col_ins1, col_ins2 = st.columns(2)
                
                with col_ins1:
                    st.markdown("**🎯 Position Accuracy**")
                    if metrics.position_mae <= 2.0:
                        st.success(f"✅ Excellent! Average error: {metrics.position_mae:.2f} positions")
                        st.caption("Predictions were highly accurate")
                    elif metrics.position_mae <= 3.5:
                        st.info(f"🟡 Good! Average error: {metrics.position_mae:.2f} positions")
                        st.caption("Predictions were generally reliable")
                    else:
                        st.warning(f"⚠️ Challenging race. Error: {metrics.position_mae:.2f} positions")
                        st.caption("Higher than usual prediction variance")
                
                with col_ins2:
                    st.markdown("**🏆 Key Predictions**")
                    if metrics.winner_correct:
                        st.success("✅ Winner correctly predicted!")
                    else:
                        st.error("❌ Winner prediction missed")
                    
                    if metrics.podium_accuracy >= 0.67:
                        st.success(f"🥇 Podium: {metrics.podium_accuracy*100:.0f}% accuracy")
                    else:
                        st.warning(f"🔶 Podium: {metrics.podium_accuracy*100:.0f}% accuracy")
                
                st.markdown("---")
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Predicted vs Actual chart
                pred_vs_actual_fig = create_prediction_vs_actual_chart(prediction_entries, results)
                if pred_vs_actual_fig:
                    st.plotly_chart(pred_vs_actual_fig, width='stretch')
            
            with col2:
                # Position delta heatmap
                delta_fig = create_position_delta_heatmap(prediction_entries, results)
                if delta_fig:
                    st.plotly_chart(delta_fig, width='stretch')
    
    # =============================================================================
    # TAB 5: TEAM STRENGTH
    # =============================================================================
    
    with tab5:
        st.header("🏆 Team Strength Index")
        
        if not prediction_entries:
            st.warning("⏳ Team strength analysis requires predictions.")
        else:
            team_strength = calculate_team_strength(prediction_entries)
            
            # Top teams
            top_teams = sorted(team_strength.items(), key=lambda x: x[1], reverse=True)[:3]
            
            col1, col2, col3 = st.columns(3)
            
            for idx, (col, (team, strength)) in enumerate(zip([col1, col2, col3], top_teams)):
                with col:
                    emoji = ["🥇", "🥈", "🥉"][idx]
                    st.metric(f"{emoji} {team}", f"{strength:.1f}/100")
            
            st.markdown("---")
            
            # Team strength chart
            team_strength_fig = create_team_strength_chart(team_strength)
            st.plotly_chart(team_strength_fig, width='stretch')
            
            st.markdown("---")
            
            # Team breakdown table
            st.subheader("📋 Team Performance Breakdown")
            
            team_data = []
            for team, strength in sorted(team_strength.items(), key=lambda x: x[1], reverse=True):
                team_drivers = [e for e in prediction_entries if e.team == team]
                avg_pos = np.mean([e.predicted_position for e in team_drivers])
                best_pos = min([e.predicted_position for e in team_drivers])
                
                team_data.append({
                    'Team': team,
                    'Strength': f"{strength:.1f}",
                    'Avg Position': f"{avg_pos:.1f}",
                    'Best Position': best_pos,
                    'Drivers': len(team_drivers)
                })
            
            team_df = pd.DataFrame(team_data)
            st.dataframe(team_df, width='stretch')
    
    # =============================================================================
    # TAB 6: EXPORT & DOWNLOAD
    # =============================================================================
    
    with tab6:
        st.header("💾 Export & Download")
        
        st.markdown("""
        Download comprehensive race data in multiple formats for analysis,
        archival, or sharing.
        """)
        
        st.markdown("---")
        
        # CSV Exports
        st.subheader("📄 CSV Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction_entries:
                pred_df = build_prediction_table(
                    prediction_entries,
                    detect_dark_horses(prediction_entries),
                    check_extreme_gaps(prediction_entries)
                )
                csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Predictions CSV",
                    data=csv,
                    file_name=f"predictions_{race.year}_R{race.round}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if results:
                results_df = build_results_table(results)
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Results CSV",
                    data=csv,
                    file_name=f"results_{race.year}_R{race.round}.csv",
                    mime="text/csv"
                )
        
        if results and prediction_entries:
            comparison_df = build_comparison_table(prediction_entries, results)
            csv = comparison_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Comparison CSV",
                data=csv,
                file_name=f"comparison_{race.year}_R{race.round}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Excel Export
        st.subheader("📊 Excel Export (All Data)")
        
        if prediction_entries:
            excel_data = {}
            
            # Add sheets
            if prediction_entries:
                excel_data['Predictions'] = build_prediction_table(
                    prediction_entries,
                    detect_dark_horses(prediction_entries),
                    check_extreme_gaps(prediction_entries)
                )
            
            if results:
                excel_data['Results'] = build_results_table(results)
            
            if results and prediction_entries:
                excel_data['Comparison'] = build_comparison_table(prediction_entries, results)
            
            if excel_data:
                excel_bytes = export_to_excel(excel_data, "F1_Race_Data")
                st.download_button(
                    label="📥 Download Complete Excel Report",
                    data=excel_bytes,
                    file_name=f"F1_Race_{race.year}_R{race.round}_Complete.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        st.markdown("---")
        
        # Snapshot info
        st.subheader("📸 Prediction Snapshot Info")
        
        if prediction:
            snapshot_info = {
                'Race': f"{race.year} - Round {race.round}: {race.name}",
                'Circuit': race.circuit,
                'Date': race.event_date.strftime("%Y-%m-%d") if race.event_date else "N/A",
                'Snapshot Timestamp': prediction.snapshot_ts.strftime("%Y-%m-%d %H:%M UTC") if prediction.snapshot_ts else "N/A",
                'Freeze Policy': prediction.freeze_policy,
                'FastF1 Version': prediction.fastf1_version or "N/A",
                'Confidence Level': prediction.confidence_level,
                'Confidence Score': f"{prediction.confidence_score:.1f}/100",
                'Feature Coverage': f"{prediction.feature_coverage*100:.1f}%",
                'Features Imputed': prediction.num_imputed or 0
            }
            
            for key, value in snapshot_info.items():
                st.text(f"{key}: {value}")

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
