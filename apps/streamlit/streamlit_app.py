"""
Production-Grade F1 Prediction Tracker Web App
Streamlit application for AI-powered Formula 1 race predictions
Built with FastF1, XGBoost, and SQLAlchemy
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sqlalchemy import create_engine
from io import StringIO
import requests
import json

from app import settings
from app.database import SessionLocal
from app import models
from data_fetcher import F1DataFetcher
from data_processor import F1DataProcessor
from predictor import F1Predictor

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="F1 Prediction Tracker",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #FF6B6B;
        }
        .success-card { border-left-color: #4CAF50; }
        .warning-card { border-left-color: #FFC107; }
        .error-card { border-left-color: #F44336; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE & CACHING
# ============================================================================

@st.cache_resource
def get_db_session():
    """Get database session."""
    return SessionLocal()

@st.cache_data(ttl=3600)
def load_all_races():
    """Load all races from database."""
    db = get_db_session()
    races = db.query(models.Race).order_by(
        models.Race.year.desc(),
        models.Race.round.desc()
    ).all()
    return races

@st.cache_data(ttl=300)
def fetch_fastf1_schedule(year: int):
    """Fetch season schedule from FastF1."""
    try:
        import fastf1
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        st.error(f"Failed to fetch schedule: {e}")
        return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_confidence(level: str, score: float) -> str:
    """Format confidence as badge."""
    if level == "HIGH":
        return f"🟢 HIGH ({score:.0f}/100)"
    elif level == "MEDIUM":
        return f"🟡 MEDIUM ({score:.0f}/100)"
    else:
        return f"🔴 LOW ({score:.0f}/100)"

def format_coverage(coverage: float) -> str:
    """Format coverage with assessment."""
    if coverage >= 85:
        return f"✅ {coverage:.1f}% (Excellent)"
    elif coverage >= 70:
        return f"⚠️ {coverage:.1f}% (Good)"
    else:
        return f"❌ {coverage:.1f}% (Below Target)"

def get_circuit_info(df_schedule, round_num: int) -> dict:
    """Extract circuit info from schedule."""
    try:
        race = df_schedule[df_schedule['RoundNumber'] == round_num].iloc[0]
        return {
            'circuit': race['Location'],
            'date': race['EventDate'],
            'laps': race.get('Laps', 'N/A'),
            'distance': race.get('Circuit Length', 'N/A')
        }
    except:
        return {'circuit': 'N/A', 'date': 'N/A', 'laps': 'N/A', 'distance': 'N/A'}

def build_team_strength_chart(prediction):
    """Build team strength visualization."""
    if not prediction or not prediction.entries:
        return None
    
    teams = {}
    for entry in prediction.entries:
        team = entry.team or "Unknown"
        if team not in teams:
            teams[team] = []
        teams[team].append(entry.predicted_position)
    
    team_strength = {}
    for team, positions in teams.items():
        avg_pos = sum(positions) / len(positions)
        strength = max(0, min(100, 100 * (21 - avg_pos) / 20))
        team_strength[team] = strength
    
    team_strength = dict(sorted(team_strength.items(), key=lambda x: x[1], reverse=True))
    
    # Create bar chart
    df_teams = pd.DataFrame(list(team_strength.items()), columns=['Team', 'Strength'])
    
    fig = px.bar(
        df_teams,
        x='Strength',
        y='Team',
        orientation='h',
        color='Strength',
        color_continuous_scale='RdYlGn',
        range_color=[0, 100],
        title='Team Strength Index (0-100)',
        labels={'Strength': 'Strength Index'},
        height=400
    )
    fig.update_yaxes(autorange="reversed")
    return fig

def build_prediction_table(prediction):
    """Build formatted prediction table."""
    if not prediction:
        return None
    
    data = []
    for entry in prediction.entries:
        data.append({
            'Position': entry.predicted_position,
            'Driver': entry.driver,
            'Team': entry.team or '—',
            'Time (s)': f"{entry.predicted_race_time:.2f}",
            'Gap (s)': f"{entry.gap:.2f}" if entry.gap > 0 else '—',
            'Δ (±s)': f"±{entry.uncertainty:.1f}" if entry.uncertainty else '—'
        })
    
    return pd.DataFrame(data).sort_values('Position')

def build_comparison_table(prediction, results):
    """Build prediction vs reality comparison."""
    if not prediction or not results:
        return None
    
    pred_dict = {e.driver: e for e in prediction.entries}
    result_dict = {r.driver: r for r in results}
    
    comparison = []
    for driver in pred_dict.keys():
        pred_entry = pred_dict[driver]
        result_entry = result_dict.get(driver)
        
        if result_entry and result_entry.position is not None:
            actual_pos = int(result_entry.position)
            pred_pos = int(pred_entry.predicted_position)
            delta = pred_pos - actual_pos
            
            if delta == 0:
                note = "✅ Exact"
            elif abs(delta) <= 2:
                note = "✓ Close"
            elif abs(delta) <= 5:
                note = "~ Off"
            else:
                note = "❌ Miss"
            
            comparison.append({
                'Driver': driver,
                'Predicted': pred_pos,
                'Actual': actual_pos,
                'Δ': delta,
                'Note': note
            })
    
    return pd.DataFrame(comparison).sort_values('Actual') if comparison else None

def export_to_csv(df: pd.DataFrame, filename: str) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')

def check_data_quality(prediction) -> list:
    """Check for data quality issues."""
    warnings = []
    
    if prediction:
        if prediction.feature_coverage and prediction.feature_coverage < 70:
            warnings.append(f"⚠️ Low feature coverage: {prediction.feature_coverage:.1f}%")
        
        if prediction.confidence_level == "LOW":
            warnings.append(f"⚠️ Prediction confidence is LOW ({prediction.confidence_score:.0f}/100)")
        
        # Check for extreme gaps
        for entry in prediction.entries:
            if entry.gap > 120:
                warnings.append(f"⚠️ Extreme gap: {entry.driver} at {entry.gap:.1f}s (potential DNF)")
    
    return warnings

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("# 🏎️ F1 Prediction Tracker")
    st.markdown("**AI-Powered Race Predictions | FastF1 + XGBoost | Production Ready**")
    st.divider()
    
    # Database connection
    db = get_db_session()
    races = load_all_races()
    
    if not races:
        st.error("❌ No races found in database. Please sync predictions first.")
        st.info("Run: `curl -X POST 'http://localhost:8000/predict?year=2024&race=Qatar'`")
        return
    
    # ========================================================================
    # SIDEBAR: RACE SELECTION & SETTINGS
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Race selector
        race_options = {
            f"{r.year} - R{r.round or '?'} - {r.name}": r 
            for r in races
        }
        race_label = st.selectbox("📅 Select Race", list(race_options.keys()))
        selected_race = race_options[race_label]
        
        # Load race-specific data
        prediction = db.query(models.Prediction).filter_by(race_id=selected_race.id).order_by(
            models.Prediction.created_at.desc()
        ).first()
        
        results = db.query(models.RaceResult).filter_by(race_id=selected_race.id).all()
        metrics = db.query(models.EvaluationMetric).filter_by(race_id=selected_race.id).all()
        
        has_results = len(results) > 0
        has_metrics = len(metrics) > 0
        
        st.subheader("📊 Status")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predictions", "✅" if prediction else "❌")
        with col2:
            st.metric("Results", "✅" if has_results else "⏳")
        with col3:
            st.metric("Metrics", "✅" if has_metrics else "⏳")
        
        # Data quality warnings
        st.subheader("⚠️ Data Quality")
        warnings = check_data_quality(prediction)
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("✅ Data quality checks passed")
        
        st.divider()
        
        # Export options
        st.subheader("📥 Export")
        if prediction:
            pred_df = build_prediction_table(prediction)
            if pred_df is not None:
                csv_pred = export_to_csv(pred_df, "predictions.csv")
                st.download_button(
                    label="📥 Predictions CSV",
                    data=csv_pred,
                    file_name=f"predictions_{selected_race.year}_{selected_race.name}.csv",
                    mime="text/csv"
                )
        
        if has_results and prediction:
            comp_df = build_comparison_table(prediction, results)
            if comp_df is not None:
                csv_comp = export_to_csv(comp_df, "comparison.csv")
                st.download_button(
                    label="📥 Comparison CSV",
                    data=csv_comp,
                    file_name=f"comparison_{selected_race.year}_{selected_race.name}.csv",
                    mime="text/csv"
                )
    
    # ========================================================================
    # MAIN CONTENT: TABS
    # ========================================================================
    
    tab_info, tab_predictions, tab_results, tab_analysis, tab_circuit = st.tabs([
        "ℹ️ Race Info",
        "🔮 Predictions",
        "🏁 Results",
        "📊 Analysis",
        "🏎️ Circuit"
    ])
    
    # ========================================================================
    # TAB 1: RACE INFORMATION
    # ========================================================================
    
    with tab_info:
        st.subheader("Race Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Year", selected_race.year)
        with col2:
            st.metric("Round", f"R{selected_race.round}" if selected_race.round else "N/A")
        with col3:
            st.metric("Name", selected_race.name)
        with col4:
            date_str = selected_race.event_date.strftime("%Y-%m-%d") if selected_race.event_date else "N/A"
            st.metric("Date", date_str)
        
        st.divider()
        
        # Prediction metadata
        if prediction:
            st.subheader("🔒 Prediction Snapshot")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                conf_str = format_confidence(prediction.confidence_level or "UNKNOWN", prediction.confidence_score or 0)
                st.metric("Confidence", conf_str)
            with col2:
                cov_str = format_coverage(prediction.feature_coverage or 0)
                st.metric("Coverage", cov_str)
            with col3:
                policy = prediction.freeze_policy.replace("_", " ").title()
                st.metric("Freeze Policy", policy)
            
            st.caption(f"📸 Snapshot: {prediction.snapshot_ts.strftime('%Y-%m-%d %H:%M UTC')}")
            
            if prediction.confidence_level == "LOW":
                with st.expander("ℹ️ Why is confidence LOW?", expanded=False):
                    st.markdown(f"""
                    **Feature Coverage:** {prediction.feature_coverage:.1f}% (target: 85%+)
                    
                    **Imputed Features:** {prediction.num_imputed or 0} (target: <3)
                    
                    **Implication:** Predictions may be less reliable due to limited pre-race data. 
                    Consider treating results as probabilistic guidance rather than certainties.
                    """)
    
    # ========================================================================
    # TAB 2: PREDICTIONS
    # ========================================================================
    
    with tab_predictions:
        st.subheader("🔮 Predicted Race Outcome")
        
        if not prediction:
            st.info("ℹ️ No predictions available for this race.")
        else:
            # Prediction table
            pred_df = build_prediction_table(prediction)
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Team strength visualization
            with st.expander("📊 Team Strength Index", expanded=True):
                fig_teams = build_team_strength_chart(prediction)
                if fig_teams:
                    st.plotly_chart(fig_teams, use_container_width=True)
            
            st.divider()
            
            # Insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Winner", prediction.entries[0].driver if prediction.entries else "N/A")
            
            with col2:
                podium = [e.driver for e in prediction.entries[:3]]
                st.metric("Predicted Podium", " → ".join(podium))
            
            with col3:
                dark_horses = [e for e in prediction.entries[3:8] if e.gap < 15]
                dark_horse_text = ", ".join([dh.driver for dh in dark_horses]) if dark_horses else "None"
                st.metric("Dark Horses", dark_horse_text if dark_horse_text else "None")
    
    # ========================================================================
    # TAB 3: OFFICIAL RESULTS
    # ========================================================================
    
    with tab_results:
        st.subheader("🏁 Official Race Results")
        
        if not has_results:
            st.info("⏳ Race results not yet available. Check back after the race concludes.")
        else:
            results_data = []
            for r in results:
                results_data.append({
                    'Position': r.position or '—',
                    'Driver': r.driver,
                    'Team': r.team or '—',
                    'Status': r.status or 'Finished',
                    'Points': int(r.points) if r.points else 0,
                    'Time': f"{r.time:.2f}s" if r.time else '—'
                })
            
            df_results = pd.DataFrame(results_data).sort_values('Position', 
                                                                 key=lambda x: pd.to_numeric(x, errors='coerce'))
            st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # TAB 4: ANALYSIS & METRICS
    # ========================================================================
    
    with tab_analysis:
        st.subheader("📊 Prediction vs Reality Analysis")
        
        if not (has_results and prediction):
            st.info("⏳ Awaiting race results for comparison.")
        else:
            # Comparison table
            comp_df = build_comparison_table(prediction, results)
            
            if comp_df is not None:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                exact_count = (comp_df['Δ'] == 0).sum()
                close_count = ((comp_df['Δ'].abs() > 0) & (comp_df['Δ'].abs() <= 2)).sum()
                off_count = ((comp_df['Δ'].abs() > 2) & (comp_df['Δ'].abs() <= 5)).sum()
                miss_count = (comp_df['Δ'].abs() > 5).sum()
                
                with col1:
                    st.metric("Exact", exact_count, delta=f"{exact_count}/{len(comp_df)}")
                with col2:
                    st.metric("Close (±1-2)", close_count)
                with col3:
                    st.metric("Off (±3-5)", off_count)
                with col4:
                    st.metric("Miss (>±5)", miss_count)
                
                st.divider()
                
                # Comparison table with styling
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Accuracy metrics
            if has_metrics:
                st.subheader("📈 Accuracy Metrics")
                metric = metrics[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mae_val = f"{metric.position_mae:.2f}" if metric.position_mae else "—"
                    st.metric("Position MAE", mae_val, help="Lower is better")
                
                with col2:
                    time_val = f"{metric.time_mae_seconds:.1f}s" if metric.time_mae_seconds else "—"
                    st.metric("Time MAE", time_val, help="Average time error")
                
                with col3:
                    winner = "✅ Correct" if metric.winner_correct else "❌ Incorrect"
                    st.metric("Winner", winner)
                
                with col4:
                    podium_pct = f"{metric.podium_accuracy * 100:.0f}%" if metric.podium_accuracy else "—"
                    st.metric("Podium Accuracy", podium_pct)
                
                st.caption(f"Computed: {metric.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # ========================================================================
    # TAB 5: CIRCUIT INFORMATION
    # ========================================================================
    
    with tab_circuit:
        st.subheader("🏎️ Circuit Information")
        
        schedule = fetch_fastf1_schedule(selected_race.year)
        
        if schedule is not None:
            circuit_info = get_circuit_info(schedule, selected_race.round or 1)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Circuit", circuit_info['circuit'])
            with col2:
                date_str = circuit_info['date'].strftime("%Y-%m-%d") if circuit_info['date'] != 'N/A' else "N/A"
                st.metric("Date", date_str)
            with col3:
                st.metric("Laps", circuit_info['laps'])
            with col4:
                st.metric("Distance", circuit_info['distance'])
        else:
            st.warning("⚠️ Could not fetch circuit information.")
        
        st.divider()
        
        st.info("""
        📍 **Circuit Highlights**
        
        - Track layout affects tire strategies and pit stop timing
        - Historical performance data influences model predictions
        - Weather conditions impact lap times and gaps
        """)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
