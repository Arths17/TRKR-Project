"""
Production-grade Streamlit dashboard for F1 Prediction Tracker.
Run with: streamlit run dashboard/dashboard.py
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from app import settings
from app.database import SessionLocal
from app import models

st.set_page_config(page_title="F1 Prediction Tracker", layout="wide", initial_sidebar_state="expanded")


@st.cache_resource
def get_engine():
    return create_engine(settings.DATABASE_URL, future=True)


def get_session() -> Session:
    return SessionLocal()


def load_races(db: Session):
    """Load all races from database, sorted by year/round descending."""
    races = db.query(models.Race).order_by(
        models.Race.year.desc(),
        models.Race.round.desc()
    ).all()
    return races


def load_prediction(db: Session, race_id: int):
    """Load frozen prediction for a race."""
    return (
        db.query(models.Prediction)
        .filter_by(race_id=race_id)
        .order_by(models.Prediction.created_at.desc())
        .first()
    )


def load_results(db: Session, race_id: int):
    """Load official race results."""
    return db.query(models.RaceResult).filter_by(race_id=race_id).all()


def load_metrics(db: Session, race_id: int):
    """Load evaluation metrics."""
    return db.query(models.EvaluationMetric).filter_by(race_id=race_id).all()


def format_confidence_score(level: str, score: float) -> str:
    """Format confidence as 'LEVEL (NN/100)' with color."""
    if level == "HIGH":
        return f"🟢 HIGH ({score:.0f}/100)"
    elif level == "MEDIUM":
        return f"🟡 MEDIUM ({score:.0f}/100)"
    else:
        return f"🔴 LOW ({score:.0f}/100)"


def format_coverage_pct(coverage: float) -> str:
    """Format coverage as percentage with assessment."""
    if coverage >= 85:
        return f"✅ {coverage:.1f}% (Excellent)"
    elif coverage >= 70:
        return f"⚠️ {coverage:.1f}% (Good)"
    else:
        return f"❌ {coverage:.1f}% (Below threshold)"


def race_has_results(db: Session, race_id: int) -> bool:
    """Check if race has official results stored."""
    return db.query(models.RaceResult).filter_by(race_id=race_id).count() > 0


def build_comparison_table(prediction, results):
    """Build prediction vs reality comparison DataFrame."""
    if not prediction or not results:
        return pd.DataFrame()
    
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
            
            # Determine note based on accuracy
            if delta == 0:
                note = "✅ Exact"
            elif abs(delta) <= 2:
                note = "✓ Close"
            elif abs(delta) <= 5:
                note = "~ Off"
            else:
                note = "❌ Large miss"
            
            comparison.append({
                "Driver": driver,
                "Predicted": pred_pos,
                "Actual": actual_pos,
                "Δ": f"{delta:+d}",
                "Note": note
            })
    
    return pd.DataFrame(comparison).sort_values("Actual")


def get_team_strength_index(prediction):
    """Extract and format team strength metrics from predictions."""
    if not prediction or not prediction.entries:
        return {}
    
    # Group drivers by team and calculate average predicted position
    teams = {}
    for entry in prediction.entries:
        team = entry.team or "Unknown"
        if team not in teams:
            teams[team] = []
        teams[team].append(entry.predicted_position)
    
    # Convert to strength index (0-100 scale)
    team_strength = {}
    for team, positions in teams.items():
        avg_pos = sum(positions) / len(positions)
        # Formula: strength = 100 * (21 - avg_position) / 20
        strength = max(0, min(100, 100 * (21 - avg_pos) / 20))
        team_strength[team] = strength
    
    return dict(sorted(team_strength.items(), key=lambda x: x[1], reverse=True))


def check_extreme_gaps(prediction, threshold=120):
    """Check for extreme gaps (potential DNFs or data issues)."""
    if not prediction or not prediction.entries:
        return []
    
    extreme = []
    for entry in prediction.entries:
        if entry.gap > threshold:
            extreme.append({
                "driver": entry.driver,
                "team": entry.team,
                "gap": entry.gap,
                "position": entry.predicted_position
            })
    
    return extreme


def main():
    st.set_page_config(page_title="F1 Prediction Tracker", layout="wide")
    
    # Header
    st.title("🏎️ F1 Prediction Tracker")
    st.markdown("**AI-Powered Prediction Tracking & Accuracy Evaluation**")
    st.divider()
    
    db = get_session()
    races = load_races(db)
    
    if not races:
        st.warning("⚠️ No races synced yet. Run: `curl -X POST 'http://localhost:8000/predict?year=2024&race=Qatar'`")
        return
    
    # Race selector
    race_options = {
        f"{r.year} - Round {r.round or '?'} - {r.name}": r 
        for r in races
    }
    race_label = st.selectbox("📅 Select Race", list(race_options.keys()))
    race = race_options[race_label]
    
    # ========== RACE METADATA SECTION ==========
    with st.container(border=True):
        st.subheader("Race Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Year", race.year)
        
        with col2:
            round_display = f"Round {race.round}" if race.round else "N/A"
            st.metric("Round", round_display)
        
        with col3:
            circuit_display = race.circuit if race.circuit else "⚠️ Not Available"
            st.metric("Circuit", circuit_display)
        
        with col4:
            if race.event_date:
                date_display = race.event_date.strftime("%Y-%m-%d")
            else:
                date_display = "N/A"
            st.metric("Date", date_display)
        
        # Data quality warning if metadata incomplete
        if not race.circuit or race.round is None:
            st.warning("⚠️ **Data Quality Notice:** Race metadata is incomplete. Some information may be unavailable.")
    
    st.divider()
    
    # Load data
    prediction = load_prediction(db, race.id)
    results = load_results(db, race.id)
    metrics = load_metrics(db, race.id)
    has_results = race_has_results(db, race.id)
    
    # ========== PREDICTIONS SECTION ==========
    if prediction:
        with st.container(border=True):
            # Header with freeze metadata
            col_title, col_meta = st.columns([3, 1])
            with col_title:
                st.subheader(f"🔒 Predictions (Frozen)")
            with col_meta:
                st.caption(f"Snapshot: {prediction.snapshot_ts.strftime('%Y-%m-%d %H:%M UTC')}")
            
            # Confidence & Coverage badges
            col_conf, col_cov, col_policy = st.columns(3)
            
            with col_conf:
                if prediction.confidence_level and prediction.confidence_score is not None:
                    conf_str = format_confidence_score(prediction.confidence_level, prediction.confidence_score)
                    st.markdown(f"**Confidence:** {conf_str}")
                else:
                    st.markdown("**Confidence:** ⚠️ Unknown")
            
            with col_cov:
                if prediction.feature_coverage is not None:
                    cov_str = format_coverage_pct(prediction.feature_coverage)
                    st.markdown(f"**Feature Coverage:** {cov_str}")
                else:
                    st.markdown("**Feature Coverage:** ⚠️ Unknown")
            
            with col_policy:
                policy_str = prediction.freeze_policy.replace("_", " ").title()
                st.markdown(f"**Freeze Policy:** {policy_str}")
            
            # Prediction table
            df_pred = pd.DataFrame([
                {
                    "Position": e.predicted_position,
                    "Driver": e.driver,
                    "Team": e.team or "—",
                    "Time (s)": f"{e.predicted_race_time:.2f}",
                    "Gap": f"{e.gap:.2f}s" if e.gap > 0 else "—",
                    "Uncertainty": f"±{e.uncertainty:.1f}s" if e.uncertainty else "—"
                }
                for e in prediction.entries
            ]).sort_values("Position")
            
            st.dataframe(df_pred, use_container_width=True, hide_index=True)
            
            # Confidence explanation (if LOW)
            if prediction.confidence_level == "LOW":
                with st.expander("ℹ️ Why is confidence LOW?"):
                    st.markdown(f"""
                    - **Feature Coverage:** {prediction.feature_coverage:.1f}% (target: 85%+)
                    - **Imputed Features:** {prediction.num_imputed or 0} (target: <3)
                    - **Interpretation:** Predictions may be less reliable due to limited pre-race data.
                    """)
            
            # Team Strength Index visualization
            with st.expander("📊 Team Strength Index (0-100)"):
                team_strength = get_team_strength_index(prediction)
                if team_strength:
                    for team, strength in list(team_strength.items())[:5]:
                        # Determine color based on strength
                        if strength >= 80:
                            color = "🟢"
                        elif strength >= 60:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        # Bar visualization
                        bar_length = int(strength / 5)
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        st.markdown(f"{color} **{team}** │ {bar} │ {strength:.1f}/100")
            
            # Extreme gap warnings
            extreme_gaps = check_extreme_gaps(prediction, threshold=120)
            if extreme_gaps:
                st.warning(f"⚠️ **Extreme Gap Alert:** {len(extreme_gaps)} driver(s) with gap >120s")
                for gap_data in extreme_gaps:
                    st.markdown(
                        f"  • **{gap_data['driver']}** ({gap_data['team']}) - "
                        f"P{gap_data['position']}: {gap_data['gap']:.1f}s gap (potential DNF or data issue)"
                    )
    else:
        st.info("ℹ️ No predictions stored for this race. Run: `curl -X POST 'http://localhost:8000/predict?year=2024&race=Qatar'`")
    
    st.divider()
    
    # ========== RACE STATE HANDLING ==========
    if not has_results:
        # Pre-race state
        st.info("📋 **Race Status:** Not yet completed. Official results will appear here after the race concludes.")
    else:
        # Post-race state
        st.success("✅ **Race Status:** Completed. Official results and accuracy metrics are available below.")
        
        # ========== OFFICIAL RESULTS SECTION ==========
        with st.container(border=True):
            st.subheader("🏁 Official Race Results")
            
            df_res = pd.DataFrame([
                {
                    "Position": r.position or "—",
                    "Driver": r.driver,
                    "Team": r.team or "—",
                    "Status": r.status or "Finished",
                    "Points": int(r.points) if r.points else 0,
                    "Time": f"{r.time:.2f}s" if r.time else "—"
                }
                for r in results
            ]).sort_values("Position", key=lambda x: pd.to_numeric(x, errors='coerce'))
            
            st.dataframe(df_res, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # ========== PREDICTION VS REALITY COMPARISON ==========
        with st.container(border=True):
            st.subheader("📊 Prediction vs Reality Comparison")
            
            comparison_df = build_comparison_table(prediction, results)
            
            if not comparison_df.empty:
                # Summary stats
                col_correct, col_close, col_miss = st.columns(3)
                
                exact_count = (comparison_df["Δ"] == "0").sum()
                close_count = (comparison_df["Note"] == "✓ Close").sum()
                miss_count = (comparison_df["Note"] == "❌ Large miss").sum()
                
                with col_correct:
                    st.metric("Exact Predictions", exact_count)
                
                with col_close:
                    st.metric("Close (±1-2 pos)", close_count)
                
                with col_miss:
                    st.metric("Large Misses (>±5)", miss_count)
                
                st.divider()
                
                # Comparison table with highlighting
                def highlight_comparison(row):
                    if row["Note"] == "❌ Large miss":
                        return ["background-color: #ffcccc"] * len(row)
                    elif row["Note"] == "✅ Exact":
                        return ["background-color: #ccffcc"] * len(row)
                    elif row["Note"] == "✓ Close":
                        return ["background-color: #ffffcc"] * len(row)
                    else:
                        return ["background-color: #f0f0f0"] * len(row)
                
                st.dataframe(
                    comparison_df.style.apply(highlight_comparison, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Legend
                st.caption(
                    "🟢 Green: Exact match | 🟡 Yellow: Close (±1-2) | 🟠 Orange: Off (±3-5) | 🔴 Red: Large miss (>±5)"
                )
            else:
                st.warning("⚠️ No comparable results. Ensure both predictions and results are available.")
        
        st.divider()
        
        # ========== ACCURACY METRICS SECTION ==========
        if metrics:
            with st.container(border=True):
                st.subheader("📈 Accuracy Metrics & Model Performance")
                
                metric = metrics[-1]  # Latest metric
                
                col_mae, col_time, col_winner, col_podium = st.columns(4)
                
                with col_mae:
                    mae_value = f"{metric.position_mae:.2f}" if metric.position_mae else "—"
                    st.metric(
                        "Position MAE",
                        mae_value,
                        help="Mean Absolute Error in positions (lower = better, 0 = perfect)"
                    )
                
                with col_time:
                    time_value = f"{metric.time_mae_seconds:.1f}s" if metric.time_mae_seconds else "—"
                    st.metric(
                        "Time MAE",
                        time_value,
                        help="Mean Absolute Error in race time seconds"
                    )
                
                with col_winner:
                    winner_icon = "✅ Correct" if metric.winner_correct else "❌ Incorrect"
                    st.metric("Winner Prediction", winner_icon)
                
                with col_podium:
                    if metric.podium_accuracy is not None:
                        podium_pct = f"{metric.podium_accuracy * 100:.0f}%"
                        podium_delta = f"{metric.podium_accuracy * 3:.0f}/3 correct"
                    else:
                        podium_pct = "—"
                        podium_delta = "N/A"
                    st.metric("Podium Accuracy", podium_pct, delta=podium_delta)
                
                st.divider()
                
                # Detailed analysis
                with st.expander("📋 Detailed Breakdown"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Prediction Accuracy**")
                        st.write(f"""
                        - Position error: ±{metric.position_mae:.2f} places on average
                        - Time error: {metric.time_mae_seconds:.1f} seconds on average
                        - Winner picked: {'Yes ✅' if metric.winner_correct else 'No ❌'}
                        - Podium hits: {int(metric.podium_accuracy * 3) if metric.podium_accuracy else 0}/3 drivers
                        """)
                    
                    with col2:
                        st.markdown("**Model Reliability**")
                        st.write(f"""
                        - Confidence score: {prediction.confidence_score:.0f}/100
                        - Feature coverage: {prediction.feature_coverage:.1f}%
                        - Assessment: {prediction.confidence_level}
                        - Last updated: {metric.created_at.strftime('%Y-%m-%d %H:%M UTC')}
                        """)
        else:
            st.info("ℹ️ Metrics will be available shortly after the race completes and results are synced.")


if __name__ == "__main__":
    main()

