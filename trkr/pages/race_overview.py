"""
TRKR Race Overview Page
=======================
Live race dashboard with leaderboard, gaps, and key metrics.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trkr.utils import live, visuals, metrics
from app.database import SessionLocal
from app import models


def show():
    """Render Race Overview page."""
    
    st.title("📊 Race Overview")
    st.markdown("Live leaderboards, gaps, and race metrics")
    st.divider()
    
    # Race selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.selectbox("Season", [2024, 2025], index=1)
    
    with col2:
        races = live.get_available_races(year)
        if races:
            race_options = {f"R{r[0]} - {r[1]}": r for r in races}
            selected = st.selectbox("Race", list(race_options.keys()), index=len(races)-1)
            round_num, race_name = race_options[selected]
        else:
            st.warning("No races found for this year.")
            return
    
    with col3:
        session_type = st.selectbox("Session", ["Race", "Q", "FP3", "FP2", "FP1"])
    
    st.divider()
    
    # Load session data
    with st.spinner(f"Loading {session_type} data for {race_name}..."):
        session = live.load_session_data(year, round_num, session_type)
    
    if session is None:
        st.error(f"Could not load session data. Session may not be available yet.")
        return
    
    # ========== RACE INFO SECTION ==========
    with st.container(border=True):
        st.subheader("🏁 Race Information")
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.metric("Year", year)
        
        with info_col2:
            st.metric("Round", round_num)
        
        with info_col3:
            st.metric("Circuit", session.event['Location'] if hasattr(session, 'event') else "—")
        
        with info_col4:
            st.metric("Session", session_type)
    
    st.divider()
    
    # ========== LEADERBOARD SECTION ==========
    if session_type == "Race":
        st.subheader("🏁 Final Results")
        
        results = live.get_race_results(session)
        
        if not results.empty:
            # Format results for display
            display_results = results.copy()
            display_results['FinalPosition'] = display_results['FinalPosition'].fillna('DNF')
            
            # Metrics row
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                p1_driver = results.iloc[0]['Driver'] if len(results) > 0 else "—"
                st.metric("🥇 Winner", p1_driver)
            
            with col_p2:
                p2_driver = results.iloc[1]['Driver'] if len(results) > 1 else "—"
                st.metric("🥈 2nd Place", p2_driver)
            
            with col_p3:
                p3_driver = results.iloc[2]['Driver'] if len(results) > 2 else "—"
                st.metric("🥉 3rd Place", p3_driver)
            
            st.divider()
            
            # Results table
            st.dataframe(
                display_results.head(15),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Race results not yet available.")
    
    else:
        # Qualifying or Practice
        st.subheader(f"📋 {session_type} Session Leaderboard")
        
        results = live.get_qualifying_results(session) if session_type == "Q" else live.get_race_results(session)
        
        if not results.empty:
            st.dataframe(
                results.head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"{session_type} results not yet available.")
    
    st.divider()
    
    # ========== LAP DATA & GAPS ==========
    st.subheader("📈 Lap Analysis")
    
    lap_data = live.get_lap_data(session)
    
    if not lap_data.empty:
        # Sample drivers for visualization
        drivers = lap_data['Driver'].unique()[:10]
        
        lap_subset = lap_data[lap_data['Driver'].isin(drivers)]
        
        # Gap chart
        gap_fig = visuals.position_gap_chart(lap_subset, race_name)
        st.plotly_chart(gap_fig, use_container_width=True)
        
        st.caption(f"Showing lap analysis for {len(drivers)} drivers. Gap data may be unavailable for some sessions.")
    else:
        st.info("Lap-by-lap data not available for this session.")
    
    st.divider()
    
    # ========== INTEGRATION WITH PREDICTIONS (FROM DATABASE) ==========
    st.subheader("🤖 AI Predictions")
    
    db = SessionLocal()
    
    try:
        # Find race in database
        race = db.query(models.Race).filter(
            models.Race.year == year,
            models.Race.round == round_num
        ).first()
        
        if race:
            prediction = db.query(models.Prediction).filter(
                models.Prediction.race_id == race.id,
                models.Prediction.status == "frozen"
            ).order_by(models.Prediction.created_at.desc()).first()
            
            if prediction:
                # Show prediction confidence
                col_conf, col_cov = st.columns(2)
                
                with col_conf:
                    conf_fig = visuals.prediction_confidence_gauge(
                        prediction.confidence_score or 0,
                        prediction.confidence_level or "UNKNOWN"
                    )
                    st.plotly_chart(conf_fig, use_container_width=True)
                
                with col_cov:
                    coverage = prediction.feature_coverage or 0
                    st.metric(
                        "Feature Coverage",
                        f"{coverage * 100:.1f}%",
                        delta="Target: 85%" if coverage >= 0.85 else "Below Target"
                    )
                
                # Predicted leaderboard
                st.markdown("**Predicted Top 10:**")
                pred_entries = db.query(models.PredictionEntry).filter(
                    models.PredictionEntry.prediction_id == prediction.id
                ).order_by(models.PredictionEntry.predicted_position).limit(10).all()
                
                if pred_entries:
                    pred_df = pd.DataFrame([
                        {
                            'Position': e.predicted_position,
                            'Driver': e.driver,
                            'Gap': f"{e.gap:.1f}s" if e.gap > 0 else "—",
                            'Uncertainty': f"±{e.uncertainty:.1f}s" if e.uncertainty else "—"
                        }
                        for e in pred_entries
                    ])
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
            else:
                st.info("No AI predictions available for this race yet.")
        else:
            st.info("Race not found in prediction database.")
    finally:
        db.close()


if __name__ == "__main__":
    show()
