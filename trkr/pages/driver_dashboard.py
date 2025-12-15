"""
TRKR Driver Dashboard Page
==========================
Individual driver performance, telemetry, and history.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from trkr.utils import live, visuals


def show():
    """Render Driver Dashboard page."""
    
    st.title("🏁 Driver Dashboard")
    st.markdown("Driver performance, telemetry, and statistics")
    st.divider()
    
    # Season and driver selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.selectbox("Season", [2024, 2025], index=1, key="driver_year")
    
    with col2:
        races = live.get_available_races(year)
        if races:
            race_options = {f"R{r[0]} - {r[1]}": r for r in races}
            selected = st.selectbox("Race", list(race_options.keys()), index=len(races)-1, key="driver_race")
            round_num, race_name = race_options[selected]
        else:
            st.warning("No races found for this year.")
            return
    
    with col3:
        session_type = st.selectbox("Session", ["Race", "Q", "FP3"], key="driver_session")
    
    st.divider()
    
    # Load session
    with st.spinner(f"Loading {session_type} data..."):
        session = live.load_session_data(year, round_num, session_type)
    
    if session is None:
        st.error("Could not load session data.")
        return
    
    # Get drivers in session
    results = live.get_race_results(session)
    if results.empty:
        st.warning("No driver data available for this session.")
        return
    
    # Driver selector
    drivers = sorted(results['Driver'].unique())
    selected_driver = st.selectbox("Select Driver", drivers)
    
    st.divider()
    
    # ========== DRIVER INFO ==========
    driver_info = results[results['Driver'] == selected_driver].iloc[0] if not results.empty else None
    
    if driver_info is not None:
        with st.container(border=True):
            st.subheader(f"👤 {selected_driver}")
            
            col_name, col_team, col_pos, col_points = st.columns(4)
            
            with col_name:
                st.metric("Full Name", driver_info.get('DriverName', '—'))
            
            with col_team:
                st.metric("Team", driver_info.get('Team', '—'))
            
            with col_pos:
                if session_type == "Race":
                    pos = driver_info.get('FinalPosition', '—')
                    st.metric("Final Position", pos)
                else:
                    st.metric("Grid Position", driver_info.get('GridPos', '—'))
            
            with col_points:
                points = driver_info.get('Points', 0)
                st.metric("Points", int(points) if points else 0)
    
    st.divider()
    
    # ========== TELEMETRY ==========
    if session_type == "Race":
        st.subheader("📊 Telemetry — Best Lap")
        
        with st.spinner(f"Loading telemetry for {selected_driver}..."):
            telemetry_df, best_lap = live.get_driver_telemetry(session, selected_driver)
        
        if not telemetry_df.empty:
            telemetry_fig = visuals.driver_telemetry_chart(telemetry_df, selected_driver)
            st.plotly_chart(telemetry_fig, use_container_width=True)
            
            # Best lap info
            if best_lap is not None:
                col_time, col_lap = st.columns(2)
                with col_time:
                    st.metric("Best Lap Time", f"{best_lap['LapTime']}")
                with col_lap:
                    st.metric("Lap Number", int(best_lap['LapNumber']))
        else:
            st.info(f"Telemetry data not available for {selected_driver}.")
    
    st.divider()
    
    # ========== LAP HISTORY ==========
    st.subheader("📈 Lap-by-Lap Performance")
    
    lap_data = live.get_lap_data(session)
    
    if not lap_data.empty:
        driver_laps = lap_data[lap_data['Driver'] == selected_driver].sort_values('LapNumber')
        
        if not driver_laps.empty:
            # Select only available columns
            display_cols = ['LapNumber', 'LapTime', 'IsAccurate']
            if 'Position' in driver_laps.columns:
                display_cols.append('Position')
            
            st.dataframe(
                driver_laps[display_cols].head(50),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption(f"Showing up to 50 laps. Total laps completed: {len(driver_laps)}")
        else:
            st.info(f"No lap data for {selected_driver}.")
    else:
        st.info("Lap-by-lap data not available.")
    
    st.divider()
    
    # ========== COMPARISON WITH TEAMMATES ==========
    st.subheader("🏆 Comparison with Teammates")
    
    if driver_info is not None and 'Team' in driver_info:
        team = driver_info['Team']
        teammates = results[results['Team'] == team]
        
        if len(teammates) > 1:
            compare_metrics = []
            for _, teammate in teammates.iterrows():
                compare_metrics.append({
                    'Driver': teammate['Driver'],
                    'Team': teammate['Team'],
                    'Position': teammate.get('FinalPosition', '—'),
                    'Points': int(teammate.get('Points', 0)) if teammate.get('Points') else 0
                })
            
            compare_df = pd.DataFrame(compare_metrics)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
        else:
            st.info(f"{selected_driver} has no visible teammates in this session.")


if __name__ == "__main__":
    show()
