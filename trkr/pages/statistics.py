"""
TRKR Statistics Page
====================
Historical standings, trends, and season-long analysis.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from trkr.utils import live, visuals, metrics

# Import database and models (will work once app is initialized)
try:
    from app.database import SessionLocal
    from app import models
except ImportError:
    st.error("Database modules not available. Ensure app is properly initialized.")
    SessionLocal = None
    models = None


def show():
    """Render Statistics page."""
    
    st.title("📈 Statistics")
    st.markdown("Season standings, historical trends, and comparative analysis")
    st.divider()
    
    # Year selector
    year = st.selectbox("Season", [2024, 2025], index=1, key="stats_year")
    
    st.divider()
    
    # ========== STANDINGS ==========
    st.subheader("🏎️ Championship Standings")
    
    races = live.get_available_races(year)
    
    if not races:
        st.warning("No races data available for this season.")
        return
    
    # Build standings from all races
    all_drivers = {}
    
    for round_num, race_name in races:
        try:
            session = live.load_session_data(year, round_num, "Race")
            if session is None:
                continue
            
            results = live.get_race_results(session)
            
            for _, result in results.iterrows():
                driver = result['Driver']
                points = int(result.get('Points', 0)) if result.get('Points') else 0
                
                if driver not in all_drivers:
                    all_drivers[driver] = {
                        'Driver': driver,
                        'Races': 0,
                        'Points': 0,
                        'Wins': 0,
                        'Podiums': 0
                    }
                
                all_drivers[driver]['Points'] += points
                all_drivers[driver]['Races'] += 1
                
                pos = result.get('FinalPosition')
                if pos == 1:
                    all_drivers[driver]['Wins'] += 1
                if pos in [1, 2, 3]:
                    all_drivers[driver]['Podiums'] += 1
        
        except Exception as e:
            st.warning(f"Could not load race {race_name}: {e}")
            continue
    
    if all_drivers:
        standings_df = pd.DataFrame(list(all_drivers.values())).sort_values('Points', ascending=False)
        standings_df['Position'] = range(1, len(standings_df) + 1)
        
        display_cols = ['Position', 'Driver', 'Races', 'Points', 'Wins', 'Podiums']
        st.dataframe(
            standings_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Standings chart
        standings_chart = visuals.leaderboard_chart(standings_df.head(10), "Top 10 Drivers by Points")
        st.plotly_chart(standings_chart, use_container_width=True)
    else:
        st.info("No race data available yet for this season.")
    
    st.divider()
    
    # ========== PREDICTION ACCURACY ACROSS RACES ==========
    st.subheader("🤖 Prediction Performance Across Races")
    
    db = SessionLocal()
    
    try:
        # Get all races with predictions
        races_with_preds = db.query(models.Race).filter(
            models.Race.year == year
        ).order_by(models.Race.round).all()
        
        perf_data = []
        
        for race in races_with_preds:
            predictions = db.query(models.Prediction).filter(
                models.Prediction.race_id == race.id,
                models.Prediction.status == "frozen"
            ).order_by(models.Prediction.created_at.desc()).first()
            
            if predictions is None:
                continue
            
            eval_metrics = db.query(models.EvaluationMetric).filter(
                models.EvaluationMetric.prediction_id == predictions.id
            ).first()
            
            if eval_metrics is not None:
                perf_data.append({
                    'Race': race.name,
                    'Round': race.round or '?',
                    'MAE': f"{eval_metrics.position_mae:.2f}" if eval_metrics.position_mae else "—",
                    'Winner': "✅" if eval_metrics.winner_correct else "❌",
                    'Podium': f"{eval_metrics.podium_accuracy * 100:.0f}%" if eval_metrics.podium_accuracy else "—",
                    'Confidence': predictions.confidence_level or "—"
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
        else:
            st.info("No prediction data available yet.")
    
    finally:
        db.close()
    
    st.divider()
    
    # ========== TEAM PERFORMANCE ==========
    st.subheader("🏆 Team Performance")
    
    if all_drivers:
        # Aggregate by team (would need team data from FastF1)
        team_stats = {}
        
        try:
            session = live.load_session_data(year, races[0][0], "Race")
            if session is not None:
                results = live.get_race_results(session)
                
                for _, driver_row in results.iterrows():
                    team = driver_row.get('Team', 'Unknown')
                    
                    driver_stats = all_drivers.get(driver_row['Driver'], {})
                    
                    if team not in team_stats:
                        team_stats[team] = {
                            'Team': team,
                            'Drivers': 0,
                            'TotalPoints': 0,
                            'AvgPoints': 0
                        }
                    
                    team_stats[team]['Drivers'] += 1
                    team_stats[team]['TotalPoints'] += driver_stats.get('Points', 0)
                
                # Calculate averages
                for team in team_stats.values():
                    team['AvgPoints'] = team['TotalPoints'] / team['Drivers'] if team['Drivers'] > 0 else 0
                
                team_df = pd.DataFrame(list(team_stats.values())).sort_values('TotalPoints', ascending=False)
                
                st.dataframe(
                    team_df[['Team', 'Drivers', 'TotalPoints', 'AvgPoints']],
                    use_container_width=True,
                    hide_index=True
                )
        
        except Exception as e:
            st.warning(f"Could not load team data: {e}")


if __name__ == "__main__":
    show()
