"""
TRKR Live Data Utilities
========================
Fetch and process live F1 data from FastF1 with caching.
"""

import fastf1
import pandas as pd
from typing import Optional, Tuple
import streamlit as st


@st.cache_resource
def load_season_schedule(year: int) -> pd.DataFrame:
    """
    Load F1 season schedule from FastF1.
    
    Args:
        year: Season year
    
    Returns:
        DataFrame with races
    """
    try:
        fastf1.Cache.enable_cache('cache')
        schedule = fastf1.get_event_schedule(year)
        return schedule
    except Exception as e:
        st.error(f"Failed to load schedule for {year}: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_session_data(year: int, round_num: int, session_type: str = "Race"):
    """
    Load session data (practice, qualifying, race) from FastF1.
    
    Args:
        year: Season year
        round_num: Race round number
        session_type: "FP1", "FP2", "FP3", "Q", or "Race"
    
    Returns:
        FastF1 Session object
    """
    try:
        fastf1.Cache.enable_cache('cache')
        session = fastf1.get_session(year, round_num, session_type)
        session.load(telemetry=False, weather=False)
        return session
    except Exception as e:
        st.error(f"Failed to load {session_type} session for {year} R{round_num}: {e}")
        return None


def get_race_results(session) -> pd.DataFrame:
    """
    Extract race results from session.
    
    Args:
        session: FastF1 Session object
    
    Returns:
        DataFrame with driver results
    """
    if session is None or session.results is None:
        return pd.DataFrame()
    
    results = session.results.copy()
    
    return results[[
        'Abbreviation', 'FullName', 'TeamName', 'Position', 'Points', 'Status'
    ]].rename(columns={
        'Abbreviation': 'Driver',
        'FullName': 'DriverName',
        'TeamName': 'Team',
        'Position': 'FinalPosition',
        'Points': 'Points',
        'Status': 'Status'
    })


def get_lap_data(session) -> pd.DataFrame:
    """
    Extract lap-by-lap data from session.
    
    Args:
        session: FastF1 Session object
    
    Returns:
        DataFrame with lap data
    """
    if session is None or not hasattr(session, 'laps'):
        return pd.DataFrame()
    
    laps = session.laps.copy()
    
    # Calculate gap to leader
    laps['GapToLeader'] = laps.groupby('Driver')['LapTime'].diff().fillna(pd.Timedelta(0))
    laps['GapToLeaderSec'] = laps['GapToLeader'].dt.total_seconds()
    
    # Select available columns only
    available_cols = ['Driver', 'LapNumber', 'LapTime', 'IsAccurate']
    if 'GapToLeader' in laps.columns:
        available_cols.append('GapToLeader')
    if 'Position' in laps.columns:
        available_cols.append('Position')
    
    return laps[available_cols]


def get_driver_telemetry(session, driver: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract telemetry for a specific driver's best lap.
    
    Args:
        session: FastF1 Session object
        driver: Driver abbreviation (e.g., "VER")
    
    Returns:
        Tuple of (telemetry_df, lap_df)
    """
    try:
        if session is None:
            return pd.DataFrame(), pd.DataFrame()
        
        # Ensure telemetry is loaded for this session
        try:
            session.load(telemetry=True, weather=False, messages=False)
        except:
            pass  # Session may already be loaded
        
        # Get best lap
        driver_laps = session.laps.pick_driver(driver)
        if driver_laps.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        best_lap = driver_laps.pick_fastest()
        
        if best_lap is None or best_lap.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Get telemetry - may need to load for this lap
        try:
            telemetry = best_lap.get_telemetry()
        except:
            return pd.DataFrame(), best_lap
        
        if telemetry is None or telemetry.empty:
            return pd.DataFrame(), best_lap
        
        return telemetry, best_lap
    except Exception as e:
        st.warning(f"Could not load telemetry for {driver}: {e}")
        return pd.DataFrame(), pd.DataFrame()


def get_qualifying_results(session) -> pd.DataFrame:
    """
    Extract qualifying results with grid positions.
    
    Args:
        session: FastF1 Qualifying Session object
    
    Returns:
        DataFrame with qualifying results
    """
    if session is None or session.results is None:
        return pd.DataFrame()
    
    results = session.results.copy()
    
    return results[[
        'Abbreviation', 'FullName', 'TeamName', 'GridPosition', 'Q1', 'Q2', 'Q3'
    ]].rename(columns={
        'Abbreviation': 'Driver',
        'FullName': 'DriverName',
        'TeamName': 'Team',
        'GridPosition': 'GridPos'
    })


def get_available_races(year: int) -> list:
    """
    Get list of races for a season.
    
    Args:
        year: Season year
    
    Returns:
        List of (RoundNumber, RaceName) tuples
    """
    schedule = load_season_schedule(year)
    
    if schedule.empty:
        return []
    
    races = []
    for _, row in schedule.iterrows():
        if pd.notna(row['RoundNumber']):
            races.append((
                int(row['RoundNumber']),
                row['EventName']
            ))
    
    return races
