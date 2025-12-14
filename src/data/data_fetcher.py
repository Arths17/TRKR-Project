"""
Data Fetcher Module - Fetches F1 data using FastF1 library
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

import config

# Enable FastF1 cache
if config.USE_CACHE:
    cache_dir = Path(config.CACHE_DIR)
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


class F1DataFetcher:
    """Fetches and aggregates F1 race data from FastF1"""
    
    def __init__(self, seasons: Optional[List[int]] = None):
        """
        Initialize the data fetcher
        
        Args:
            seasons: List of seasons to fetch data for. If None, uses config default.
        """
        self.seasons = seasons or config.HISTORICAL_SEASONS
        self.raw_data = []
        
    def fetch_season_schedule(self, year: int) -> pd.DataFrame:
        """
        Fetch the race schedule for a given season
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with race schedule
        """
        try:
            schedule = fastf1.get_event_schedule(year)
            return schedule
        except Exception as e:
            print(f"Error fetching schedule for {year}: {e}")
            return pd.DataFrame()
    
    def fetch_session_data(self, year: int, race_name: str, session_type: str = 'R') -> Optional[fastf1.core.Session]:
        """
        Fetch data for a specific session
        
        Args:
            year: Season year
            race_name: Name or round number of the race
            session_type: 'FP1', 'FP2', 'FP3', 'Q', 'R' (Race)
            
        Returns:
            FastF1 Session object or None if error
        """
        try:
            session = fastf1.get_session(year, race_name, session_type)
            session.load()
            return session
        except Exception as e:
            print(f"Error fetching {session_type} session for {year} {race_name}: {e}")
            return None

    def extract_qualifying_results(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract qualifying results including best lap times per driver.
        """
        try:
            results = session.results
            laps = session.laps
            best_laps = laps.groupby('Driver')['LapTime'].min().dt.total_seconds()
            q_df = pd.DataFrame({
                'Year': session.event.year,
                'Race': session.event['EventName'],
                'Date': session.event['EventDate'],
                'Circuit': session.event['Location'],
                'Driver': results['Abbreviation'],
                'DriverNumber': results['DriverNumber'],
                'Team': results['TeamName'],
                'QualiPosition': pd.to_numeric(results['Position'], errors='coerce'),
                'QualiBestLap': results['BestLapTime'].dt.total_seconds() if 'BestLapTime' in results.columns else results['Time'].dt.total_seconds() if 'Time' in results.columns else pd.Series(dtype=float)
            })
            # Merge computed best laps if available
            q_df = q_df.merge(best_laps.rename('QualiBestLapComputed'), left_on='Driver', right_index=True, how='left')
            q_df['QualiBestLap'] = q_df['QualiBestLap'].fillna(q_df['QualiBestLapComputed'])
            q_df.drop(columns=['QualiBestLapComputed'], inplace=True, errors='ignore')
            return q_df
        except Exception as e:
            print(f"Error extracting qualifying results: {e}")
            return pd.DataFrame()

    def load_external_qualifying(self, filepath: Optional[str]) -> pd.DataFrame:
        """
        Load qualifying data from external CSV or JSON. Must contain columns:
        ['Year','Race','Driver','Team','QualiPosition','QualiBestLap']
        """
        if not filepath:
            return pd.DataFrame()
        try:
            path = Path(filepath)
            if path.suffix.lower() == '.csv':
                df = pd.read_csv(path)
            elif path.suffix.lower() == '.json':
                df = pd.read_json(path)
            else:
                print(f"Unsupported qualifying file format: {path.suffix}")
                return pd.DataFrame()
            return df
        except Exception as e:
            print(f"Error loading external qualifying file: {e}")
            return pd.DataFrame()
    
    def extract_race_results(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract race results and basic statistics from a session
        
        Args:
            session: Loaded FastF1 session
            
        Returns:
            DataFrame with race results
        """
        try:
            results = session.results
            
            # Basic race information
            race_data = pd.DataFrame({
                'Year': session.event.year,
                'Race': session.event['EventName'],
                'Date': session.event['EventDate'],
                'Circuit': session.event['Location'],
                'Driver': results['Abbreviation'],
                'DriverNumber': results['DriverNumber'],
                'Team': results['TeamName'],
                'Position': results['Position'],
                'GridPosition': results['GridPosition'],
                'Points': results['Points'],
                'Status': results['Status'],
                'Time': results['Time']
            })
            
            return race_data
        except Exception as e:
            print(f"Error extracting race results: {e}")
            return pd.DataFrame()
    
    def extract_lap_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract detailed lap data including lap times, sectors, and tire info
        
        Args:
            session: Loaded FastF1 session
            
        Returns:
            DataFrame with lap-level data
        """
        try:
            laps = session.laps
            
            # Filter out invalid laps
            valid_laps = laps[laps['LapTime'].notna()]
            
            if len(valid_laps) == 0:
                return pd.DataFrame()
            
            lap_data = pd.DataFrame({
                'Year': session.event.year,
                'Race': session.event['EventName'],
                'Driver': valid_laps['Driver'],
                'DriverNumber': valid_laps['DriverNumber'],
                'Team': valid_laps['Team'],
                'LapNumber': valid_laps['LapNumber'],
                'LapTime': valid_laps['LapTime'].dt.total_seconds(),
                'Sector1Time': valid_laps['Sector1Time'].dt.total_seconds() if 'Sector1Time' in valid_laps.columns else np.nan,
                'Sector2Time': valid_laps['Sector2Time'].dt.total_seconds() if 'Sector2Time' in valid_laps.columns else np.nan,
                'Sector3Time': valid_laps['Sector3Time'].dt.total_seconds() if 'Sector3Time' in valid_laps.columns else np.nan,
                'Compound': valid_laps['Compound'],
                'TyreLife': valid_laps['TyreLife'],
                'FreshTyre': valid_laps['FreshTyre'] if 'FreshTyre' in valid_laps.columns else False,
                'Stint': valid_laps['Stint'],
                'TrackStatus': valid_laps['TrackStatus'] if 'TrackStatus' in valid_laps.columns else '1',
                'IsPersonalBest': valid_laps['IsPersonalBest'] if 'IsPersonalBest' in valid_laps.columns else False
            })
            
            return lap_data
        except Exception as e:
            print(f"Error extracting lap data: {e}")
            return pd.DataFrame()
    
    def extract_pitstop_data(self, session: fastf1.core.Session) -> pd.DataFrame:
        """
        Extract pit stop information
        
        Args:
            session: Loaded FastF1 session
            
        Returns:
            DataFrame with pit stop data
        """
        try:
            laps = session.laps
            
            # Get pit stops (laps where PitInTime or PitOutTime is not null)
            pit_laps = laps[laps['PitInTime'].notna()]
            
            if len(pit_laps) == 0:
                return pd.DataFrame()
            
            pitstop_data = pd.DataFrame({
                'Year': session.event.year,
                'Race': session.event['EventName'],
                'Driver': pit_laps['Driver'],
                'LapNumber': pit_laps['LapNumber'],
                'PitInTime': pit_laps['PitInTime'].dt.total_seconds() if 'PitInTime' in pit_laps.columns else np.nan,
                'PitOutTime': pit_laps['PitOutTime'].dt.total_seconds() if 'PitOutTime' in pit_laps.columns else np.nan,
            })
            
            # Calculate pit stop duration
            if 'PitInTime' in pit_laps.columns and 'PitOutTime' in pit_laps.columns:
                pitstop_data['PitDuration'] = pitstop_data['PitOutTime'] - pitstop_data['PitInTime']
            
            return pitstop_data
        except Exception as e:
            print(f"Error extracting pit stop data: {e}")
            return pd.DataFrame()
    
    def fetch_race_with_qualifying(self, year: int, race_identifier) -> Dict[str, pd.DataFrame]:
        """
        Fetch race data INCLUDING qualifying (slower, use for predictions)
        
        Args:
            year: Season year
            race_identifier: Race name or round number
            
        Returns:
            Dictionary with results, laps, pitstops, and qualifying
        """
        print(f"Fetching race + qualifying for {year} - {race_identifier}...")
        
        # Get race data
        race_data = self.fetch_all_race_data(year, race_identifier)
        
        if not race_data:
            return {}
        
        # Add qualifying
        print(f"  Fetching qualifying session...")
        quali_session = self.fetch_session_data(year, race_identifier, 'Q')
        if quali_session:
            race_data['qualifying'] = self.extract_qualifying_results(quali_session)
        
        return race_data
    
    def fetch_all_race_data(self, year: int, race_identifier) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data for a specific race
        
        Args:
            year: Season year
            race_identifier: Race name or round number
            
        Returns:
            Dictionary containing race results, lap data, and pit stop data
        """
        print(f"Fetching data for {year} - {race_identifier}...")
        
        # Fetch race session
        race_session = self.fetch_session_data(year, race_identifier, 'R')
        
        if race_session is None:
            return {}
        
        # Extract all data types
        results = self.extract_race_results(race_session)
        laps = self.extract_lap_data(race_session)
        pitstops = self.extract_pitstop_data(race_session)
        # Qualifying session (disabled for training speed - enable for predictions)
        # quali_session = self.fetch_session_data(year, race_identifier, 'Q')
        # qualifying = self.extract_qualifying_results(quali_session) if quali_session else pd.DataFrame()
        qualifying = pd.DataFrame()
        
        return {
            'results': results,
            'laps': laps,
            'pitstops': pitstops,
            'qualifying': qualifying
        }
    
    def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all configured seasons
        
        Returns:
            Dictionary with aggregated DataFrames for results, laps, and pitstops
        """
        all_results = []
        all_laps = []
        all_pitstops = []
        
        for year in self.seasons:
            print(f"\n=== Fetching data for {year} season ===")
            
            # Get race schedule
            schedule = self.fetch_season_schedule(year)
            
            if schedule.empty:
                continue
            
            # Filter to only completed race events
            races = schedule[schedule['EventFormat'] != 'testing']
            
            for idx, race in races.iterrows():
                try:
                    race_round = race['RoundNumber']
                    
                    # Fetch race data
                    race_data = self.fetch_all_race_data(year, race_round)
                    
                    if race_data:
                        if not race_data.get('results', pd.DataFrame()).empty:
                            all_results.append(race_data['results'])
                        if not race_data.get('laps', pd.DataFrame()).empty:
                            all_laps.append(race_data['laps'])
                        if not race_data.get('pitstops', pd.DataFrame()).empty:
                            all_pitstops.append(race_data['pitstops'])
                    
                except Exception as e:
                    print(f"Error processing race {race.get('EventName', 'Unknown')}: {e}")
                    continue
        
        # Combine all data
        results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
        laps_df = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
        pitstops_df = pd.concat(all_pitstops, ignore_index=True) if all_pitstops else pd.DataFrame()
        
        print(f"\n=== Data Collection Summary ===")
        print(f"Total races: {len(results_df['Race'].unique()) if not results_df.empty else 0}")
        print(f"Total race results: {len(results_df)}")
        print(f"Total laps: {len(laps_df)}")
        print(f"Total pit stops: {len(pitstops_df)}")
        
        return {
            'results': results_df,
            'laps': laps_df,
            'pitstops': pitstops_df
        }


if __name__ == "__main__":
    # Test the fetcher
    fetcher = F1DataFetcher(seasons=[2024])
    data = fetcher.fetch_historical_data()
    
    if not data['results'].empty:
        print("\nSample race results:")
        print(data['results'].head(10))
