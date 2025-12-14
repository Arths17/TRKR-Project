"""
Data Processor Module - Processes raw F1 data and engineers features for ML
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import config


class F1DataProcessor:
    """Processes and engineers features from raw F1 data"""
    
    def __init__(self, raw_data: Dict[str, pd.DataFrame]):
        """
        Initialize the processor with raw data
        
        Args:
            raw_data: Dictionary containing 'results', 'laps', and 'pitstops' DataFrames
        """
        self.results = raw_data.get('results', pd.DataFrame())
        self.laps = raw_data.get('laps', pd.DataFrame())
        self.pitstops = raw_data.get('pitstops', pd.DataFrame())
        self.qualifying = raw_data.get('qualifying', pd.DataFrame())
        self.processed_data = None
        
    def clean_data(self):
        """Clean and validate the raw data"""
        print("Cleaning data...")
        
        # Clean results
        if not self.results.empty:
            # Remove DNF and DNS where appropriate
            self.results = self.results.copy()
            self.results['Position'] = pd.to_numeric(self.results['Position'], errors='coerce')
            self.results['GridPosition'] = pd.to_numeric(self.results['GridPosition'], errors='coerce')
            self.results['Points'] = pd.to_numeric(self.results['Points'], errors='coerce').fillna(0)
            
        # Clean lap data
        if not self.laps.empty:
            self.laps = self.laps.copy()
            # Remove outlier lap times (likely errors or pit laps)
            self.laps = self.laps[self.laps['LapTime'] > 0]
            self.laps = self.laps[self.laps['LapTime'] < 300]  # Max 5 minutes per lap

        # Clean qualifying
        if not self.qualifying.empty:
            self.qualifying = self.qualifying.copy()
            self.qualifying['QualiPosition'] = pd.to_numeric(self.qualifying['QualiPosition'], errors='coerce')
            if 'QualiBestLap' in self.qualifying.columns:
                self.qualifying['QualiBestLap'] = pd.to_numeric(self.qualifying['QualiBestLap'], errors='coerce')
            
        print("Data cleaning completed.")
    
    def compute_weather_features(self) -> pd.DataFrame:
        """
        Extract weather conditions that affect race performance
        
        Returns:
            DataFrame with weather features per race
        """
        if self.results.empty:
            return pd.DataFrame()
        
        # Get weather data from laps (FastF1 includes weather per lap)
        if self.laps.empty or 'AirTemp' not in self.laps.columns:
            return pd.DataFrame()
        
        # Aggregate weather conditions per race
        weather_stats = self.laps.groupby(['Year', 'Race']).agg({
            'AirTemp': 'mean',
            'TrackTemp': 'mean',
            'Humidity': 'mean',
            'Rainfall': 'max'  # Any rain during race
        }).reset_index()
        
        weather_stats.columns = ['Year', 'Race', 'AvgAirTemp', 'AvgTrackTemp', 'AvgHumidity', 'RainRace']
        
        return weather_stats
    
    def compute_tire_performance(self) -> pd.DataFrame:
        """
        Analyze tire compound performance per driver/circuit
        
        Returns:
            DataFrame with tire performance metrics
        """
        if self.laps.empty or 'Compound' not in self.laps.columns:
            return pd.DataFrame()
        
        # Calculate average lap time per compound per driver per circuit
        tire_perf = self.laps.groupby(['Year', 'Race', 'Driver', 'Compound'])['LapTime'].mean().reset_index()
        tire_perf.columns = ['Year', 'Race', 'Driver', 'Compound', 'CompoundAvgLap']
        
        # Pivot to get compound performance as features
        tire_pivot = tire_perf.pivot_table(
            index=['Year', 'Race', 'Driver'],
            columns='Compound',
            values='CompoundAvgLap',
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        tire_pivot.columns = [f'TirePerf_{col}' if col not in ['Year', 'Race', 'Driver'] 
                             else col for col in tire_pivot.columns]
        
        return tire_pivot
    
    def compute_recent_performance(self) -> pd.DataFrame:
        """
        Compute recent driver performance from last 3-5 races (pre-race features)
        
        Returns:
            DataFrame with recent performance metrics
        """
        if self.results.empty:
            return pd.DataFrame()
        
        df = self.results.copy()
        
        # Ensure Date column exists
        if 'Date' not in df.columns:
            return pd.DataFrame()
        
        # Sort chronologically
        df = df.sort_values(['Driver', 'Date'])
        
        recent_features = []
        
        for driver in df['Driver'].unique():
            driver_data = df[df['Driver'] == driver].copy()
            driver_data = driver_data.sort_values('Date')
            
            # Recent position (last 3 races avg, exclude current)
            driver_data['RecentPosition'] = driver_data['Position'].shift(1).rolling(window=3, min_periods=1).mean()
            
            # Recent points (last 5 races avg)
            driver_data['RecentPoints'] = driver_data['Points'].shift(1).rolling(window=5, min_periods=1).mean()
            
            # Recent average lap time (if available)
            if 'AvgLapTime' in driver_data.columns:
                driver_data['RecentAvgLapTime'] = driver_data['AvgLapTime'].shift(1).rolling(window=3, min_periods=1).mean()
                driver_data['RecentFastestLap'] = driver_data.get('FastestLap', driver_data['AvgLapTime']).shift(1).rolling(window=3, min_periods=1).min()
            
            recent_features.append(driver_data)
        
        recent_df = pd.concat(recent_features, ignore_index=True)
        
        # Fill NaN
        fill_cols = ['RecentPosition', 'RecentPoints', 'RecentAvgLapTime', 'RecentFastestLap']
        for col in fill_cols:
            if col in recent_df.columns:
                recent_df[col] = recent_df[col].fillna(0)
        
        cols_to_return = ['Year', 'Race', 'Driver', 'RecentPosition', 'RecentPoints']
        if 'RecentAvgLapTime' in recent_df.columns:
            cols_to_return.extend(['RecentAvgLapTime', 'RecentFastestLap'])
        
        return recent_df[cols_to_return]
    
    def compute_driver_lap_stats(self) -> pd.DataFrame:
        """
        Compute aggregated lap statistics per driver per race
        
        Returns:
            DataFrame with driver lap statistics
        """
        if self.laps.empty:
            return pd.DataFrame()
        
        # Build aggregation dict dynamically based on available columns
        agg_dict = {}
        
        # LapTime should always be present
        if 'LapTime' in self.laps.columns:
            agg_dict['LapTime'] = ['mean', 'std', 'min', 'count']
        
        # Optional sector times
        for sector in ['Sector1Time', 'Sector2Time', 'Sector3Time']:
            if sector in self.laps.columns:
                agg_dict[sector] = ['mean']
        
        # Optional tire life
        if 'TyreLife' in self.laps.columns:
            agg_dict['TyreLife'] = ['max', 'mean']
        
        # Optional stint info
        if 'Stint' in self.laps.columns:
            agg_dict['Stint'] = ['max']
        
        if not agg_dict:
            print("Warning: No valid columns found for lap statistics")
            return pd.DataFrame()
        
        # Group by race and driver
        lap_stats = self.laps.groupby(['Year', 'Race', 'Driver', 'Team']).agg(agg_dict).reset_index()

        # Add driver consistency metrics if possible
        if 'LapTime_mean' in lap_stats.columns and 'LapTime_std' in lap_stats.columns:
            # Coefficient of variation (consistency: lower is better)
            with np.errstate(divide='ignore', invalid='ignore'):
                lap_stats['ConsistencyCV'] = (lap_stats['LapTime_std'] / lap_stats['LapTime_mean']).fillna(0)
        
        # Flatten column names
        lap_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in lap_stats.columns.values]
        
        # Rename for clarity
        rename_map = {
            'LapTime_mean': 'AvgLapTime',
            'LapTime_std': 'LapTimeStd',
            'LapTime_min': 'FastestLap',
            'LapTime_count': 'LapsCompleted',
            'Sector1Time_mean': 'AvgSector1',
            'Sector2Time_mean': 'AvgSector2',
            'Sector3Time_mean': 'AvgSector3',
            'TyreLife_max': 'MaxTyreLife',
            'TyreLife_mean': 'AvgTyreLife',
            'Stint_max': 'NumStints'
        }
        
        # Only rename columns that exist
        existing_renames = {k: v for k, v in rename_map.items() if k in lap_stats.columns}
        lap_stats.rename(columns=existing_renames, inplace=True)
        
        return lap_stats
    
    def compute_tire_strategy_stats(self) -> pd.DataFrame:
        """
        Compute tire strategy statistics
        
        Returns:
            DataFrame with tire strategy metrics
        """
        if self.laps.empty:
            return pd.DataFrame()
        
        # Check if Compound column exists
        if 'Compound' not in self.laps.columns:
            print("Warning: 'Compound' column not found in lap data. Skipping tire strategy stats.")
            return pd.DataFrame()
        
        # Analyze tire compound usage
        tire_stats = self.laps.groupby(['Year', 'Race', 'Driver', 'Compound']).agg({
            'LapTime': ['mean', 'count'],
            'TyreLife': ['mean', 'max']
        }).reset_index()
        
        tire_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in tire_stats.columns.values]
        
        # Pivot to get compound usage per driver
        compound_pivot = tire_stats.pivot_table(
            index=['Year', 'Race', 'Driver'],
            columns='Compound',
            values='LapTime_count',
            fill_value=0
        ).reset_index()
        
        # Flatten column names
        compound_pivot.columns = [f'Laps_{col}' if col not in ['Year', 'Race', 'Driver'] 
                                  else col for col in compound_pivot.columns]
        
        # Add standard tire compound columns with zeros if missing
        expected_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
        for compound in expected_compounds:
            col_name = f'Laps_{compound}'
            if col_name not in compound_pivot.columns:
                compound_pivot[col_name] = 0
        
        # Handle 'None' compound (sometimes appears in data)
        if 'Laps_None' not in compound_pivot.columns:
            compound_pivot['Laps_None'] = 0
        
        # Stint effectiveness: compare average lap times per stint (if available)
        stint_effect = pd.DataFrame()
        if 'Stint' in self.laps.columns:
            se = self.laps.groupby(['Year', 'Race', 'Driver', 'Stint'])['LapTime'].mean().reset_index()
            # Calculate improvement from stint to stint (negative means faster)
            se['StintAvgLap'] = se['LapTime']
            se.sort_values(['Year', 'Race', 'Driver', 'Stint'], inplace=True)
            se['StintDelta'] = se.groupby(['Year', 'Race', 'Driver'])['StintAvgLap'].diff().fillna(0)
            stint_effect = se.groupby(['Year', 'Race', 'Driver']).agg({
                'StintDelta': ['mean']
            }).reset_index()
            stint_effect.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in stint_effect.columns]
            stint_effect.rename(columns={'StintDelta_mean': 'AvgStintImprovement'}, inplace=True)

        # Merge compound and stint effectiveness
        out = compound_pivot
        if not stint_effect.empty:
            out = out.merge(stint_effect, on=['Year', 'Race', 'Driver'], how='left')
            out['AvgStintImprovement'] = out['AvgStintImprovement'].fillna(0)

        return out
    
    def compute_historical_driver_performance(self) -> pd.DataFrame:
        """
        Compute historical driver performance metrics (pre-race features)
        Uses only data from PREVIOUS races, not the current race
        
        Returns:
            DataFrame with historical performance features
        """
        if self.results.empty:
            return pd.DataFrame()
        
        df = self.results.copy()
        
        # Ensure we have a Date column for proper chronological sorting
        if 'Date' not in df.columns:
            print("Warning: No Date column for historical features")
            return pd.DataFrame()
        
        # Sort by driver and date to ensure chronological order
        df = df.sort_values(['Driver', 'Date'])
        
        # For each driver, compute rolling stats from previous races
        historical_features = []
        
        for driver in df['Driver'].unique():
            driver_data = df[df['Driver'] == driver].copy()
            driver_data = driver_data.sort_values('Date')
            
            # Rolling average position (last 5 races)
            driver_data['HistoricalAvgPosition'] = driver_data['Position'].shift(1).rolling(window=5, min_periods=1).mean()
            
            # Rolling average points (last 5 races)
            driver_data['HistoricalAvgPoints'] = driver_data['Points'].shift(1).rolling(window=5, min_periods=1).mean()
            
            # Count of podiums in last 10 races
            driver_data['RecentPodiums'] = (driver_data['Position'] <= 3).shift(1).rolling(window=10, min_periods=1).sum()
            
            # Count of wins in last 10 races
            driver_data['RecentWins'] = (driver_data['Position'] == 1).shift(1).rolling(window=10, min_periods=1).sum()
            
            # Recent form (avg position in last 3 races)
            driver_data['RecentForm'] = driver_data['Position'].shift(1).rolling(window=3, min_periods=1).mean()
            
            historical_features.append(driver_data)
        
        historical_df = pd.concat(historical_features, ignore_index=True)
        
        # Fill NaN for first race of each driver
        fill_cols = ['HistoricalAvgPosition', 'HistoricalAvgPoints', 'RecentPodiums', 'RecentWins', 'RecentForm']
        for col in fill_cols:
            if col in historical_df.columns:
                historical_df[col] = historical_df[col].fillna(0)
        
        return historical_df[['Year', 'Race', 'Driver', 'HistoricalAvgPosition', 
                              'HistoricalAvgPoints', 'RecentPodiums', 'RecentWins', 'RecentForm']]
    
    def compute_pitstop_stats(self) -> pd.DataFrame:
        """
        Compute pit stop statistics
        
        Returns:
            DataFrame with pit stop metrics
        """
        if self.pitstops.empty:
            return pd.DataFrame()
        
        pitstop_stats = self.pitstops.groupby(['Year', 'Race', 'Driver']).agg({
            'PitDuration': ['mean', 'count']
        }).reset_index()
        
        pitstop_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                 for col in pitstop_stats.columns.values]
        
        pitstop_stats.rename(columns={
            'PitDuration_mean': 'AvgPitDuration',
            'PitDuration_count': 'NumPitStops'
        }, inplace=True)
        
        return pitstop_stats
    
    def compute_recent_form(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Compute recent performance metrics (rolling averages)
        
        Args:
            df: DataFrame with race results
            window: Number of recent races to consider
            
        Returns:
            DataFrame with recent form metrics
        """
        if df.empty:
            return df
        
        # Sort by date
        df = df.sort_values(['Driver', 'Date'])
        
        # Calculate rolling metrics
        for col in ['Position', 'Points', 'AvgLapTime', 'FastestLap']:
            if col in df.columns:
                df[f'Recent{col}'] = df.groupby('Driver')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        return df
    
    def compute_circuit_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute driver performance at specific circuits
        
        Args:
            df: DataFrame with race results
            
        Returns:
            DataFrame with circuit-specific metrics
        """
        if df.empty:
            return df
        
        # Calculate average position and points at each circuit
        circuit_stats = df.groupby(['Driver', 'Circuit']).agg({
            'Position': ['mean', 'count'],
            'Points': 'mean'
        }).reset_index()
        
        circuit_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                 for col in circuit_stats.columns.values]
        
        circuit_stats.rename(columns={
            'Position_mean': 'CircuitAvgPosition',
            'Position_count': 'CircuitRaces',
            'Points_mean': 'CircuitAvgPoints'
        }, inplace=True)
        
        return circuit_stats
    
    def compute_team_performance(self) -> pd.DataFrame:
        """
        Compute team-level performance metrics
        
        Returns:
            DataFrame with team metrics
        """
        if self.results.empty:
            return pd.DataFrame()
        
        team_stats = self.results.groupby(['Year', 'Race', 'Team']).agg({
            'Position': ['mean', 'min'],
            'Points': 'sum'
        }).reset_index()
        
        team_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                              for col in team_stats.columns.values]
        
        team_stats.rename(columns={
            'Position_mean': 'TeamAvgPosition',
            'Position_min': 'TeamBestPosition',
            'Points_sum': 'TeamPoints'
        }, inplace=True)
        
        return team_stats
    
    def ensure_standard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all expected feature columns exist, adding missing ones with zeros
        
        Args:
            df: DataFrame to check and fill
            
        Returns:
            DataFrame with all standard columns
        """
        # Define all expected tire compound columns
        expected_tire_cols = [
            'Laps_SOFT', 'Laps_MEDIUM', 'Laps_HARD',
            'Laps_INTERMEDIATE', 'Laps_WET', 'Laps_None'
        ]
        
        # Define other expected feature columns that might be missing
        expected_lap_cols = [
            'AvgLapTime', 'LapTimeStd', 'FastestLap', 'LapsCompleted',
            'AvgSector1', 'AvgSector2', 'AvgSector3',
            'MaxTyreLife', 'AvgTyreLife', 'NumStints'
        ]
        
        expected_pitstop_cols = ['AvgPitDuration', 'NumPitStops']
        
        expected_team_cols = ['TeamAvgPosition', 'TeamBestPosition', 'TeamPoints']
        
        expected_circuit_cols = ['CircuitAvgPosition', 'CircuitRaces', 'CircuitAvgPoints']
        
        expected_qualifying_cols = ['QualiPosition', 'QualiBestLap']
        
        expected_historical_cols = ['HistoricalAvgPosition', 'HistoricalAvgPoints',
                                   'RecentPodiums', 'RecentWins', 'RecentForm',
                                   'RecentPosition', 'RecentPoints', 'RecentAvgLapTime', 'RecentFastestLap']
        
        expected_weather_cols = ['AvgAirTemp', 'AvgTrackTemp', 'AvgHumidity', 'RainRace']
        
        expected_tire_cols = ['TirePerf_SOFT', 'TirePerf_MEDIUM', 'TirePerf_HARD', 'TirePerf_INTERMEDIATE', 'TirePerf_WET']
        
        all_expected = (expected_tire_cols + expected_lap_cols + 
                       expected_pitstop_cols + expected_team_cols + expected_circuit_cols +
                       expected_qualifying_cols + expected_historical_cols + 
                       expected_weather_cols + expected_tire_cols)
        
        missing_cols = []
        imputed_cols = []
        for col in all_expected:
            if col not in df.columns:
                # Smart imputation based on feature type
                if 'TirePerf' in col:
                    # Use median tire performance from team or overall
                    df[col] = 90.0  # Reasonable default lap time
                    imputed_cols.append(col)
                elif 'Temp' in col or 'Humidity' in col:
                    # Typical race conditions
                    if 'AirTemp' in col:
                        df[col] = 25.0
                    elif 'TrackTemp' in col:
                        df[col] = 35.0
                    elif 'Humidity' in col:
                        df[col] = 60.0
                    imputed_cols.append(col)
                elif any(x in col for x in ['Position', 'Points', 'Podiums', 'Wins']):
                    # Use median/mean from historical data if available
                    if 'Position' in col:
                        df[col] = 10.5  # Mid-grid default
                    else:
                        df[col] = 0
                    imputed_cols.append(col)
                else:
                    df[col] = 0
                    missing_cols.append(col)
        
        if missing_cols:
            print(f"Warning: Added {len(missing_cols)} missing columns with zeros")
            if len(missing_cols) <= 6:
                print(f"Zero-filled: {', '.join(missing_cols)}")
        if imputed_cols:
            print(f"✓ Smart imputation for {len(imputed_cols)} features (using realistic defaults)")
        
        return df
    
    def merge_all_features(self) -> pd.DataFrame:
        """
        Merge all computed features into a single dataset
        
        Returns:
            DataFrame with all features
        """
        print("Merging features...")
        
        # Start with race results
        merged = self.results.copy()
        
        # Add historical driver performance (pre-race features)
        hist_perf = self.compute_historical_driver_performance()
        if not hist_perf.empty:
            merged = merged.merge(hist_perf, on=['Year', 'Race', 'Driver'], how='left')
            print(f"Added historical performance features")
        
        # Add recent performance (last 3-5 races)
        recent_perf = self.compute_recent_performance()
        if not recent_perf.empty:
            merged = merged.merge(recent_perf, on=['Year', 'Race', 'Driver'], how='left')
            print(f"Added recent performance features")
        
        # Add weather features
        weather_feats = self.compute_weather_features()
        if not weather_feats.empty:
            merged = merged.merge(weather_feats, on=['Year', 'Race'], how='left')
            print(f"Added weather features")
        
        # Add tire performance features
        tire_feats = self.compute_tire_performance()
        if not tire_feats.empty:
            merged = merged.merge(tire_feats, on=['Year', 'Race', 'Driver'], how='left')
            print(f"Added tire performance features")
        
        # Add lap statistics
        lap_stats = self.compute_driver_lap_stats()
        if not lap_stats.empty:
            merged = merged.merge(
                lap_stats,
                on=['Year', 'Race', 'Driver', 'Team'],
                how='left'
            )
        
        # Add tire strategy stats
        tire_stats = self.compute_tire_strategy_stats()
        if not tire_stats.empty:
            merged = merged.merge(
                tire_stats,
                on=['Year', 'Race', 'Driver'],
                how='left'
            )
        
        # Add pit stop stats
        pitstop_stats = self.compute_pitstop_stats()
        if not pitstop_stats.empty:
            merged = merged.merge(
                pitstop_stats,
                on=['Year', 'Race', 'Driver'],
                how='left'
            )
        
        # Add team performance
        team_stats = self.compute_team_performance()
        if not team_stats.empty:
            merged = merged.merge(
                team_stats,
                on=['Year', 'Race', 'Team'],
                how='left'
            )
        
        # Add circuit performance (historical)
        circuit_stats = self.compute_circuit_performance(merged)
        if not circuit_stats.empty:
            merged = merged.merge(
                circuit_stats,
                on=['Driver', 'Circuit'],
                how='left'
            )
        
        # Merge qualifying features
        if not self.qualifying.empty:
            qcols = ['Year', 'Race', 'Driver', 'QualiPosition', 'QualiBestLap']
            qdf = self.qualifying[qcols].copy()
            merged = merged.merge(qdf, on=['Year', 'Race', 'Driver'], how='left')

        # Compute recent form (must be done after initial merge)
        merged = self.compute_recent_form(merged, window=config.RECENT_RACES_WINDOW)
        
        # Ensure all standard columns exist (defensive programming)
        merged = self.ensure_standard_columns(merged)
        
        print(f"Features merged. Total records: {len(merged)}")
        
        return merged
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for ML model
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with target variables added
        """
        if df.empty:
            return df
        
        # Binary target: Won race (Position == 1)
        df['Won'] = (df['Position'] == 1).astype(int)
        
        # Binary target: Podium finish (Position <= 3)
        df['Podium'] = (df['Position'] <= 3).astype(int)
        
        # Binary target: Points finish (Position <= 10 for most seasons)
        df['PointsFinish'] = (df['Points'] > 0).astype(int)
        
        # Binary target: Top 5 finish
        df['Top5'] = (df['Position'] <= 5).astype(int)
        
        return df
    
    def process(self) -> pd.DataFrame:
        """
        Execute the full processing pipeline
        
        Returns:
            Processed DataFrame ready for ML
        """
        print("\n=== Starting Data Processing ===")
        
        # Clean data
        self.clean_data()
        
        # Merge all features
        processed = self.merge_all_features()
        
        # Create target variables
        processed = self.create_target_variables(processed)
        
        # Fill NaN values with appropriate defaults
        numeric_columns = processed.select_dtypes(include=[np.number]).columns
        processed[numeric_columns] = processed[numeric_columns].fillna(0)
        
        # Store processed data
        self.processed_data = processed
        
        print("\n=== Processing Summary ===")
        print(f"Total features: {len(processed.columns)}")
        print(f"Total records: {len(processed)}")
        print(f"Date range: {processed['Date'].min()} to {processed['Date'].max()}")
        
        return processed
    
    def get_feature_columns(self, pre_race_only: bool = True) -> List[str]:
        """
        Get list of feature columns for ML model
        
        Args:
            pre_race_only: If True, exclude post-race features (for prediction)
        
        Returns:
            List of feature column names
        """
        if self.processed_data is None:
            return []
        
        # Exclude non-feature columns (targets, identifiers, etc.)
        exclude_cols = ['Year', 'Race', 'Date', 'Circuit', 'Driver', 'DriverNumber', 
                       'Team', 'Status', 'Time', 'Won', 'Podium', 'PointsFinish', 'Top5',
                       'PredRaceTime']  # Don't use target as feature
        
        # Post-race features (only available after race, not for prediction)
        post_race_features = ['Position', 'Points', 'AvgLapTime', 'LapTimeStd', 
                             'FastestLap', 'LapsCompleted', 'AvgSector1', 'AvgSector2', 
                             'AvgSector3', 'MaxTyreLife', 'AvgTyreLife', 'NumStints',
                             'AvgPitDuration', 'NumPitStops', 'ConsistencyCV',
                             'Laps_SOFT', 'Laps_MEDIUM', 'Laps_HARD', 'Laps_INTERMEDIATE',
                             'Laps_WET', 'Laps_None', 'AvgStintImprovement']
        
        if pre_race_only:
            exclude_cols.extend(post_race_features)
        
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in exclude_cols and self.processed_data[col].dtype in ['int64', 'float64']]
        
        return feature_cols


if __name__ == "__main__":
    # Test the processor with sample data
    from data_fetcher import F1DataFetcher
    
    fetcher = F1DataFetcher(seasons=[2024])
    raw_data = fetcher.fetch_historical_data()
    
    processor = F1DataProcessor(raw_data)
    processed_data = processor.process()
    
    print("\nSample processed data:")
    print(processed_data.head())
    
    print("\nFeature columns:")
    print(processor.get_feature_columns())
