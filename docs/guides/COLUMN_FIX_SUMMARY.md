# F1 Prediction System - Robust Column Handling Fix

## Problem Summary
The script was crashing with the error:
```
['Laps_INTERMEDIATE', 'Laps_None', 'Laps_WET'] not in index
```

This occurred because not all F1 races use all tire compounds (e.g., dry races don't use INTERMEDIATE or WET tires).

## Solution Implemented

I've rewritten the data processing functions to be **fully defensive** against missing columns. The system now:

### ✅ 1. Never Crashes on Missing Columns
All functions check for column existence before using them.

### ✅ 2. Automatically Adds Missing Columns with Zeros
Expected columns that don't exist are created with default values (0).

### ✅ 3. Maintains All Feature Engineering
All existing calculations (lap times, pit stops, tire strategy) work unchanged.

### ✅ 4. Prints Warnings for Debugging
You'll see informative warnings when columns are missing:
```
Warning: Added 3 missing columns with default values (0)
Missing columns: Laps_INTERMEDIATE, Laps_WET, Laps_None
```

### ✅ 5. Compatible with Any FastF1 Data
Works with past, current, and future races (including 2025 Abu Dhabi GP).

---

## Files Modified

### 1. `data_processor.py`

#### **Function: `compute_tire_strategy_stats()`**
**Changes:**
- Checks if 'Compound' column exists before processing
- After pivoting tire data, adds all expected tire columns:
  - `Laps_SOFT`, `Laps_MEDIUM`, `Laps_HARD`
  - `Laps_INTERMEDIATE`, `Laps_WET`
  - `Laps_None` (for unspecified compounds)
- Missing columns are filled with 0

**Code snippet:**
```python
# Check if Compound column exists
if 'Compound' not in self.laps.columns:
    print("Warning: 'Compound' column not found in lap data...")
    return pd.DataFrame()

# ... existing pivot logic ...

# Add standard tire compound columns with zeros if missing
expected_compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
for compound in expected_compounds:
    col_name = f'Laps_{compound}'
    if col_name not in compound_pivot.columns:
        compound_pivot[col_name] = 0

# Handle 'None' compound
if 'Laps_None' not in compound_pivot.columns:
    compound_pivot['Laps_None'] = 0
```

---

#### **Function: `compute_driver_lap_stats()`**
**Changes:**
- Builds aggregation dictionary dynamically based on available columns
- Only processes columns that exist (LapTime, Sector times, TyreLife, Stint)
- Safely renames only columns that were actually created

**Code snippet:**
```python
# Build aggregation dict dynamically based on available columns
agg_dict = {}

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
```

---

#### **New Function: `ensure_standard_columns()`**
**Purpose:** Central defensive check that ensures ALL expected feature columns exist.

**What it does:**
- Defines all expected columns for:
  - Tire compounds (6 columns)
  - Lap statistics (10 columns)
  - Pit stops (2 columns)
  - Team performance (3 columns)
  - Circuit history (3 columns)
- Adds any missing column with value = 0
- Prints warning with list of added columns

**Code snippet:**
```python
def ensure_standard_columns(self, df: pd.DataFrame) -> pd.DataFrame:
    # Define all expected tire compound columns
    expected_tire_cols = [
        'Laps_SOFT', 'Laps_MEDIUM', 'Laps_HARD',
        'Laps_INTERMEDIATE', 'Laps_WET', 'Laps_None'
    ]
    
    # ... define other expected columns ...
    
    missing_cols = []
    for col in all_expected:
        if col not in df.columns:
            df[col] = 0
            missing_cols.append(col)
    
    if missing_cols:
        print(f"Warning: Added {len(missing_cols)} missing columns...")
```

---

#### **Function: `merge_all_features()`**
**Changes:**
- Now calls `ensure_standard_columns()` after all merges
- This guarantees consistent feature set regardless of data availability

**Code snippet:**
```python
# Compute recent form (must be done after initial merge)
merged = self.compute_recent_form(merged, window=config.RECENT_RACES_WINDOW)

# Ensure all standard columns exist (defensive programming)
merged = self.ensure_standard_columns(merged)

print(f"Features merged. Total records: {len(merged)}")
```

---

### 2. `predictor.py`

#### **Function: `prepare_prediction_features()`**
**Changes:**
- Iterates through expected feature columns
- Adds each column from race data if available, otherwise uses 0
- Prints warning if features are missing

**Code snippet:**
```python
# Add each feature column, using 0 if missing
missing_features = []
for col in self.feature_columns:
    if col in race_data.columns:
        X[col] = race_data[col]
    else:
        X[col] = 0
        missing_features.append(col)

if missing_features:
    print(f"Warning: {len(missing_features)} features missing, filled with zeros")
```

---

## Expected Behavior

### Before Fix:
```
❌ KeyError: ['Laps_INTERMEDIATE', 'Laps_None', 'Laps_WET'] not in index
❌ Script crashes
❌ Can't process races without all tire types
```

### After Fix:
```
✅ Warning: Added 3 missing columns with default values (0)
✅ Missing columns: Laps_INTERMEDIATE, Laps_WET, Laps_None
✅ Processing continues normally
✅ All features available for ML model
✅ Works for any race configuration
```

---

## Testing

Run `test_robust_processing.py` to verify:
```bash
python test_robust_processing.py
```

This tests:
1. Missing tire compound columns
2. Missing sector time data
3. Missing stint information
4. Complete absence of compound data

---

## Compatibility

The fixed code is now compatible with:

✅ **Dry races** (no INTERMEDIATE/WET tires)  
✅ **Wet races** (limited dry tire usage)  
✅ **Sprint races** (different tire allocations)  
✅ **Historical data** (older seasons with different tire rules)  
✅ **Future races** (2025 Abu Dhabi GP and beyond)  
✅ **Incomplete data** (when FastF1 hasn't received full telemetry)  
✅ **Testing sessions** (limited data availability)  

---

## Key Principles Applied

1. **Defensive Programming**: Check before accessing columns
2. **Fail-Safe Defaults**: Use 0 for missing numeric features
3. **Informative Warnings**: Tell user what's missing for debugging
4. **Consistent Feature Sets**: Always produce same columns for ML model
5. **Graceful Degradation**: Work with whatever data is available

---

## Usage Example

```python
from data_fetcher import F1DataFetcher
from data_processor import F1DataProcessor

# Fetch any race (even with missing tire data)
fetcher = F1DataFetcher(seasons=[2025])
race_data = fetcher.fetch_all_race_data(2025, "Abu Dhabi")

# Process safely - will not crash!
processor = F1DataProcessor(race_data)
processed = processor.process()

# All expected columns will be present
print(f"Features: {len(processed.columns)}")
# Output: Features: 40+ (even if some were missing in raw data)
```

---

## Summary

Your F1 prediction system is now **production-ready** and will handle:
- Any race from any season
- Any tire compound configuration
- Missing telemetry data
- Future races with unknown tire strategies

**No more crashes from missing tire columns!** 🏎️✅
