import pandas as pd
import numpy as np
from scipy.stats import linregress

def calculate_mets(row):
    """
    Maps specific exercise types to their estimated Metabolic Equivalent of Task (MET) values.
    Returns the total energy expenditure in MET-minutes.
    """
    met_map = {
        'running': 8.0,
        'swimming': 7.0,
        'basketball': 6.0,
        'tennis': 7.3,
        'rock-climbing': 5.5,
        'lifting': 3.5
    }
    exercise = str(row['exercise_type']).lower()
    duration = row['exercise_duration'] if pd.notnull(row['exercise_duration']) else 0
    met_value = met_map.get(exercise, 0.0)
    return met_value * duration

def calculate_trend(series):
    """
    Calculates the linear trend (slope) of a time series to capture physiological deterioration 
    or improvement over the monitoring period.
    """
    series = series.dropna()
    if len(series) < 5:
        return np.nan
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x, series.values)
    return slope

def engineer_wearable_features(df_wearable):
    """
    Aggregates longitudinal wearable data into participant-level features.
    """
    # Ensure timestamp is datetime and sort
    df_wearable['timestamp'] = pd.to_datetime(df_wearable['timestamp'])
    df_wearable = df_wearable.sort_values(by=)
    
    # Calculate metabolic expenditure
    df_wearable['met_minutes'] = df_wearable.apply(calculate_mets, axis=1)
    
    # Calculate sleep efficiency (Deep + REM / Total Duration)
    # Adding a small epsilon to avoid division by zero
    df_wearable['sleep_efficiency'] = (df_wearable['sleep_deep'] + df_wearable['sleep_rem']) / (df_wearable['sleep_duration'] + 1e-9)
    
    # Calculate hypoxic burden (days where lower SpO2 drops below 92%)
    df_wearable['hypoxic_event'] = (df_wearable < 92.0).astype(int)
    
    # Calculate Heart Rate Reserve (MaxHR - Resting HR Lower)
    df_wearable['hr_reserve'] = df_wearable - df_wearable
    
    # Aggregate features per participant
    agg_funcs = {
        'steps': ['mean', 'median', 'std'],
        'met_minutes': ['sum', 'mean'],
        'afib_daily': ['sum', 'mean'],
        'RHR_L': ['mean', 'max', calculate_trend],
        'hr_reserve': ['mean', 'min'],
        'Sp02_L': ['min', 'mean'],
        'SpO2_U': ['mean'],
        'hypoxic_event': ['sum'],
        'sleep_duration': ['mean', 'std'],
        'sleep_efficiency': ['mean'],
        'n_wakeups': ['mean', 'max'],
        'snoring': ['mean']
    }
    
    df_features = df_wearable.groupby('ID').agg(agg_funcs)
    
    # Flatten multi-level columns
    df_features.columns = ['_'.join(col).strip() for col in df_features.columns.values]
    
    # Rename complex function columns for clarity
    df_features = df_features.rename(columns={'RHR_L_calculate_trend': 'RHR_L_trend'})
    
    return df_features.reset_index()