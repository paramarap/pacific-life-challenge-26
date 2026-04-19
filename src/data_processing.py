import pandas as pd
import os

def load_raw_data(data_dir='data/raw/'):
    """Loads the participant and wearable datasets."""
    print("Loading raw data...")
    train_participant = pd.read_csv(os.path.join(data_dir, 'participant_data_train.csv'))
    test_participant = pd.read_csv(os.path.join(data_dir, 'participant_data_test.csv'))
    
    train_wearable = pd.read_csv(os.path.join(data_dir, 'wearable_data_train.csv'))
    test_wearable = pd.read_csv(os.path.join(data_dir, 'wearable_data_test.csv'))
    
    return train_participant, test_participant, train_wearable, test_wearable

def aggregate_wearable_data(wearable_df):
    """
    Collapses daily wearable time-series data into single-row patient summaries.
    Focuses on key actuarial risk indicators: resting heart rate, SpO2, sleep, and AFib.
    """
    print("Aggregating wearable time-series data...")
    agg_funcs = {
        'steps': ['mean', 'min', 'max'],
        'sleep_duration': ['mean', 'std'],
        'RHR_U': ['mean', 'max'],        # Upper resting heart rate
        'Sp02_L': ['mean', 'min'],       # Lower SpO2 
        '__afib_daily': ['sum', 'mean'], # Total days and % of days with AFib
        '__exercise': ['mean']           # % of days exercised
    }
    
    # Group by participant ID
    wearable_agg = wearable_df.groupby('ID').agg(agg_funcs).reset_index()
    
    # Flatten MultiIndex columns (e.g., 'steps_mean')
    wearable_agg.columns = ['ID'] + [f"{col[0]}_{col[1]}" for col in wearable_agg.columns[1:]]
    
    return wearable_agg

def build_merged_datasets(train_part, test_part, train_wear, test_wear):
    """Aggregates wearables and merges them with traditional participant data."""
    train_wear_agg = aggregate_wearable_data(train_wear)
    test_wear_agg = aggregate_wearable_data(test_wear)
    
    # Left join ensures we keep all participants, even those without wearables
    train_full = train_part.merge(train_wear_agg, on='ID', how='left')
    test_full = test_part.merge(test_wear_agg, on='ID', how='left')
    
    return train_full, test_full