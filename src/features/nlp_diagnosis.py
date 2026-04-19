import pandas as pd
import re

def clean_and_map_diagnoses(df_diag):
    """
    Cleans unstructured diagnostic text and maps to standardized clinical concepts
    based on the specific noise and typographical errors observed in the dataset.
    """
    df_diag['diagnosis_clean'] = df_diag['diagnosis'].astype(str).str.lower()
    
    # Define regex patterns for entity resolution
    patterns = {
        'hypertension': r'hbp|high blood pressure|hypertension',
        'diabetes_t2': r'type ii diabetes|type 2|t2dm|diabetes 2',
        'diabetes_t1': r'type 1|diabetes 1',
        'sleep_apnea': r'osa|csa|sleep apnea|sleep-disordered breathing',
        'afib': r'afib|atrial fibrillation',
        'covid': r'covid',
        'flu': r'flu|influenza',
        'chronic_fatigue': r'chonic fatigue|chronic fatigue', # Catching observed typo
        'eczema_dermatitis': r'eczema|dermatitis',
        'gastrointestinal': r'ibs|diarrhea|stomach ache|food poisoning',
        'concussion': r'concussion'
    }
    
    # Create binary indicator columns for each concept
    for concept, pattern in patterns.items():
        df_diag[f'diag_concept_{concept}'] = df_diag['diagnosis_clean'].str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)
        
    # Calculate an approximate comorbidity weight 
    # (Assigning higher arbitrary weights to chronic cardiovascular/metabolic conditions)
    df_diag['comorbidity_weight'] = (
        df_diag['diag_concept_hypertension'] * 2 +
        df_diag['diag_concept_diabetes_t2'] * 2 +
        df_diag['diag_concept_diabetes_t1'] * 2 +
        df_diag['diag_concept_afib'] * 3 +
        df_diag['diag_concept_sleep_apnea'] * 1
    )
    
    # Aggregate to the participant level
    agg_funcs = {col: 'max' for col in df_diag.columns if col.startswith('diag_concept_')}
    agg_funcs['comorbidity_weight'] = 'sum'
    agg_funcs['date'] = 'count' # Proxy for total hospital visits
    
    df_patient_diagnoses = df_diag.groupby('ID').agg(agg_funcs).reset_index()
    df_patient_diagnoses = df_patient_diagnoses.rename(columns={'date': 'total_diagnostic_visits'})
    
    return df_patient_diagnoses