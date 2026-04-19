import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, roc_auc_score

# Import custom modules
from data_processing import load_raw_data, build_merged_datasets
from features import create_preprocessor

def main():
    # 1. Load and Process Data
    train_p, test_p, train_w, test_w = load_raw_data(data_dir='../data/raw/')
    train_full, test_full = build_merged_datasets(train_p, test_p, train_w, test_w)
    
    # 2. Separate Features and Target
    X = train_full.drop(columns=['ID', 'outcome'])
    y = train_full['outcome']
    X_test_final = test_full.drop(columns=['ID'])
    test_ids = test_full['ID']
    
    # 3. Build the Modeling Pipeline
    preprocessor = create_preprocessor(X)
    
    # HistGradientBoosting is fast, interpretable, and natively handles complex tabular data
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            learning_rate=0.05, 
            max_iter=200, 
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # 4. Local Validation (to ensure the model is actuarially sound)
    print("Splitting data for local validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training validation model...")
    model_pipeline.fit(X_train, y_train)
    val_preds = model_pipeline.predict_proba(X_val)[:, 1]
    
    print(f"Validation Log-Loss: {log_loss(y_val, val_preds):.4f}")
    print(f"Validation ROC-AUC: {roc_auc_score(y_val, val_preds):.4f}")
    
    # 5. Train Final Model on 100% of the Training Data
    print("Retraining model on full dataset for final submission...")
    model_pipeline.fit(X, y)
    
    # 6. Generate Predictions on Blind Test Set
    test_preds = model_pipeline.predict_proba(X_test_final)[:, 1]
    
    # 7. Export Scoring Key (Must strictly follow naming conventions)
    submission_df = pd.DataFrame({
        'ID': test_ids,
        'predicted_probability': test_preds # Must be continuous value between 0.0 and 1.0
    })
    
    csv_filename = '../team_6_scoring_key.csv'
    submission_df.to_csv(csv_filename, index=False)
    print(f"Scoring key saved to {csv_filename}")
    
    # 8. Export Model Artifact
    pkl_filename = '../team_6_model_artifact.pkl'
    with open(pkl_filename, 'wb') as f:
        pickle.dump(model_pipeline, f)
    print(f"Model artifact saved to {pkl_filename}")

if __name__ == "__main__":
    main()