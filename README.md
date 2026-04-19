# Pacific Life AI Actuary Challenge 2026 - Team 6

## Repository Structure
- `data/raw/`: Place the provided competition CSVs here before running.
- `src/data_processing.py`: Handles data loading and wearable time-series aggregation.
- `src/features.py`: Scikit-learn preprocessing pipelines for missing values and encodings.
- `src/train.py`: Main execution script that trains the Gradient Boosting model.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt