# Predicting Customer Trend

## What this repo contains
- ETL: `src/etl.py` — make features and processed CSV
- Train: `src/train_model.py` — trains RandomForest and saves model/encoder in `models/`
- API: `api/app.py` — Flask app to serve predictions
- Streamlit prototype: `apps/streamlit_app.py` — simple demand forecast UI
- Data: `data/raw/purchases.csv` (raw), `data/processed/processed_purchases.csv` (processed)
- Models: saved in `models/`

## Quickstart
1. Create environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
