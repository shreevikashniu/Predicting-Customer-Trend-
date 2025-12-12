# src/etl.py
import pandas as pd
from pathlib import Path

def run_etl(raw_path: str, processed_path: str):
    """
    Basic ETL and feature engineering:
    - read raw csv with purchase records
    - create date-based features (month, dayofweek)
    - aggregate customer-level metrics (orders_count, total_spent, total_quantity)
    - merge back to transactions to create per-record feature set
    """
    raw = Path(raw_path)
    proc = Path(processed_path)
    proc.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw, parse_dates=['date'])
    # ensure columns exist
    required = {'order_id','customer_id','date','product','quantity','amount'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Missing required cols: {required - set(df.columns)}")

    # basic cleaning
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

    # date features
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day

    # customer aggregations (simple RFM-ish)
    agg = df.groupby('customer_id').agg(
        orders_count = ('order_id','count'),
        total_spent = ('amount','sum'),
        total_quantity = ('quantity','sum'),
        last_purchase = ('date','max'),
        first_purchase = ('date','min')
    ).reset_index()

    # recency (days since last purchase) - relative to max date in data
    max_date = df['date'].max()
    agg['recency_days'] = (max_date - pd.to_datetime(agg['last_purchase'])).dt.days

    # merge back features to transaction-level data
    merged = df.merge(agg, on='customer_id', how='left')

    # ensure label exists (label_repeat_30d) — if not, keep as-is
    if 'label_repeat_30d' not in merged.columns:
        # keep existing or create placeholder 0
        merged['label_repeat_30d'] = merged.get('label_repeat_30d', 0)

    merged.to_csv(proc, index=False)
    print(f"ETL complete. Wrote {len(merged)} rows to {proc}")

if __name__ == "__main__":
    run_etl("../data/raw/purchases.csv", "../data/processed/processed_purchases.csv")
