# apps/streamlit_app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Demand Forecast Prototype", layout="wide")
st.title("Demand Forecasting Prototype")

data_path = Path("data/processed/processed_purchases.csv")
if not data_path.exists():
    st.error("Processed data not found. Run ETL first (src/etl.py).")
else:
    df = pd.read_csv(data_path, parse_dates=["date"])
    st.subheader("Data Sample")
    st.dataframe(df.head())

    prod = st.selectbox("Select product", df["product"].unique())
    prod_df = df[df["product"] == prod].copy()
    ts = prod_df.groupby("date").agg({"quantity":"sum"}).asfreq("D").fillna(0)
    st.subheader("Quantity time series (7-day MA)")
    st.line_chart(ts["quantity"].rolling(window=7).mean())

    st.subheader("Simple 7-day moving average forecast (next 7 days)")
    last_vals = ts["quantity"].tail(7).values
    forecast = [float(last_vals.mean())]*7
    st.write(forecast)
