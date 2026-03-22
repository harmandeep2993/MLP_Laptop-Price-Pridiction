import pickle

import numpy as np
import pandas as pd
import streamlit as st

INR_TO_USD_RATE = 83

# ── Page configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
)

# ── Load model artefacts ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/pipe.pkl", "rb") as f:
        pipe = pickle.load(f)
    with open("models/df.pkl", "rb") as f:
        df = pickle.load(f)
    return pipe, df


pipe, df = load_artifacts()

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("💻 Laptop Price Predictor")
st.markdown(
    "Select your laptop specifications below and click **Predict Price** "
    "to get an estimated price (in INR)."
)
st.divider()

# ── Sidebar instructions ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown(
        """
        1. Choose your desired laptop specifications from the drop-downs.
        2. Enter the weight and click **Predict Price**.
        3. The predicted price will appear below the button.
        """
    )
    st.markdown("---")
    st.markdown("**Model:** Random Forest Regressor")
    st.markdown("**Accuracy:** ~87% R²")

# ── Input form ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", sorted(df["Company"].unique()))
    laptop_type = st.selectbox("Type", sorted(df["TypeName"].unique()))
    processor = st.selectbox("CPU", sorted(df["Processor"].unique()))
    graphics = st.selectbox("Graphics Card", sorted(df["VC"].unique()))
    operating_system = st.selectbox("Operating System", sorted(df["OS"].unique()))

with col2:
    ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
    hdd = st.selectbox("HDD (GB)", sorted(df["HDD"].unique()))
    ssd = st.selectbox("SSD (GB)", sorted(df["SSD"].unique()))
    inches = st.selectbox("Screen Size (inches)", sorted(df["Inches"].unique()))
    ppi = st.selectbox("Pixels Per Inch (PPI)", sorted(df["PPI"].unique()))

col3, col4 = st.columns(2)
with col3:
    touchscreen = st.selectbox("Touchscreen", ["No (0)", "Yes (1)"])
    touchscreen_val = 1 if touchscreen.startswith("Yes") else 0
with col4:
    ips_display = st.selectbox("IPS Display", ["No (0)", "Yes (1)"])
    ips_val = 1 if ips_display.startswith("Yes") else 0

weight = st.number_input(
    "Weight (kg)", min_value=0.5, max_value=10.0, value=1.5, step=0.1
)

st.divider()

# ── Prediction ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
    # Feature order must match the training data column order:
    # Company, OS, Processor, TypeName, VC, Ram, TS, Weight, HDD, SSD, IPS, Inches, PPI
    query = pd.DataFrame(
        [[
            company,
            operating_system,
            processor,
            laptop_type,
            graphics,
            ram,
            touchscreen_val,
            weight,
            hdd,
            ssd,
            ips_val,
            inches,
            ppi,
        ]],
        columns=df.columns,
    )

    predicted_price = int(np.exp(pipe.predict(query)[0]))

    st.success(
        f"💰 Estimated Price: **₹ {predicted_price:,}** INR  "
        f"(≈ $ {int(predicted_price / INR_TO_USD_RATE):,} USD)"
    )
    st.caption(
        "Note: This is a model estimate based on historical laptop data. "
        "Actual prices may vary."
    )