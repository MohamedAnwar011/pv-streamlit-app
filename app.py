import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data from the Excel file in the repo
@st.cache_data
def load_data():
    return pd.read_excel("phase1-1aa.xlsx", sheet_name="Data")

# Load and preprocess data
data = load_data()
data_cleaned = data.select_dtypes(include=["number"])

# Features and target
X = data_cleaned.drop("PV results", axis=1)
y = data_cleaned["PV results"]

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Streamlit app UI
st.title("☀️ PV Result Prediction App")
st.write("Enter building parameters to estimate PV energy output.")

# Get column names used for features
feature_names = X.columns

# User inputs
user_inputs = []
for col in feature_names:
    if "area" in col.lower() or "angle" in col.lower() or "height" in col.lower():
        value = st.number_input(f"{col}", min_value=0.0, format="%.2f")
    else:
        value = st.number_input(f"{col}", min_value=0)
    user_inputs.append(value)

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs], columns=feature_names)

# Scale and predict
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)

# Show result
st.subheader("🔋 Predicted PV Result (kWh/year):")
st.success(f"{prediction[0]:,.2f}")

st.markdown("---")
st.caption("Model: Linear Regression trained on your uploaded dataset.")
