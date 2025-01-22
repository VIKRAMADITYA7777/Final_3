# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:17:43 2025

@author: HP
"""

import streamlit as st
import numpy as np
import pandas as pd  # Added import for DataFrame conversion
from data_loader import load_data
from model import train_model

# Load data and train model
df = load_data()
model, error = train_model(df)

# Streamlit App
st.title("Cricket Score Predictor")

st.sidebar.header("Input Match Details")

overd = st.sidebar.slider("Overs Bowled", min_value=0, max_value=50, step=1, value=20)
runs_till_now = st.sidebar.slider("Runs Scored Till Now", min_value=0, max_value=500, step=1, value=120)
wickets = st.sidebar.slider("Wickets Lost", min_value=0, max_value=10, step=1, value=2)
runs_last_5 = st.sidebar.slider("Runs Scored in Last 5 Overs", min_value=0, max_value=100, step=1, value=30)
wickets_last_5 = st.sidebar.slider("Wickets Lost in Last 5 Overs", min_value=0, max_value=10, step=1, value=1)

# Prediction Input
input_data = np.array([[overd, runs_till_now, wickets, runs_last_5, wickets_last_5]])

# Ensure feature names match training data
def feature_array_to_dataframe(input_array):
    return pd.DataFrame(input_array, columns=["overs", "runs", "wickets", "runs_last_5", "wickets_last_5"])

input_df = feature_array_to_dataframe(input_data)
predicted_score = model.predict(input_df)[0]

st.write("### Predicted Total Score:", round(predicted_score, 2))

st.write("### Model Evaluation")
st.write(f"Mean Absolute Error: {error:.2f}")

st.write("### Training Data Sample")
st.dataframe(df.head())

st.write("### Instructions")
st.markdown(
    """This app predicts the total score in a cricket match based on:
    - Overs bowled
    - Runs scored till the given over
    - Wickets lost
    - Runs scored in the last 5 overs
    - Wickets lost in the last 5 overs

    **Project Structure:**
    - `data_loader.py`: Contains the function to load and preprocess data
    - `model.py`: Contains the machine learning model training and evaluation logic
    - `app.py`: Streamlit app to provide an interactive user interface

    **Steps to Run the Project:**
    1. Place your dataset file `re_cricketdata.csv` in the project directory.
    2. Install the required libraries: `streamlit`, `scikit-learn`, `pandas`, `numpy`
    3. Save the files in a single directory.
    4. Run the command `streamlit run app.py` in your terminal from the directory.

    **Enhancements You Can Add:**
    - Use additional features like pitch type and player form.
    - Experiment with advanced machine learning models or hyperparameter tuning.
    - Add data visualization for better insights.
    """
)
