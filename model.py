# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 22:16:53 2025

@author: HP
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_model(df):
    # Prepare features and target variable
    X = df[["overs", "runs", "wickets", "runs_last_5", "wickets_last_5"]]
    y = df["total"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest Regressor
    model = RandomForestRegressor(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    
    return model, error
