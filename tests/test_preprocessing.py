import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_preprocessing import clean_raw_data, build_preprocessor

def test_clean_raw_data():
    """Test that data cleaning handles TotalCharges and SeniorCitizen correctly."""
    data = {
        'customerID': ['1', '2'],
        'gender': ['Male', 'Female'],
        'SeniorCitizen': [0, 1],
        'tenure': [10, 20],
        'MonthlyCharges': [50.0, 70.0],
        'TotalCharges': ['500.0', ' '],  # One valid, one empty string
        'Churn': ['No', 'Yes']
    }
    df = pd.DataFrame(data)
    
    df_clean = clean_raw_data(df)
    
    # Check TotalCharges is numeric
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])
    # Check blank space was filled (MonthlyCharges * tenure = 70 * 20 = 1400)
    assert df_clean.loc[1, 'TotalCharges'] == 1400.0
    # Check SeniorCitizen is string (for categorical encoding)
    assert df_clean['SeniorCitizen'].dtype == object
    # Check customerID is dropped
    assert 'customerID' not in df_clean.columns

def test_build_preprocessor():
    """Test that the preprocessor can be fit and transform data."""
    data = {
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': ['0', '1', '0'],
        'tenure': [10, 20, 30],
        'MonthlyCharges': [50.0, 70.0, 60.0],
        'TotalCharges': [500.0, 1400.0, 1800.0]
    }
    df = pd.DataFrame(data)
    
    preprocessor = build_preprocessor(df)
    preprocessor.fit(df)
    transformed = preprocessor.transform(df)
    
    # Check shape (numeric columns: tenure, MonthlyCharges, TotalCharges; categorical: gender, SeniorCitizen)
    # gender -> 1 col (drop first), SeniorCitizen -> 1 col (drop first)
    # Total = 3 + 1 + 1 = 5
    assert transformed.shape[1] == 5
