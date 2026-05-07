import pandas as pd
import numpy as np
import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a scikit-learn ColumnTransformer for preprocessing.
    
    Args:
        X: Feature dataframe to determine column types.
        
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    logger.info("Initializing preprocessing pipeline...")
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    
    logger.info(f"Numeric features: {list(numeric_cols)}")
    logger.info(f"Categorical features: {list(categorical_cols)}")

    # Numeric pipeline: Impute medians and scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: Impute most frequent and encode
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    logger.info("Preprocessing pipeline built successfully.")
    return preprocessor

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning on the raw Telco dataset.
    
    Args:
        df: Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    logger.info("Starting data cleaning process...")
    df_clean = df.copy()
    
    # 1. Handle TotalCharges conversion and imputation
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        mask = df_clean['TotalCharges'].isna()
        if mask.any():
            logger.info(f"Imputing {mask.sum()} missing TotalCharges values.")
            df_clean.loc[mask, 'TotalCharges'] = df_clean.loc[mask, 'MonthlyCharges'] * df_clean.loc[mask, 'tenure']
            
    # 2. Convert SeniorCitizen to object for categorical treatment
    if 'SeniorCitizen' in df_clean.columns:
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].astype(str)
        
    # 3. Drop unique identifiers
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    logger.info("Data cleaning completed.")
    return df_clean
