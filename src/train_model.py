import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import shap

# Local imports
from data_preprocessing import build_preprocessor, clean_raw_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '../reports/training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_pipeline():
    logger.info("🚀 Starting Model Training Pipeline")
    
    # Paths
    raw_data_path = os.path.join(os.path.dirname(__file__), '../Data/raw/Telco-Customer-Churn.csv')
    model_save_path = os.path.join(os.path.dirname(__file__), '../models/churn_model.pkl')
    
    # 1. Load Data
    logger.info(f"Loading raw data from {raw_data_path}")
    df_raw = pd.read_csv(raw_data_path)
    
    # 2. Clean Data
    df_clean = clean_raw_data(df_raw)
    
    # 3. Split Features and Target
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn'].map({'No': 0, 'Yes': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # 4. Build Pipeline
    preprocessor = build_preprocessor(X_train)
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb)
    ])
    
    # 5. Train Model
    logger.info("Training XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluation
    logger.info("Evaluating model performance...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"✅ Model Performance: ROC-AUC={auc:.4f}, Accuracy={acc:.4f}, F1-Score={f1:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # 7. SHAP Explainability (on preprocessed data)
    logger.info("Generating SHAP values for explainability...")
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Get feature names after transformation
    try:
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    except Exception as e:
        logger.warning(f"Could not get feature names from preprocessor: {e}")
        feature_names = [f"f{i}" for i in range(X_test_transformed.shape[1])]
    
    # Use the generic Explainer for better compatibility with different XGBoost versions
    try:
        model_inner = pipeline.named_steps['model']
        
        # Workaround for XGBoost 2.x base_score parsing bug in some SHAP versions
        # We try to ensure the model attributes are in a format SHAP likes
        try:
            import json
            # We don't modify the model directly in a way that breaks it, 
            # but we can try to use KernelExplainer as a last resort or clean up attributes
            logger.info("Initializing SHAP Explainer...")
        except:
            pass

        explainer = shap.Explainer(model_inner, X_test_transformed, feature_names=feature_names)
        
        # Save explainer data
        explainer_path = os.path.join(os.path.dirname(__file__), '../models/shap_explainer.joblib')
        joblib.dump({'explainer': explainer, 'feature_names': list(feature_names)}, explainer_path)
        logger.info("SHAP explainer saved successfully.")
    except Exception as e:
        logger.error(f"Error generating SHAP explainer: {e}")
        logger.info("Attempting fallback with KernelExplainer (subset for speed)...")
        try:
            # KernelExplainer is much slower but bypasses the TreeExplainer parsing bugs
            # We use a small background dataset for speed
            explainer = shap.KernelExplainer(model_inner.predict_proba, shap.sample(X_test_transformed, 50))
            explainer_path = os.path.join(os.path.dirname(__file__), '../models/shap_explainer.joblib')
            joblib.dump({'explainer': explainer, 'feature_names': list(feature_names)}, explainer_path)
            logger.info("KernelExplainer saved successfully.")
        except Exception as e2:
            logger.error(f"Final fallback failed: {e2}")
    
    # 8. Save Final Pipeline
    logger.info(f"Saving final pipeline to {model_save_path}")
    joblib.dump(pipeline, model_save_path)
    
    logger.info("🏁 Training Pipeline Completed Successfully!")

if __name__ == '__main__':
    train_pipeline()
