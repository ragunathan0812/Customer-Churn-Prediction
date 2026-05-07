import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import os
def evaluate():
    print("Loading the saved pipeline artifact...")
    model_path = os.path.join(os.path.dirname(__file__), '../churn_model.pkl')
    pipeline = joblib.load(model_path)
    
    print("Loading the test dataset...")
    # Loading exactly as train_model did to retrieve the same test set:
    data_path = os.path.join(os.path.dirname(__file__), '../Data/processed/cleaned.csv')
    df = pd.read_csv(data_path)
    
    columns_to_drop = ['Churn', 'customerID']
    if 'Unnamed: 0' in df.columns:
        columns_to_drop.append('Unnamed: 0')
        
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df['Churn']
    
    # Ideally standard practice implies saving the test set, but for verification we recreate the exact same split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scoring the model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Evaluation Results ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

if __name__ == '__main__':
    evaluate()
