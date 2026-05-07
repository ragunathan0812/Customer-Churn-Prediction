import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, 
    classification_report, auc
)

def evaluate_model():
    print("Evaluating Model Performance...")
    
    # Paths
    model_path = os.path.join(os.path.dirname(__file__), '../models/churn_model.pkl')
    data_path = os.path.join(os.path.dirname(__file__), '../Data/processed/cleaned.csv')
    report_dir = os.path.join(os.path.dirname(__file__), '../reports')
    
    # 1. Load Model and Data
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # 2. Generate Predictions
    y_pred = pipeline.predict(X)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    
    # 3. Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(report_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(report_dir, 'roc_curve.png'))
    plt.close()
    
    # 5. Classification Report
    report = classification_report(y, y_pred)
    with open(os.path.join(report_dir, 'performance_report.txt'), 'w') as f:
        f.write("Customer Churn Prediction - Model Performance Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"ROC-AUC Score: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        
    print(f"Evaluation completed. Reports saved to {report_dir}")

if __name__ == '__main__':
    evaluate_model()
