# 🚀 Customer Churn Prediction System

![Banner](https://raw.githubusercontent.com/ragunathan0812/Customer-Churn-Prediction/main/reports/banner.png)

## 📊 Executive Summary
This project delivers an end-to-end Machine Learning solution to predict customer churn for a telecommunications provider. By identifying high-risk customers, the system enables proactive retention strategies, potentially saving the company significant revenue. The final model achieves an **ROC-AUC of 0.84** and provides **explainable predictions** using SHAP.

---

## 🛠️ Tech Stack
- **Languages**: Python 3.10+
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib, Plotly
- **Machine Learning**: Scikit-Learn, XGBoost
- **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations)
- **Deployment**: Streamlit
- **Testing**: Pytest

---

## 📈 Key Insights from EDA
- **Contract Type**: Customers with Month-to-month contracts are significantly more likely to churn compared to those with one or two-year contracts.
- **Tenure**: New customers (tenure < 6 months) represent the highest churn risk.
- **Services**: Fiber optic internet users exhibit higher churn rates, suggesting potential service quality or pricing issues.
- **Payment Method**: Electronic check users have a higher propensity to churn.

---

## 🤖 Modeling & Methodology
1.  **Preprocessing**: Robust pipeline handling missing values, categorical encoding (OneHot), and feature scaling.
2.  **Model Selection**: Evaluated multiple algorithms (Random Forest, Logistic Regression, XGBoost).
3.  **Optimization**: Hyperparameter tuning using GridSearchCV.
4.  **Explainability**: Integrated SHAP to explain individual feature contributions to churn risk.

---

## 🌐 Interactive Dashboard
The project includes a premium Streamlit dashboard that allows stakeholders to:
- Explore real-time churn metrics.
- Visualize feature importance across the entire customer base.
- **Predict Churn Risk**: Input individual customer data to get a churn probability and a SHAP-based explanation.

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/ragunathan0812/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
```bash
python src/train_model.py
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard.py
```

---

## 📁 Project Structure
```text
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned and engineered data
├── models/                 # Saved models and scalers
├── notebooks/              # Jupyter notebooks for EDA
├── reports/                # Generated plots and performance reports
├── src/                    # Modular Python scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
├── tests/                  # Unit tests
├── README.md               # Project documentation
└── requirements.txt        # Dependency list
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author
**Ragunathan**
- [LinkedIn](https://www.linkedin.com/in/ragunathan0812/)
- [Portfolio](https://yourportfolio.com)
