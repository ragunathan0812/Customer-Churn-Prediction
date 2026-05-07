import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import shap

# Set page config
st.set_page_config(
    page_title="ChurnGuard AI | Customer Retention Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .churn-risk-high {
        color: #d9534f;
        font-weight: bold;
    }
    .churn-risk-low {
        color: #5cb85c;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/cleaned.csv")
    return df

@st.cache_resource
def load_model():
    model_path = "models/churn_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_explainer():
    explainer_path = "models/shap_explainer.joblib"
    if os.path.exists(explainer_path):
        return joblib.load(explainer_path)
    return None

# --- Load Resources ---
df = load_data()
model = load_model()
explainer_data = load_explainer()

# --- Sidebar ---
st.sidebar.image("reports/banner.png", use_column_width=True)
st.sidebar.title("🛡️ ChurnGuard AI")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigate", ["Strategic Overview", "Customer Predictor", "Model Insights"])

# --- Tab 1: Strategic Overview ---
if menu == "Strategic Overview":
    st.title("📊 Strategic Churn Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_customers = len(df)
    churn_rate = (df['Churn'] == 1).mean() * 100
    avg_monthly_charges = df['MonthlyCharges'].mean()
    total_revenue_at_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum()

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Average Churn Rate", f"{churn_rate:.1f}%", delta="-1.2%", delta_color="inverse")
    col3.metric("Avg. Monthly Charge", f"${avg_monthly_charges:.2f}")
    col4.metric("Revenue at Risk", f"${total_revenue_at_risk:,.0f}", delta_color="off")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Churn by Contract Type")
        fig_contract = px.histogram(df, x="Contract", color="Churn", 
                                  barmode="group", color_discrete_sequence=["#636EFA", "#EF553B"],
                                  labels={"Churn": "Churned?"})
        st.plotly_chart(fig_contract, use_container_width=True)

    with col_right:
        st.subheader("Tenure vs Monthly Charges")
        fig_scatter = px.scatter(df.sample(min(1000, len(df))), x="tenure", y="MonthlyCharges", color="Churn",
                               opacity=0.6, color_discrete_sequence=["#636EFA", "#EF553B"],
                               title="Sample of 1,000 Customers")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Top Risk Factors (Segment Analysis)")
    selected_feat = st.selectbox("Analyze Segment by:", ["InternetService", "PaymentMethod", "PaperlessBilling", "StreamingTV"])
    fig_segment = px.sunburst(df, path=[selected_feat, 'Contract', 'Churn'], color='MonthlyCharges',
                             color_continuous_scale='RdBu')
    st.plotly_chart(fig_segment, use_container_width=True)

# --- Tab 2: Customer Predictor ---
elif menu == "Customer Predictor":
    st.title("🎯 Real-time Churn Predictor")
    st.markdown("Enter customer details to assess churn risk and view personalized retention insights.")

    if model is None:
        st.error("Model not found! Please run the training pipeline first.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior Citizen", ["0", "1"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.subheader("Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            st.subheader("Billing & Contract")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            total = st.number_input("Total Charges", 0.0, 10000.0, monthly * tenure)

        # Prediction Logic
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone],
            'MultipleLines': [multiple],
            'InternetService': [internet],
            'OnlineSecurity': [security],
            'OnlineBackup': [backup],
            'DeviceProtection': [protection],
            'TechSupport': [support],
            'StreamingTV': [tv],
            'StreamingMovies': [movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless],
            'PaymentMethod': [payment],
            'MonthlyCharges': [monthly],
            'TotalCharges': [total]
        })

        if st.button("Calculate Churn Risk"):
            prob = model.predict_proba(input_data)[0][1]
            st.markdown("---")
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.subheader("Risk Score")
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability %"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#EF553B" if prob > 0.5 else "#636EFA"},
                        'steps' : [
                            {'range': [0, 40], 'color': "lightgreen"},
                            {'range': [40, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "pink"}],
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with c2:
                st.subheader("Explanation (SHAP)")
                if explainer_data:
                    # Preprocess data for SHAP
                    preprocessor = model.named_steps['preprocessor']
                    input_transformed = preprocessor.transform(input_data)
                    
                    explainer = explainer_data['explainer']
                    feature_names = explainer_data['feature_names']
                    
                    try:
                        shap_vals = explainer.shap_values(input_transformed)
                        
                        # Handle different formats (KernelExplainer returns list for binary, TreeExplainer might return single array)
                        if isinstance(shap_vals, list):
                            # For binary classification, take the values for class 1 (churn)
                            # SHAP 0.45+ often returns [class0, class1]
                            s_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
                        else:
                            s_vals = shap_vals

                        # Handle expected_value format
                        ev = explainer.expected_value
                        if isinstance(ev, (list, np.ndarray)):
                            ev = ev[1] if len(ev) > 1 else ev[0]

                        # Create a SHAP Explanation object for the bar plot
                        explanation = shap.Explanation(
                            values=s_vals[0] if len(s_vals.shape) > 1 else s_vals, 
                            base_values=ev, 
                            data=input_transformed[0], 
                            feature_names=feature_names
                        )
                        
                        fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
                        shap.plots.bar(explanation, show=False)
                        st.pyplot(fig_shap)
                    except Exception as e:
                        st.error(f"Error generating SHAP plot: {e}")
                else:
                    st.info("SHAP explainer data not found. Re-run training to enable.")

# --- Tab 3: Model Insights ---
elif menu == "Model Insights":
    st.title("🧪 Model Performance & Insights")
    
    if os.path.exists("reports/training.log"):
        st.subheader("Recent Training Logs")
        with open("reports/training.log", "r") as f:
            st.code(f.readlines()[-20:])
    
    st.subheader("Global Feature Importance")
    if model:
        try:
            # Extract feature importance from XGBoost
            xgb_model = model.named_steps['model']
            preprocessor = model.named_steps['preprocessor']
            feature_names = preprocessor.get_feature_names_out()
            
            importances = xgb_model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
            
            fig_imp = px.bar(feat_df.head(15), x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Developed by Ragunathan | Portfolio Project 2026")
