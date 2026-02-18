import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn EDA Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df

df = load_data()

st.title("📊 Customer Churn Analytics Dashboard")

# Sidebar filter
st.sidebar.header("Filter Options")

contract_filter = st.sidebar.multiselect(
    "Select Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

df_filtered = df[df["Contract"].isin(contract_filter)]

# Layout columns
col1, col2 = st.columns(2)

# -----------------------
# 1️⃣ Churn Distribution
# -----------------------
with col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df_filtered, ax=ax)
    ax.set_xticklabels(["No", "Yes"])
    st.pyplot(fig)

# -----------------------
# 2️⃣ Churn by Contract
# -----------------------
with col2:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df_filtered, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

# -----------------------
# 3️⃣ Tenure Distribution
# -----------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("Tenure vs Churn")
    fig, ax = plt.subplots()
    sns.histplot(data=df_filtered, x="tenure", hue="Churn", bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# -----------------------
# 4️⃣ Monthly Charges
# -----------------------
with col4:
    st.subheader("Monthly Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df_filtered, ax=ax)
    st.pyplot(fig)

# -----------------------
# 5️⃣ Dynamic Feature Explorer
# -----------------------

st.subheader("🔎 Explore Churn by Feature")

selected_feature = st.selectbox(
    "Choose a feature",
    ["InternetService", "PaymentMethod", "gender", "SeniorCitizen"]
)

fig, ax = plt.subplots()
sns.countplot(x=selected_feature, hue="Churn", data=df_filtered, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)
