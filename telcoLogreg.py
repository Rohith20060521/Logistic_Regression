import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background-color: #1f2937;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
}
.metric-card h2 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>Customer Churn Prediction Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown("---")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Churn Probability Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("TelcoCom.csv")

df = load_data()

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# --------------------------------------------------
# CHURN DISTRIBUTION (COUNT PLOT)
# --------------------------------------------------
st.subheader("Churn Distribution")

fig_count, ax_count = plt.subplots(figsize=(5, 4))
sns.countplot(x="Churn", data=df, ax=ax_count)
ax_count.set_xlabel("Churn")
ax_count.set_ylabel("Count")
st.pyplot(fig_count)

st.markdown("---")

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

# --------------------------------------------------
# METRIC CARDS
# --------------------------------------------------
st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"<div class='metric-card'><h2>{acc:.2f}</h2>Accuracy</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h2>{prec:.2f}</h2>Precision</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h2>{rec:.2f}</h2>Recall</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h2>{f1:.2f}</h2>F1 Score</div>", unsafe_allow_html=True)

st.markdown("---")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("Confusion Matrix")

fig_cm, ax_cm = plt.subplots(figsize=(4,3))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="viridis",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"],
    ax=ax_cm
)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

st.markdown("---")

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------
st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.2f}"),
    use_container_width=True
)

# --------------------------------------------------
# BUSINESS INSIGHTS
# --------------------------------------------------
st.subheader("Business Insights")

col5, col6 = st.columns(2)
col5.success(f"Churn customers correctly identified (TP): {TP}")
col6.error(f"Non-churn misclassified as churn (FP): {FP}")

st.markdown("---")





# --------------------------------------------------
# PREDICT CUSTOMER CHURN (3 FEATURES)
# --------------------------------------------------
st.subheader("Predict Customer Churn")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0.0,
        max_value=float(df["tenure"].max()),
        value=12.0
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=float(df["MonthlyCharges"].max()),
        value=70.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=float(df["TotalCharges"].max()),
        value=1000.0
    )

if st.button("Predict Churn"):
    # Create input dataframe with ALL FEATURES
    input_data = X.copy().iloc[:1] * 0

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict probability
    churn_prob = model.predict_proba(input_scaled)[0][1]

    if churn_prob >= threshold:
        st.error(f"Customer is likely to churn (Probability: {churn_prob:.2f})")
    else:
        st.success(f"Customer is unlikely to churn (Probability: {churn_prob:.2f})")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center;'>Built with Streamlit and Logistic Regression</p>",
    unsafe_allow_html=True
)
