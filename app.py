import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page configuration
# ==========================================

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# Load model
# ==========================================

artifact = joblib.load("models/churn_model.joblib")

model = artifact["model"]
threshold = artifact["threshold"]


# ==========================================
# Header
# ==========================================

st.title("📊 Customer Churn Predictor")

st.write(
    "Enter customer information to estimate the probability "
    "of customer churn."
)

st.divider()


# ==========================================
# Customer information
# ==========================================

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)


with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure (months)",
        min_value=1,
        max_value=72,
        value=12
    )


with col2:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


with col3:

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# ==========================================
# Contract and billing
# ==========================================

st.subheader("Contract & Billing")

col1, col2, col3 = st.columns(3)


with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


with col3:

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=18.25,
        max_value=118.75,
        value=70.0,
        step=0.01
    )

    total_charges = st.number_input(
    "Total Charges",
    min_value=18.80,
    max_value=8684.80,
    value=float(monthly_charges * tenure),
    step=0.01
)


st.divider()

# ==========================================
# Prediction
# ==========================================

if st.button(
    "Predict Churn",
    type="primary",
    use_container_width=True
):
        # ======================================
    # Input validation
    # ======================================

    minimum_total = monthly_charges

    if total_charges < minimum_total:
        st.error(
            "Total Charges cannot be lower than Monthly Charges."
        )
        st.stop()
    customer = pd.DataFrame(
        [{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }]
    )


    # ======================================
    # Model prediction
    # ======================================

    probability = model.predict_proba(customer)[0][1]

    prediction = int(probability >= threshold)


    # ======================================
    # Results
    # ======================================

    st.subheader("Prediction")

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )


    with col2:

        if probability < 0.30:
            risk_level = "Low"
        elif probability < 0.60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        st.metric(
            "Risk Level",
            risk_level
        )


    with col3:

        prediction_text = (
            "Likely to Churn"
            if prediction == 1
            else "Likely to Stay"
        )

        st.metric(
            "Prediction",
            prediction_text
        )


    # ======================================
    # Probability bar
    # ======================================

    st.write("Churn probability")

    st.progress(
        min(probability, 1.0)
    )


    st.caption(
        f"Classification threshold: {threshold:.0%}"
    )


    # ======================================
    # Explanation
    # ======================================

    if prediction == 1:

        st.warning(
            "The predicted churn probability is above the "
            "classification threshold. This customer is "
            "classified as likely to churn."
        )

    else:

        st.success(
            "The predicted churn probability is below the "
            "classification threshold. This customer is "
            "classified as likely to stay."
        )


# ==========================================
# Model information
# ==========================================

st.divider()

st.subheader("About the Model")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        "Model",
        "Logistic Regression"
    )

with info_col2:
    st.metric(
        "ROC-AUC",
        "0.836"
    )

with info_col3:
    st.metric(
        "Recall",
        "75.9%"
    )

st.caption(
    "The model was trained on the Telco Customer Churn dataset. "
    "The classification threshold was selected using cross-validation "
    "and evaluated on a separate test set."
)