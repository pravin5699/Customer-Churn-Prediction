import streamlit as st
import pandas as pd
import joblib

# Model load karein
model = joblib.load('model/churn_model.pkl')

st.title("📊 Customer Churn Prediction App")

# Layout ke liye columns ka use
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", value=12)
    monthly_charges = st.number_input("Monthly Charges", value=50.0)
    total_charges = st.number_input("Total Charges", value=500.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Baaki missing features ko default values ke saath define karna
# Kyunki model ko ye saare columns ek saath chahiye
if st.button("Predict Churn"):
    input_data = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No', 
        'InternetService': internet_service, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # DataFrame banana
    data_df = pd.DataFrame([input_data])
    
    # Prediction
    prediction = model.predict(data_df)
    probability = model.predict_proba(data_df)[0][1]
    
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk! Churn Probability: {probability:.2%}")
    else:
        st.success(f"✅ Safe! Stay Probability: {1-probability:.2%}")