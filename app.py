import streamlit as st
import pandas as pd
import joblib

# Model load karein - Path check kar lijiye notebook ke hisaab se
model = joblib.load('model/churn_model.pkl')

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Prediction App")
st.markdown("Through this app, you can check whether a customer will leave your service or not.")

# Layout ke liye columns ka use
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen (1=Yes, 0=No)", [0, 1])

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Predict Button
if st.button("🔍 Predict Churn Risk"):
    # Input dictionary (Ensure columns match your training data exactly)
    input_data = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': 'No', 'Dependents': 'No',
        'tenure': tenure, 'PhoneService': 'Yes', 'MultipleLines': 'No', 
        'InternetService': internet_service, 'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': contract, 'PaperlessBilling': 'Yes',
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    # 1. Convert to DataFrame
    data_df = pd.DataFrame([input_data])
    
    # 2. Prediction (Using 'data_df' instead of 'input_df')
    prediction = model.predict(data_df)
    probability = model.predict_proba(data_df)[0][1] 

    st.divider()
    st.subheader("Prediction Result:")

    # 3. User-Friendly Output
    if prediction[0] == 1:
        st.error(f"### ⚠️ High Risk: Customer likely to CHURN")
        st.progress(probability) # Visual bar
        st.write(f"Model is **{probability:.2%}** confident that this customer will leave.")
    else:
        st.success(f"### ✅ Low Risk: Customer likely to STAY")
        st.progress(1 - probability) # Visual bar
        st.write(f"Model is **{1 - probability:.2%}** confident that this customer will stay.")

    # Business Insight Note
    st.info("💡 *Tip*: Higher churn has been observed among users with monthly contracts and Fiber optic internet service.")