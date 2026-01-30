# 📊 Customer Churn Prediction System

This project is an end-to-end Machine Learning solution designed to identify customers at risk of leaving a service (churning). By predicting churn, businesses can take proactive measures to retain customers and protect revenue.

## 🚀 Project Overview
- **Objective:** Predict customer churn using historical data to minimize business loss.
- **Key Results:** Achieved an **ROC-AUC of 0.84** using Logistic Regression.
- **Business Impact:** Identified **286 potential churners**, saving an estimated **₹85,800 monthly revenue**.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, Joblib, SHAP
- **Dashboard:** Streamlit (for real-time predictions)

## 📈 Key Features
- **Automated Pipeline:** Integrated preprocessing (Scaling & Encoding) and modeling into a single pipeline.
- **Model Explainability:** Used **SHAP** values to identify that `tenure` and `MonthlyCharges` are the biggest churn drivers.
- **Interactive UI:** A web app where users can input customer details and get instant churn risk probabilities.

## ⚙️ How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the app: `streamlit run app.py`