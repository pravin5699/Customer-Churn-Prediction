
# 📊 Customer Churn Prediction System

## 📌 Project Overview

Customer churn is a critical business problem where customers stop using a company’s services, directly impacting revenue and growth. This project builds an end-to-end machine learning system to predict customer churn and provide actionable business insights to enable proactive retention strategies.
The solution is designed in a modular, production-style workflow and focuses on both predictive performance and business interpretability.

## 🎯 Business Problem

Telecom and subscription-based companies face significant revenue loss due to customer churn. The objective of this project is to:

Identify customers who are likely to churn in the near future

Understand key drivers behind churn

Enable data-driven retention strategies

Estimate potential revenue savings through targeted interventions

## 🗂 Dataset

Source: https://raw.githubusercontent.com/Nas-virat/Telco-Customer-Churn/main/Telco-Customer-Churn.csv

Records: 7,043 customers

Features: 20 customer attributes + churn label

Target Variable: Churn (Yes/No → 1/0)

##### Key feature categories:

Customer demographics

Account information

Service usage

Billing and payment details

## 🔄 Workflow

#### 1. Data Loading
💠 Loaded raw CSV data

💠 Performed initial inspection and target distribution analysis

💠 Identified data quality issues

#### 2. Data Cleaning & EDA
💠 Converted TotalCharges to numeric

💠 Handled missing values

💠 Removed non-informative identifiers

💠 Performed exploratory data analysis to identify churn patterns

##### Key EDA Insights:

💠 Month-to-month contracts have higher churn

💠 Low tenure customers are more likely to churn

💠 Higher monthly charges are associated with higher churn

#### 3. Preprocessing (Production-Style Pipeline)
💠 Separated categorical and numerical features

##### Applied:
💠 OneHotEncoding for categorical variables

💠 StandardScaler for numerical variables

##### Built modular preprocessing pipeline using:
💠 ColumnTransformer

💠 Pipeline

##### Performed stratified train-test split to handle class imbalance

#### 4. Modeling & Evaluation
#### Baseline Model: Logistic Regression

💠 ROC-AUC: 0.842

💠 Selected due to:

Strong performance

High interpretability

Suitability for production use

#### Advanced Model: Random Forest

ROC-AUC: 0.823

Did not outperform Logistic Regression

##### Final Model Selection:

Logistic Regression was selected as the final model due to better ROC-AUC and superior interpretability for business stakeholders.


## 📈 Model Performance
Logistic Regression:
0.842

Random Forest:
0.823

### Evaluation Metrics Used:
ROC-AUC (primary due to class imbalance)

Precision, Recall, F1-score

Confusion Matrix

ROC Curve comparison

## 🧠 Explainability (SHAP)
SHAP (SHapley Additive exPlanations) was used to interpret model predictions and identify key churn drivers.

#### Top Churn Drivers:
Month-to-month contract type

High monthly charges

Low customer tenure

Lack of value-added services (e.g., Online Security)

Electronic check payment method

##### This enables actionable business strategies rather than black-box predictions.


## 💰 Business Impact Simulation
A revenue impact analysis was conducted to estimate business value.

#### Assumptions:
Average revenue per customer = ₹1,000/month

Retention success rate = 30%

#### Estimated Impact:
Proactively identified high-risk churn customers

Simulated targeted retention campaigns

Estimated monthly revenue savings based on predicted churners

#### This demonstrates direct business value of the model beyond technical metrics.

## 🛠 Tools & Technologies
Python

Pandas, NumPy

Scikit-learn

Matplotlib

SHAP

Jupyter Notebook

## 🚀 Key Takeaways
Built a complete end-to-end ML pipeline

Applied production-style preprocessing and modeling

Selected model based on both performance and business 
interpretability

Used explainable AI (SHAP) to generate business insights

Connected ML results to real business KPIs (revenue impact)


## 👤 Author: PRAVIN
 Data Scienist | Machine Learning Enthusiast

## 🏆 Resume Highlight
Built an end-to-end customer churn prediction system using Logistic Regression achieving 0.84 ROC-AUC, with SHAP-based explainability and simulated revenue impact to enable data-driven retention strategies.