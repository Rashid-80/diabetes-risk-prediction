# 🩺 Diabetes Risk: Analysis and Prediction Using Machine Learning

An end-to-end Data Analytics and Machine Learning project that analyzes health survey records from **50,000 patients** across **42 features** to classify diabetes risk into **Low**, **Moderate**, or **High** categories. Includes an interactive **Streamlit** web application for real-time patient risk prediction.

🔗 **[Live Streamlit Web App]** *([Paste your Streamlit Cloud link here after deploying](https://diabetes-risk-prediction-by-rashid.streamlit.app/))*

## 📌 Project Overview
* **Objective:** Identify key clinical and lifestyle risk factors for diabetes and build an automated multi-class classification model.
* **Dataset:** `diabetes_risk_prediction_dataset.csv` (50,000 rows, 42 columns covering demographics, vitals, blood panels, and daily habits).

## ⚙️ Workflow & Methodology
1. **Data Cleaning & Preprocessing:** Imputed missing numerical values with the median and categorical values with the mode. Removed target-leaking columns (`Diabetes_Risk_Score`, `AI_Health_Recommendation`) and identifiers (`Patient_ID`).
2. **Exploratory Data Analysis (EDA):** Visualized class distributions, demographic breakdowns, outlier box plots, and feature correlations using `matplotlib` and `seaborn`.
3. **Feature Engineering & Balancing:** Encoded categorical variables using `LabelEncoder`, scaled numerical features with `StandardScaler`, and resolved severe class imbalance (73.19% High, 25.87% Moderate, 0.94% Low) using random oversampling (`resample`).
4. **Model Comparison & Hyperparameter Tuning:** Compared **Logistic Regression**, **Decision Tree**, and **Random Forest** using `GridSearchCV` (3-fold cross-validation).
5. **Model Evaluation:** Selected **Random Forest** as the final model, achieving **>98% accuracy** on the test set.

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/your-username/diabetes-risk-prediction.git](https://github.com/your-username/diabetes-risk-prediction.git)
   cd diabetes-risk-prediction
