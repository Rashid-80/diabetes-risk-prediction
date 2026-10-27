import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

# App Title
st.title("🩺 Simple Diabetes Risk Predictor")
st.write("Adjust the sliders below to predict diabetes risk based on patient health metrics.")

# 1. Load Data & Train Model
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv(r"C:\Users\korad\Downloads\diabetes_risk_prediction_dataset.csv")
    
    # We will use just 4 key features to keep the app simple
    features = ['Age', 'BMI', 'Blood_Glucose', 'HbA1c']
    X = df[features].fillna(df[features].median())
    y = df['Diabetes_Risk']
    
    # Balance the data so the model doesn't just guess "High" every time
    train_data = pd.concat([X, y], axis=1)
    high = train_data[train_data['Diabetes_Risk'] == 'High']
    mod = train_data[train_data['Diabetes_Risk'] == 'Moderate']
    low = train_data[train_data['Diabetes_Risk'] == 'Low']
    
    low_up = resample(low, replace=True, n_samples=len(high), random_state=42)
    mod_up = resample(mod, replace=True, n_samples=len(high), random_state=42)
    balanced_data = pd.concat([high, mod_up, low_up])
    
    # Train a quick Random Forest model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(balanced_data[features], balanced_data['Diabetes_Risk'])
    
    return model

model = train_model()

# 2. User Inputs (Sliders)
st.subheader("Enter Patient Details")
age = st.slider("Age", 10, 100, 35)
bmi = st.slider("BMI", 10.0, 60.0, 25.0)
glucose = st.slider("Blood Glucose Level", 50.0, 300.0, 100.0)
hba1c = st.slider("HbA1c Level", 4.0, 15.0, 5.5)

# 3. Predict Button & Result
if st.button("Predict Risk"):
    # Format the user's input for the model
    user_data = pd.DataFrame({
        'Age': [age],
        'BMI': [bmi],
        'Blood_Glucose': [glucose],
        'HbA1c': [hba1c]
    })
    
    # Get prediction
    prediction = model.predict(user_data)[0]
    
    # Show the result in a big highlight box
    if prediction == "High":
        st.error(f"🚨 Predicted Risk: **{prediction}**")
    elif prediction == "Moderate":
        st.warning(f"⚠️ Predicted Risk: **{prediction}**")
    else:
        st.success(f"✅ Predicted Risk: **{prediction}**")
