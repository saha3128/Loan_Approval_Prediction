import pickle
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

# Load models and encoders from joblib/pickle files
def load_model(filename, use_joblib=True):
    try:
        if use_joblib:
            return joblib.load(f'{filename}.joblib')
        else:
            with open(f'{filename}.pkl', 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        raise Exception(f"Model file not found: {filename}. Please run the notebook's model-saving cell first.")

try:
    best_model = load_model('best_model', use_joblib=True)
    scaler = load_model('scaler', use_joblib=True)
    le_gender = load_model('gender_encoder', use_joblib=True)
    feature_names = load_model('feature_names', use_joblib=True)
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")
    raise

# Threshold optimized for F1 score (from notebook analysis)
THRESHOLD = 0.45

class ChurnPredictionInput(BaseModel):
    age: int
    gender: str  # "Male" or "Female"
    country: str
    subscription_type: str
    contract_type: str
    monthly_charges: float
    total_charges: float
    tenure_months: int
    support_tickets: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Use POST /predict to make predictions."}

@app.post("/predict")
def predict(customer: ChurnPredictionInput):
    """Predict churn probability for a given customer"""
    
    try:
        # Create base DataFrame
        df = pd.DataFrame([{
            'Age': customer.age,
            'Gender': customer.gender,
            'Country': customer.country,
            'SubscriptionType': customer.subscription_type,
            'ContractType': customer.contract_type,
            'MonthlyCharges': customer.monthly_charges,
            'TotalCharges': customer.total_charges,
            'TenureMonths': customer.tenure_months,
            'SupportTickets': customer.support_tickets,
        }])
        
        # Feature engineering (must match notebook)
        df['ChargesPerMonth'] = df['TotalCharges'] / (df['TenureMonths'] + 1)
        df['TicketsPerMonth'] = df['SupportTickets'] / (df['TenureMonths'] + 1)
        df['HighSupport'] = (df['SupportTickets'] >= 3).astype(int)
        df['ShortTenure'] = (df['TenureMonths'] <= 6).astype(int)
        df['IsMonthly'] = (df['ContractType'] == 'Monthly').astype(int)
        df['ChargeToAge'] = df['MonthlyCharges'] / df['Age']
        df['TenureBucket'] = pd.cut(df['TenureMonths'], bins=[-1,6,12,24,1000],
                                    labels=['0-6','7-12','13-24','25+']).astype(str)
        
        # Encode gender
        df['Gender'] = le_gender.transform(df['Gender'])
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df,
            columns=['Country','SubscriptionType','ContractType','PaymentMethod','TenureBucket'],
            drop_first=False)
        
        # Ensure all features exist (add missing ones with 0)
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Select features in correct order and scale
        X = df[feature_names]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        churn_proba = best_model.predict_proba(X_scaled)[0, 1]
        will_churn = churn_proba >= THRESHOLD
        
        return {
            "customer": customer.dict(),
            "churn_probability": round(float(churn_proba), 4),
            "will_churn": bool(will_churn),
            "threshold": THRESHOLD,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }