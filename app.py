import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="⚠️",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(filename, use_joblib=True):
    try:
        if use_joblib:
            return joblib.load(f'{filename}.joblib')
        else:
            with open(f'{filename}.pkl', 'rb') as f:
                return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {filename}.joblib/.pkl")
        st.stop()

try:
    best_model = load_model('best_model', use_joblib=True)
    scaler = load_model('scaler', use_joblib=True)
    le_gender = load_model('gender_encoder', use_joblib=True)
    feature_names = load_model('feature_names', use_joblib=True)
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.info("📌 Please run the notebook's model-saving cell first to create the model files.")
    st.stop()

# Threshold optimized for F1 score
THRESHOLD = 0.45

# ── Feature engineering (must match training notebook) ────────────────────────
def engineer_features(df):
    """Apply same feature engineering as training notebook"""
    df = df.copy()
    
    # Engineered features
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
    
    # Ensure all feature names exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚠️ Churn Prediction Model")
st.markdown("Predict customer churn risk based on subscription and service metrics.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Customer Profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    country = st.selectbox("Country", ["Spain", "France", "Germany"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)

with col2:
    st.subheader("📋 Subscription Details")
    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Digital Wallet"])
    support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=2)

with col3:
    st.subheader("💰 Charges & Usage")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=65.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=50.0)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Churn", use_container_width=True, type="primary"):
    
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'Country': country,
        'SubscriptionType': subscription_type,
        'ContractType': contract_type,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'TenureMonths': tenure,
        'SupportTickets': support_tickets,
        'PaymentMethod': payment_method,
    }])
    
    # Feature engineering
    input_fe = engineer_features(input_data)
    
    # Select features and scale
    X = input_fe[feature_names]
    X_scaled = scaler.transform(X)
    
    # Make prediction
    churn_proba = best_model.predict_proba(X_scaled)[0, 1]
    will_churn = churn_proba >= THRESHOLD
    
    st.divider()
    st.subheader("📊 Prediction Result")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        if will_churn:
            st.error("⚠️ **AT RISK**")
        else:
            st.success("✅ **RETAINED**")
    
    with col_res2:
        st.metric("Churn Probability", f"{churn_proba*100:.1f}%")
    
    with col_res3:
        st.metric("Decision Threshold", f"{THRESHOLD:.3f}")
    
    # ── Risk gauge ────────────────────────────────────────────────────────────
    st.markdown("#### Risk Level")
    bar_color = "🔴" if churn_proba >= 0.7 else ("🟡" if churn_proba >= 0.4 else "🟢")
    risk_label = "High Risk" if churn_proba >= 0.7 else ("Medium Risk" if churn_proba >= 0.4 else "Low Risk")
    st.progress(float(churn_proba), text=f"{bar_color} {risk_label} — {churn_proba*100:.1f}%")
    
    # ── Risk factors ──────────────────────────────────────────────────────────
    st.markdown("#### ⚡ Key Risk Factors Detected")
    flags = []
    if support_tickets >= 7:
        flags.append("🔴 High support calls (≥7) — strong churn signal")
    if tenure <= 6:
        flags.append("🟡 Low tenure (≤6 months) — new customer, higher churn risk")
    if contract_type == "Month-to-month":
        flags.append("🟡 Month-to-month contract — less committed customer")
    if monthly_charges > 100:
        flags.append("🟡 High monthly charges (>$100) — price sensitivity risk")
    if support_tickets >= 5 and tenure <= 12:
        flags.append("🔴 CRITICAL: High support calls + low tenure combined")
    
    if flags:
        for f in flags:
            st.markdown(f"- {f}")
    else:
        st.markdown("- 🟢 No major risk factors detected")
    
    # ── Recommendation ────────────────────────────────────────────────────────
    st.markdown("#### 💡 Recommended Action")
    if churn_proba >= 0.7:
        st.warning("**Immediate action needed.** Assign a retention agent, offer loyalty discount or service upgrade.")
    elif churn_proba >= 0.4:
        st.info("**Monitor closely.** Send re-engagement email or check-in call.")
    else:
        st.success("**No action needed.** Customer appears stable and engaged.")

# ── Batch prediction ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📁 Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV with columns: `Age, Gender, Country, SubscriptionType, ContractType, MonthlyCharges, TotalCharges, TenureMonths, SupportTickets, PaymentMethod`")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    batch_fe = engineer_features(batch_df.copy())
    
    # Ensure feature order
    X_batch = batch_fe[feature_names]
    X_batch_scaled = scaler.transform(X_batch)
    
    probas = best_model.predict_proba(X_batch_scaled)[:, 1]
    preds = (probas >= THRESHOLD).astype(int)
    
    batch_df["Churn_Probability"] = (probas * 100).round(1)
    batch_df["Predicted_Churn"] = preds
    batch_df["Risk"] = pd.cut(
        probas,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )
    
    st.success(f"✅ Processed {len(batch_df)} customers — "
               f"{preds.sum()} predicted to churn ({preds.mean()*100:.1f}%)")
    
    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Total Customers", len(batch_df))
    col_b2.metric("At Risk (Churn)", preds.sum())
    col_b3.metric("Churn Rate", f"{preds.mean()*100:.1f}%")
    
    st.dataframe(batch_df.sort_values("Churn_Probability", ascending=False),
                 use_container_width=True)
    
    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv,
                       file_name="churn_predictions.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: RandomForest | Threshold: Optimised for F1 Score | Data: Loan Approval Dataset")
