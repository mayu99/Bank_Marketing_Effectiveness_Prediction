import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("🌲 Bank Term Deposit Predictor")

# Load the model with error handling
try:
    model = joblib.load("random_forest_pipeline.pkl")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")
    st.stop()

def preprocess_input(age, job, marital, education, default, balance, housing, loan, 
                    contact, day, month, duration, campaign, pdays, previous, poutcome):
    # Create base dataframe
    df = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }])
    
    # Convert binary categorical variables to numeric
    binary_cols = ['default', 'housing', 'loan']
    for col in binary_cols:
        df[col] = (df[col] == 'yes').astype(int)
    
    # Age categories
    df['age_categories'] = pd.cut(df['age'], 
                                bins=[0, 30, 45, 60, 100],
                                labels=['struggling', 'stable', 'about to retire', 'old age'])
    
    # Job categories (simplified grouping)
    job_categories = {
        'admin.': 'cat1',
        'technician': 'cat1',
        'services': 'cat2',
        'management': 'cat2',
        'retired': 'cat3',
        'blue-collar': 'cat2',
        'entrepreneur': 'cat4',
        'housemaid': 'cat2',
        'student': 'cat3',
        'unemployed': 'cat3',
        'self-employed': 'cat4',
        'unknown': 'cat3'
    }
    df['job_categories'] = df['job'].map(job_categories)
    
    # One-hot encode categorical variables
    categorical_cols = ['education', 'marital', 'contact', 'month', 'poutcome', 
                       'age_categories', 'job_categories']
    
    # Create dummy variables
    dummies = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
    
    # Drop original categorical columns
    df = df.drop(categorical_cols, axis=1)
    
    # Combine with dummy variables
    df = pd.concat([df, dummies], axis=1)
    
    # Ensure all expected columns are present
    expected_columns = {
        'education_secondary', 'age_categories_old age', 'education_tertiary',
        'month_sep', 'job_categories_cat1', 'poutcome_failure', 'marital_single',
        'month_jan', 'age_categories_struggling', 'month_nov', 'month_feb',
        'marital_married', 'month_jul', 'education_primary', 'contact_telephone',
        'month_apr', 'month_may', 'month_oct', 'job_categories_cat2',
        'job_categories_cat4', 'contact_cellular', 'poutcome_success',
        'poutcome_unknown', 'month_jun', 'month_aug', 'age_categories_stable',
        'age_categories_about to retire', 'month_mar'
    }
    
    # Add missing columns with zeros
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Ensure columns are in the same order as expected by the model
    df = df.reindex(columns=sorted(df.columns))
    
    return df

# Add some styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stForm {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

with st.form("form"):
    st.subheader("Customer Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 90, 35)
        job = st.selectbox("Job", ['admin.', 'technician', 'services', 'retired', 'management', 'blue-collar', 'entrepreneur', 'housemaid', 'student', 'unemployed', 'self-employed', 'unknown'])
        marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
        education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
        default = st.radio("Credit in Default?", ['yes', 'no'])
        balance = st.number_input("Balance", -5000, 100000, 1000)
    
    with col2:
        housing = st.radio("Housing Loan?", ['yes', 'no'])
        loan = st.radio("Personal Loan?", ['yes', 'no'])
        contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
        day = st.slider("Day", 1, 31, 15)
        month = st.selectbox("Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
        duration = st.slider("Duration (seconds)", 0, 1200, 300)
    
    st.subheader("Campaign Information")
    col3, col4 = st.columns(2)
    
    with col3:
        campaign = st.slider("Number of Contacts", 1, 5, 1)
        pdays = st.selectbox("Days Since Last Contact", [999, 5, 10, 20])
    
    with col4:
        previous = st.slider("Previous Contacts", 0, 5, 0)
        poutcome = st.selectbox("Previous Outcome", ['success', 'failure', 'unknown'])
    
    submit = st.form_submit_button("Predict Subscription")

if submit:
    try:
        # Preprocess the input data
        df = preprocess_input(age, job, marital, education, default, balance, housing, loan,
                            contact, day, month, duration, campaign, pdays, previous, poutcome)
        
        # Display the preprocessed data
        st.write("Preprocessed Data:")
        st.write(df)
        
        # Make prediction
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        
        # Display results with more detail
        st.write("Prediction Probabilities:")
        st.write(f"Not Subscribe: {prob[0]:.1%}")
        st.write(f"Subscribe: {prob[1]:.1%}")
        
        if pred == 1:
            st.success(f"✅ Customer is likely to subscribe! (Confidence: {prob[1]:.1%})")
        else:
            st.error(f"❌ Customer is unlikely to subscribe (Confidence: {prob[0]:.1%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check the console for more details.")
