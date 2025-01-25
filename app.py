import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('notebook/model.h5')

# Load the encoders and scaler
with open('notebook/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('notebook/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('notebook/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.set_page_config(page_title="Customer Churn Prediction", layout="centered", initial_sidebar_state="expanded")

st.title("📊 Customer Churn Prediction")
st.write(
    """
    Welcome to the **Customer Churn Prediction App**!  
    This tool helps predict whether a customer is likely to churn based on various factors.
    Please provide the necessary details below, and click on **Predict** to get insights.
    """
)

# Sidebar for instructions
with st.sidebar:
    st.header("🔎 Instructions")
    st.write(
        """
        - Fill in all the fields in the main panel.
        - Click **Predict** to view the churn probability.
        - Use the slider for age, tenure, and number of products.
        - Adjust numerical fields such as balance and salary as needed.
        """
    )
    st.markdown("---")
    st.header("📄 About the Model")
    st.write("The model is built using TensorFlow and trained on customer data. It uses one-hot encoding and scaling to preprocess inputs.")

# Input Section
st.header("📝 Input Customer Details")
st.subheader("Demographics")
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
with col2:
    age = st.slider('🎂 Age', 18, 92, value=30)
    balance = st.number_input('💰 Balance', value=0.0)

st.subheader("Financial Details")
col3, col4 = st.columns(2)
with col3:
    credit_score = st.number_input('📈 Credit Score', value=650)
    estimated_salary = st.number_input('💵 Estimated Salary', value=50000.0)
with col4:
    tenure = st.slider('📅 Tenure (Years)', 0, 10, value=5)
    num_of_products = st.slider('🛍️ Number of Products', 1, 4, value=1)

st.subheader("Activity and Card Details")
col5, col6 = st.columns(2)
with col5:
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col6:
    is_active_member = st.selectbox('🟢 Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict button
if st.button("🚀 Predict"):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    numerical_columns = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])

    # Predict churn
    prediction = model.predict(input_data)
    prediction_proba = prediction[0][0]

    # Display the results
    st.header("🔍 Prediction Results")
    st.write(f"**Churn Probability:** {prediction_proba:.2f}")
    if prediction_proba > 0.5:
        st.error("⚠️ The customer is likely to churn.")
    else:
        st.success("✅ The customer is not likely to churn.")

    # Additional context
    st.markdown(
        """
        ---
        **Note:** This prediction is based on statistical modeling and should not be the sole determinant for business decisions.
        """
    )
