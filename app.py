import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model and label encoder
model = load_model('best_model (2).h5')

# Ensure correct path for the label encoder
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Label encoder file not found!")
    st.stop()

st.title("Human Activity Recognition for Elderly Monitoring")

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload Sensor CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display the uploaded data preview
    st.write("Data Preview:", df.head())

    # Check if the necessary columns are present
    required_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    if not all(col in df.columns for col in required_columns):
        st.error("CSV file is missing required columns!")
        st.stop()

    # Preprocess the data (standard scaling)
    X = df[required_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for prediction (ensure there are enough rows for sequences)
    sequence_length = 128
    n_samples = X_scaled.shape[0] // sequence_length
    if n_samples == 0:
        st.error("Not enough data to create sequences. Please upload more data.")
        st.stop()

    X_seq = X_scaled[:n_samples * sequence_length].reshape(n_samples, sequence_length, 6)

    # Predict activities
    y_pred = model.predict(X_seq)
    pred_classes = np.argmax(y_pred, axis=1)
    activity_names = label_encoder.inverse_transform(pred_classes)

    # Display predicted activities
    st.write("Predicted Activities:")
    st.write(activity_names)
