import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load model and label encoder
model = load_model('best_model (2).h5')
with open('path/to/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

st.title("Human Activity Recognition for Elderly Monitoring")

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload Sensor CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Preprocess the data (adjust according to your features)
    X = df[['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for prediction
    sequence_length = 128
    n_samples = X_scaled.shape[0] // sequence_length
    X_seq = X_scaled[:n_samples * sequence_length].reshape(n_samples, sequence_length, 6)

    # Predict activities
    y_pred = model.predict(X_seq)
    pred_classes = np.argmax(y_pred, axis=1)
    activity_names = label_encoder.inverse_transform(pred_classes)

    # Display predicted activities
    st.write("Predicted Activities:")
    st.write(activity_names)
