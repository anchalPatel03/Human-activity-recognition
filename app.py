import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the model and label encoder
try:
    model = load_model('best_model (2).h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Label encoder file not found!")
    st.stop()
except Exception as e:
    st.error(f"Error loading label encoder: {e}")
    st.stop()

st.title("Human Activity Recognition for Elderly Monitoring")

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload Sensor CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Check for missing columns
    required_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        st.stop()

    # Preprocess data
    X = df[required_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences for prediction
    sequence_length = 128
    n_samples = X_scaled.shape[0] // sequence_length
    if n_samples == 0:
        st.error("Not enough data for sequences!")
        st.stop()

    X_seq = X_scaled[:n_samples * sequence_length].reshape(n_samples, sequence_length, 6)

    with st.spinner("Making predictions..."):
        y_pred = model.predict(X_seq)
        pred_classes = np.argmax(y_pred, axis=1)
        activity_names = label_encoder.inverse_transform(pred_classes)

    # Display predicted activities
    st.write("Predicted Activities:")
    st.write(activity_names)

    # Plot activity distribution
    activity_counts = pd.Series(activity_names).value_counts()
    st.bar_chart(activity_counts)

    # Display confusion matrix if ground truth is available
    if 'activity_true' in df.columns:
        cm = confusion_matrix(df['activity_true'], pred_classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

    # Provide download option
    output_df = pd.DataFrame({'Predicted Activity': activity_names})
    st.download_button(
        label="Download Predictions",
        data=output_df.to_csv(index=False).encode('utf-8'),
        file_name="predicted_activities.csv",
        mime="text/csv"
    )
