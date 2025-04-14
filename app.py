import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

# Title for the Streamlit app
st.title("Human Activity Recognition for Elderly Monitoring")

# Instructions for uploading CSV
st.markdown("""
**Instructions for Uploading CSV File:**
1. The uploaded CSV must contain the following columns:
    - `back_x`: X-axis data from the back sensor
    - `back_y`: Y-axis data from the back sensor
    - `back_z`: Z-axis data from the back sensor
    - `thigh_x`: X-axis data from the thigh sensor
    - `thigh_y`: Y-axis data from the thigh sensor
    - `thigh_z`: Z-axis data from the thigh sensor

2. Ensure the data is in a numeric format, with one row for each time step.
3. If your data contains any missing values, handle them before uploading.
""")

# File uploader for CSV input
uploaded_file = st.file_uploader("Upload Sensor CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Check if necessary columns are present
    required_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"CSV file is missing required columns: {', '.join(missing_columns)}")
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

    # Show a loading spinner while making predictions
    with st.spinner('Making predictions...'):
        try:
            # Predict activities
            y_pred = model.predict(X_seq)
            pred_classes = np.argmax(y_pred, axis=1)
            activity_names = label_encoder.inverse_transform(pred_classes)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            st.stop()

    # Display predicted activities
    st.write("Predicted Activities:")
    st.write(activity_names)

    # Show activity distribution (Bar Chart for Activity Count)
    activity_counts = pd.Series(activity_names).value_counts()
    st.write("Activity Distribution:")
    st.bar_chart(activity_counts)

    # Plot confusion matrix (if ground truth is available)
    if 'activity_true' in df.columns:
        cm = confusion_matrix(df['activity_true'], pred_classes)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

    # Allow users to download predicted activities as a CSV file
    output_df = pd.DataFrame({'Predicted Activity': activity_names})
    st.download_button(
        label="Download Predicted Activities",
        data=output_df.to_csv(index=False).encode('utf-8'),
        file_name="predicted_activities.csv",
        mime="text/csv"
    )
