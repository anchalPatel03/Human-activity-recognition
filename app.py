import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Streamlit page configuration
st.set_page_config(page_title="HAR for Elderly", layout="centered")

# Centered logo above the title
st.image("logo.png", width=700)  # Set the width to your desired size (e.g., 700 pixels)

# Title centered below the logo
st.markdown(
    """
    <h1 style='padding-top: 10px; font-size: 34px; text-align: center;'>
    Human Activity Recognition for Elderly Monitoring
    </h1>
    """,
    unsafe_allow_html=True
)

# Instructions Section
st.markdown("""
### How to Use:
1. **Upload Sensor Data CSV**: Click on the "Upload" button and choose your sensor data CSV file. The file should contain sensor readings such as `back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, and `thigh_z`.
2. **Prediction**: After uploading, the app will automatically predict the activities of elderly individuals based on the sensor data.
3. **Data Visualization**: View visualizations like activity distribution, confusion matrix, and more to understand the performance of the model.

**Important Notes:**
- Make sure your CSV file includes all the necessary columns.
- The app will output predictions and visualizations once the file is uploaded and processed.
""", unsafe_allow_html=True)

# Load model
try:
    model = load_model('best_model (2).h5')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Load label encoder
try:
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
except FileNotFoundError:
    st.error("Label encoder file not found!")
    st.stop()
except Exception as e:
    st.error(f"Error loading label encoder: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📁 Upload Sensor CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Data Preview:", df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    required_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"CSV is missing required columns: {', '.join(missing_columns)}")
        st.stop()

    # Preprocess
    X = df[required_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sequence_length = 128
    n_samples = X_scaled.shape[0] // sequence_length

    if n_samples == 0:
        st.error("⚠️ Not enough data to create sequences. Please upload more data.")
        st.stop()

    X_seq = X_scaled[:n_samples * sequence_length].reshape(n_samples, sequence_length, 6)

    with st.spinner('🔍 Making predictions...'):
        try:
            y_pred = model.predict(X_seq)
            pred_classes = np.argmax(y_pred, axis=1)
            activity_names = label_encoder.inverse_transform(pred_classes)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

    # Show predictions
    st.success("✅ Prediction Completed!")
    st.write("🕵️‍♀️ Predicted Activities:")
    st.write(activity_names)

    # Data Visualization: Activity count
    st.subheader("Activity Count Distribution")
    activity_counts = pd.Series(activity_names).value_counts()
    st.bar_chart(activity_counts)

    # Visualizing Sensor Data Distributions
    st.subheader("Sensor Data Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].hist(df['back_x'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Back X Distribution")

    axes[0, 1].hist(df['back_y'], bins=30, color='salmon', edgecolor='black')
    axes[0, 1].set_title("Back Y Distribution")

    axes[0, 2].hist(df['back_z'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 2].set_title("Back Z Distribution")

    axes[1, 0].hist(df['thigh_x'], bins=30, color='yellow', edgecolor='black')
    axes[1, 0].set_title("Thigh X Distribution")

    axes[1, 1].hist(df['thigh_y'], bins=30, color='orange', edgecolor='black')
    axes[1, 1].set_title("Thigh Y Distribution")

    axes[1, 2].hist(df['thigh_z'], bins=30, color='purple', edgecolor='black')
    axes[1, 2].set_title("Thigh Z Distribution")

    for ax in axes.flatten():
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    st.pyplot(fig)

    # Data Visualization: Confusion Matrix (if actual labels present)
    if 'activity_true' in df.columns:
        st.write("📌 Confusion Matrix:")
        try:
            cm = confusion_matrix(df['activity_true'][:len(pred_classes)], pred_classes)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_encoder.classes_,
                        yticklabels=label_encoder.classes_)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Unable to display confusion matrix: {e}")

    # Download predictions
    output_df = pd.DataFrame({'Predicted Activity': activity_names})
    st.download_button(
        label="⬇️ Download Predictions",
        data=output_df.to_csv(index=False).encode('utf-8'),
        file_name="predicted_activities.csv",
        mime="text/csv"
    )
