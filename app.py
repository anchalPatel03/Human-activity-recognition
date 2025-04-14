import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import shap

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

# Title and description
st.title("Human Activity Recognition for Elderly Monitoring")
st.write("This app predicts human activities for elderly monitoring based on sensor data collected from wearable devices.")

# File uploader to upload the CSV file
uploaded_file = st.file_uploader("Upload Sensor CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:", df.head())
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # Check for missing columns
    required_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        st.stop()

    # Display raw data visualization
    st.subheader("Sensor Data Visualization")
    st.line_chart(df[['back_x', 'back_y', 'back_z']].head(100))  # Visualizing the first 100 rows of sensor data

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
    st.subheader("Predicted Activities")
    st.write(activity_names)

    # Show activity distribution
    activity_counts = pd.Series(activity_names).value_counts()
    st.subheader("Activity Distribution")
    st.bar_chart(activity_counts)

    # Confusion Matrix (if ground truth is available)
    if 'activity_true' in df.columns:
        cm = confusion_matrix(df['activity_true'], pred_classes)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        # Model Evaluation Metrics (Precision, Recall, F1 Score)
        st.subheader("Model Evaluation Metrics")
        st.write("Accuracy:", accuracy_score(df['activity_true'], pred_classes))
        st.write("Classification Report:")
        st.text(classification_report(df['activity_true'], pred_classes))

    # SHAP or LIME for Model Explainability
    if st.checkbox('Show SHAP Values (Explaining Predictions)'):
        st.write("Explaining the predictions with SHAP...")
        explainer = shap.KernelExplainer(model.predict, X_seq[:100])  # Sample from the data
        shap_values = explainer.shap_values(X_seq[:10])  # Explain first 10 predictions
        shap.initjs()
        st.shap_plot(shap_values)  # Display SHAP plots

    # Provide download option for predictions
    output_df = pd.DataFrame({'Predicted Activity': activity_names})
    st.download_button(
        label="Download Predicted Activities",
        data=output_df.to_csv(index=False).encode('utf-8'),
        file_name="predicted_activities.csv",
        mime="text/csv"
    )

    # Allow users to download the processed data along with predictions
    processed_data = df.copy()
    processed_data['Predicted Activity'] = activity_names
    st.download_button(
        label="Download Processed Data with Predictions",
        data=processed_data.to_csv(index=False).encode('utf-8'),
        file_name="processed_data_with_predictions.csv",
        mime="text/csv"
    )

    # Show real-time prediction visualization (optional)
    if st.checkbox("Show Real-Time Activity Prediction Over Time"):
        predictions_df = pd.DataFrame({'Time': df.index, 'Predicted Activity': activity_names})
        st.line_chart(predictions_df.set_index('Time'))

    # Option to change sequence length
    sequence_length = st.slider("Select Sequence Length", min_value=50, max_value=500, value=128)
    st.write(f"Sequence length set to {sequence_length}")
