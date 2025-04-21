import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import os
from pathlib import Path
import json

# Set page config
st.set_page_config(
    page_title="MRI Ligament Analysis",
    page_icon="🏥",
    layout="wide"
)

# Title and description
st.title("MRI Ligament Analysis")
st.markdown("""
This application analyzes knee MRI images to detect:
- ACL tears
- Meniscus tears
- General abnormalities

Upload an MRI scan to get instant analysis results.
""")

def load_model(model_path):
    """Load the trained model"""
    model = tf.keras.models.load_model(model_path)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model input"""
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize and normalize
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimensions
    return image

def get_prediction(model, image):
    """Get model prediction"""
    prediction = model.predict(image)
    return float(prediction[0][0])  # Convert to Python float

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    
    with col2:
        try:
            # Load and preprocess the image
            image = Image.open(uploaded_file)
            processed_image = preprocess_image(image)
            
            # Load models
            model_paths = {
                'ACL': './model/acl_model.h5',
                'Meniscus': './model/meniscus_model.h5',
                'Abnormal': './model/abnormal_model.h5'
            }

            st.subheader("Analysis Results")

            for condition, model_path in model_paths.items():
                if Path(model_path).exists():
                    model = load_model(model_path)
                    prediction = get_prediction(model, processed_image)
                    
                    # Display prediction with progress bar
                    st.write(f"{condition}:")
                    st.progress(prediction)
                    st.write(f"Probability: {prediction * 100:.1f}%")
                else:
                    st.error(f"Model file for {condition} not found. Please ensure models are properly saved in the model directory.")

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

# Add model statistics
st.sidebar.header("Model Statistics")
try:
    # Load training statistics
    train_stats = pd.read_csv("./model/train-abnormal.csv")
    valid_stats = pd.read_csv("./model/valid-abnormal.csv")
    
    st.sidebar.write(f"Training samples: {len(train_stats)}")
    st.sidebar.write(f"Validation samples: {len(valid_stats)}")
    
    # Add model performance metrics
    st.sidebar.header("Model Performance")
    
    # Load and display metrics for each condition
    conditions = ['ACL', 'Meniscus', 'Abnormal']
    for condition in conditions:
        metrics_path = f'./model/{condition.lower()}_metrics.json'
        if Path(metrics_path).exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                st.sidebar.subheader(f"{condition} Model")
                st.sidebar.write(f"Accuracy: {metrics['accuracy']*100:.1f}%")
                st.sidebar.write(f"Precision: {metrics['precision']*100:.1f}%")
                st.sidebar.write(f"Recall: {metrics['recall']*100:.1f}%")
                st.sidebar.write("---")
    
except Exception as e:
    st.sidebar.warning("Could not load model statistics")

# Add information about the model
st.sidebar.header("About")
st.sidebar.info("""
This application uses deep learning models trained on knee MRI images to detect:
- ACL tears
- Meniscus tears
- General abnormalities

The models were trained on a dataset of over 1000 MRI scans with expert annotations.
""")

# Add usage instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Upload an MRI scan using the file uploader
2. The system will automatically analyze the image
3. Results show probability scores for each condition
4. Higher percentages indicate higher likelihood of the condition
""")
