import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from skimage.transform import resize
from pathlib import Path

def load_npy_data(directory):
    """Load .npy files and their filenames from a directory"""
    data, filenames = [], []
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            filepath = os.path.join(directory, filename)
            npy_array = np.load(filepath)
            data.append(npy_array)
            filenames.append(filename)
    return data, filenames

def preprocess_scan(scan, num_slices=3, target_size=(224, 224)):
    """Preprocess a single scan by selecting and resizing middle slices"""
    total_slices = scan.shape[0]
    mid_slice = total_slices // 2
    start_slice = max(mid_slice - num_slices // 2, 0)
    selected_slices = scan[start_slice : start_slice + num_slices]
    resized_slices = np.array([resize(slice, target_size) for slice in selected_slices])
    return resized_slices

def create_model():
    """Create a CNN model for binary classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(condition):
    """Train a model for a specific condition (ACL/Meniscus/Abnormal)"""
    print(f"\nTraining model for {condition}...")
    
    # Load and preprocess image data
    train_data, train_filenames = load_npy_data('./model/train/coronal')
    valid_data, valid_filenames = load_npy_data('./model/valid/coronal')
    
    # Load labels
    train_labels_df = pd.read_csv(f'./model/train-{condition.lower()}.csv', header=None)
    valid_labels_df = pd.read_csv(f'./model/valid-{condition.lower()}.csv', header=None)
    
    # Create label dictionaries
    train_labels_dict = dict(zip(train_labels_df[0].astype(str).str.zfill(4) + '.npy', train_labels_df[1]))
    valid_labels_dict = dict(zip(valid_labels_df[0].astype(str).str.zfill(4) + '.npy', valid_labels_df[1]))
    
    # Map labels to filenames
    y_train = np.array([train_labels_dict.get(fname, 0) for fname in train_filenames], dtype='int')
    y_valid = np.array([valid_labels_dict.get(fname, 0) for fname in valid_filenames], dtype='int')
    
    # Preprocess scans
    X_train = np.array([preprocess_scan(scan) for scan in train_data])
    X_valid = np.array([preprocess_scan(scan) for scan in valid_data])
    
    # Reshape data for CNN
    X_train_flat = X_train.reshape(-1, 224, 224, 1)
    X_valid_flat = X_valid.reshape(-1, 224, 224, 1)
    
    # Replicate labels for each slice
    y_train_expanded = np.repeat(y_train, 3)
    y_valid_expanded = np.repeat(y_valid, 3)
    
    print(f"Training data shape: {X_train_flat.shape}")
    print(f"Validation data shape: {X_valid_flat.shape}")
    
    # Create and train model
    model = create_model()
    
    history = model.fit(
        X_train_flat,
        y_train_expanded,
        epochs=20,
        batch_size=32,
        validation_data=(X_valid_flat, y_valid_expanded)
    )
    
    # Save model
    model_path = f'./model/{condition.lower()}_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate model
    val_loss, val_accuracy = model.evaluate(X_valid_flat, y_valid_expanded)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    return history

def main():
    """Train models for all conditions"""
    conditions = ['ACL', 'Meniscus', 'Abnormal']
    
    for condition in conditions:
        history = train_model(condition)
        print(f"\nFinished training {condition} model\n{'='*50}")

if __name__ == "__main__":
    main() 