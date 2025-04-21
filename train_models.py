import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from skimage.transform import resize
from pathlib import Path
import json

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

def limit_dataset_size(data, filenames, labels, max_samples=10):
    """Limit the dataset to a small number of samples while maintaining class balance"""
    # If max_samples is -1, return the full dataset without limiting
    if max_samples == -1:
        return data, filenames, labels
        
    if len(data) <= max_samples:
        return data, filenames, labels
    
    # Convert to numpy arrays for easier manipulation
    data = np.array(data, dtype=object)  # Use dtype=object to handle varying shapes
    filenames = np.array(filenames)
    labels = np.array(labels)
    
    # Get indices for positive and negative samples
    pos_indices = np.where(labels == 1)[0]
    neg_indices = np.where(labels == 0)[0]
    
    # Calculate how many samples to take from each class
    pos_samples = min(len(pos_indices), max_samples // 2)
    neg_samples = max_samples - pos_samples
    
    # Randomly select samples from each class
    np.random.seed(42)  # For reproducibility
    selected_pos = np.random.choice(pos_indices, pos_samples, replace=False)
    selected_neg = np.random.choice(neg_indices, neg_samples, replace=False)
    
    # Combine selected indices
    selected_indices = np.concatenate([selected_pos, selected_neg])
    
    # Convert back to list for data to maintain original format
    return data[selected_indices].tolist(), filenames[selected_indices], labels[selected_indices]

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, and recall manually"""
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Calculate true positives, false positives, true negatives, false negatives
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return accuracy, precision, recall

def save_metrics(condition, metrics):
    """Save metrics to a JSON file"""
    metrics_path = f'./model/{condition.lower()}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

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
    
    # Limit dataset size
    # Use -1 to use the full dataset without limiting
    train_data, train_filenames, y_train = limit_dataset_size(train_data, train_filenames, y_train, max_samples=10)
    valid_data, valid_filenames, y_valid = limit_dataset_size(valid_data, valid_filenames, y_valid, max_samples=5)
    
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
    
    # Add early stopping to prevent overfitting on small dataset
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_flat,
        y_train_expanded,
        epochs=20,
        batch_size=8,  # Reduced batch size for small dataset
        validation_data=(X_valid_flat, y_valid_expanded),
        callbacks=[early_stopping]
    )
    
    # Calculate predictions
    y_pred = model.predict(X_valid_flat)
    
    # Calculate metrics
    accuracy, precision, recall = calculate_metrics(y_valid_expanded, y_pred)
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall)
    }
    save_metrics(condition, metrics)
    
    # Print metrics
    print(f"\nPerformance Metrics for {condition}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Save model
    model_path = f'./model/{condition.lower()}_model.h5'
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    return history, metrics

def main():
    """Train models for all conditions"""
    conditions = ['ACL', 'Meniscus', 'Abnormal']
    
    for condition in conditions:
        history, metrics = train_model(condition)
        print(f"\nFinished training {condition} model\n{'='*50}")

if __name__ == "__main__":
    main() 