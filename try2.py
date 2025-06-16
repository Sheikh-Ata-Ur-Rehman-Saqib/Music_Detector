# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:30:56 2024

@author: PMYLS
"""

import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Feature Extraction Function
def extract_features(file_path, use_mfcc=True, n_mfcc=13):
    """Extract MFCC or Spectrogram features from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file
        if use_mfcc:
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            return np.mean(features.T, axis=0)  # Mean across time axis
        else:
            spectrogram = np.abs(librosa.stft(y))  # Spectrogram
            return np.mean(spectrogram.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# 2. Manual k-NN Classifier
def knn_classifier(train_features, train_labels, test_sample, k=5):
    """Manual implementation of k-Nearest Neighbors."""
    distances = np.linalg.norm(train_features - test_sample, axis=1)
    neighbors = np.argsort(distances)[:k]  # Get k nearest neighbors
    labels, counts = np.unique(train_labels[neighbors], return_counts=True)
    return labels[np.argmax(counts)]  # Majority vote

# 3. Main Workflow
def main():
    # Paths and Labels
    data_dir = "audio_dataset/"  # Directory containing speech/music audio files
    categories = {'speech': 0, 'music': 1}
    
    # Prepare Data
    features, labels = [], []
    for category, label in categories.items():
        folder_path = os.path.join(data_dir, category)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            feature = extract_features(file_path, use_mfcc=True)  # Using MFCC
            if feature is not None:
                features.append(feature)
                labels.append(label)
    
    features = np.array(features)
    labels = np.array(labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Evaluate with manual k-NN
    predictions = []
    for test_sample in X_test:
        prediction = knn_classifier(X_train, y_train, test_sample, k=3)
        predictions.append(prediction)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
