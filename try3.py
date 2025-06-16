# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:34:08 2024

@author: PMYLS
"""

import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# 1. Feature Extraction Function
def extract_features(file_path, use_mfcc=True, n_mfcc=13, visualize=False):
    """Extract MFCC or Spectrogram features from an audio file and optionally visualize."""
    try:
        y, sr = librosa.load(file_path, sr=None)  # Load audio file
        
        if visualize:
            # Visualize waveform
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr)
            plt.title("Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.show()

        if use_mfcc:
            features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            if visualize:
                # Visualize MFCC
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(features, x_axis='time', sr=sr, cmap='viridis')
                plt.colorbar(label="MFCC Coefficients")
                plt.title("MFCC")
                plt.show()
            return np.mean(features.T, axis=0)  # Mean across time axis
        else:
            spectrogram = np.abs(librosa.stft(y))  # Spectrogram
            if visualize:
                # Visualize Spectrogram
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max),
                                         x_axis='time', y_axis='log', sr=sr, cmap='viridis')
                plt.colorbar(label="dB")
                plt.title("Spectrogram")
                plt.show()
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

# 3. Main Workflow with Visualization
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
            feature = extract_features(file_path, use_mfcc=True, visualize=True)  # Using MFCC with visualization
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

    # Visualize Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Speech', 'Music'])
    disp.plot(cmap='viridis')
    plt.title("Confusion Matrix")
    plt.show()

    # Plot feature distribution (2D scatter of first two features for visualization)
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(['Speech', 'Music']):
        plt.scatter(features[labels == i, 0], features[labels == i, 1], label=label, alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Feature Distribution (First Two Features)")
    plt.show()

if __name__ == "__main__":
    main()
