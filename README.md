# Music_Detector
Music vs Speech Detector using KNN and Logistic Regression

# How to use

1. install any python ide like Spyder, Anaconda, or even Google-colab.
2. open the file named "Realtime_Testing.py" on it.
3. make sure that the files "extracted_features.pkl", "extracted_labels.pkl", "knn_model.pkl" and "log_reg_model.pkl" are in the same directory as that of python project file.
4. Run the script on the IDE, and upload any test file using the browse option.
5. sometimes it still makes mistakes, especially on those files containing lots of noise, but it is staill beneficial project for starters.

# Background:
Audio classification plays a crucial role in applications like voice assistants, music recommendation systems, and content filtering.
Differentiating between speech and music is a foundational step for advanced audio processing tasks.

# Motivation:
Building a classifier for detecting music within web browsers so that Sin of listening to music can be avoided.

# Objective:
To build an efficient classifier for distinguishing speech from music using machine learning.
To compare the performance of KNN and Logistic Regression models.

# System Architecture details

# Workflow:
Data Preprocessing: Audio loading, resampling, and feature extraction.
**Model Training:** Using MFCC feature vectors.
**Evaluation:** Accuracy, precision, recall, and F1-score.
**Interactive Testing:** Real-time or uploaded audio classification.

# Tools and Technologies:
**Libraries:** Librosa, Tkinter, Scikit-learn.
**Platforms:** Google Colab, Spyder IDE.

# Components:
**Component list:** Audio files, Python libraries, and GUI environment.
Diagram illustrating the connection between preprocessing, classification, and user interface.

# Results

# Data Used and Experimental Setup:
**KNN:**
500 speech and 500 music samples (split equally between training and testing sets).
**Logistic Regression:**
400 total samples, balanced across speech and music.

# Hardware/Software Simulation Results:
# KNN:
**Accuracy:** 95%
**Precision, Recall, F1-Score (for both classes):** 0.95
# Logistic Regression:
**Accuracy:** 96%
**Precision, Recall, F1-Score (for both classes):** 0.96

# Metrics Table

![image](https://github.com/user-attachments/assets/95015549-5ae4-474f-a319-9f3da9579444)

# Appendix

# Psuedocode

# Step 1: Initialize
Import necessary libraries (Librosa, Soundfile, Scikit-learn, etc.)
Load datasets: Speech and Music audio files
# Step 2: Data Preprocessing
FOR each audio file in the dataset:
    Load audio file
    Resample to 16 kHz
    Extract MFCC features
    Create feature vector by averaging MFCC features
    Assign label (0 for Speech, 1 for Music)
END FOR
Split data into training and testing sets (e.g., 80% training, 20% testing)
# Step 3: Train Models
Initialize KNN model with desired k value
Fit KNN model to training data (features and labels)
Initialize Logistic Regression model
Fit Logistic Regression model to training data
# Step 4: Evaluate Models
FOR each model (KNN, Logistic Regression):
    Predict labels for testing data
    Calculate performance metrics (Accuracy, Precision, Recall, F1-Score)
    Generate confusion matrix
END FOR
# Step 5: Interactive Testing
FUNCTION classify_audio(input_audio):
    Load the input audio file
    Resample to 16 kHz
    Extract MFCC features
    Create feature vector
    Predict label using the chosen model (KNN or Logistic Regression)
    RETURN predicted label ("Speech" or "Music")
END FUNCTION
# Step 6: Output Results
Display performance metrics and comparison between models
Visualize confusion matrices
Provide an interactive interface for testing new audio inputs
# Step 7: Future Extensions (optional)
    Suggest deep learning models for further research
    Include more diverse datasets for robustness

# Contact me
**Email address** : sheikhataurrehmansaqib@gmail.com
**Contact-no** : +92 317 7979822
