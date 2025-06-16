# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:27:32 2024

@author: PMYLS
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to compute MFCC features manually (simplified example)
def compute_mfcc(signal, rate, num_ceps=13):
    from scipy.fftpack import dct

    # Framing
    frame_size = 0.025  # 25ms frame
    frame_stride = 0.01  # 10ms stride
    frame_length = int(frame_size * rate)
    frame_step = int(frame_stride * rate)
    signal_length = len(signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step))

    # Pad signal
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Windowing
    frames *= np.hamming(frame_length)

    # Fourier Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Filter Banks
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz_points / rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # MFCCs
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc

# Load and preprocess audio data
def load_audio(file_path):
    rate, signal = wav.read(file_path)
    signal = signal[:rate * 3]  # Use first 3 seconds for simplicity
    return rate, signal

# Dataset creation (speech/music labels)
X = []  # Features
y = []  # Labels

# Process training data
for file_path, label in [("speech1.wav", 0), ("music1.wav", 1)]:
    rate, signal = load_audio(file_path)
    features = compute_mfcc(signal, rate)
    X.append(features.mean(axis=0))
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple classifier (e.g., k-NN manually)
def knn_classifier(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = np.linalg.norm(X_train - test_point, axis=1)
        k_indices = np.argsort(distances)[:k]
        k_labels = y_train[k_indices]
        y_pred.append(np.argmax(np.bincount(k_labels)))
    return np.array(y_pred)

# Classify and evaluate
k = 3
y_pred = knn_classifier(X_train, y_train, X_test, k)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
