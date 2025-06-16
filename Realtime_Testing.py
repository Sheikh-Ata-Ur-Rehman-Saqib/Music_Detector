import sounddevice as sd
import librosa
import numpy as np
import joblib
import tkinter as tk
from tkinter import messagebox, filedialog
import soundfile as sf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

#Load models
knn_model = joblib.load("./knn_model.pkl")
log_reg_model = joblib.load("./log_reg_model.pkl")

#MFCC features
def extract_features(audio, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

#Predicting
def predict_audio_label(audio_data):
    features = extract_features(audio_data)
    knn_prediction = knn_model.predict([features])
    log_reg_prediction = log_reg_model.predict([features])
    knn_label = 'Speech' if knn_prediction == 0 else 'Music'
    log_reg_label = 'Speech' if log_reg_prediction == 0 else 'Music'
    return knn_label, log_reg_label

def load_audio_file(file_path, sr=16000):
    try:
        audio, sr_native = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"Error loading audio with librosa: {e}")
        try:
            audio, sr_native = sf.read(file_path)
            audio = librosa.resample(audio, sr_native, sr)
        except Exception as e:
            print(f"Error loading audio with soundfile: {e}")
            audio = None
            sr_native = None
    return audio, sr_native

class AudioClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Classifier")

        self.instructions = tk.Label(root, text="Select an option to start.", font=("Helvetica", 14))
        self.instructions.pack(pady=10)

        self.record_button = tk.Button(root, text="Record Audio", font=("Helvetica", 14), command=self.record_audio_option)
        self.record_button.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Audio", font=("Helvetica", 14), command=self.upload_audio_option)
        self.upload_button.pack(pady=10)

        self.prediction_label = tk.Label(root, text="", font=("Helvetica", 14))
        self.prediction_label.pack(pady=10)

        self.recording_window = None

        self.audio_data = None
        self.is_playing = False  # Track playing state

    def record_audio_option(self):
        self.instructions.config(text="Recording options: Start/Stop recording.")
        self.record_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)

        if not self.recording_window:
            self.recording_window = tk.Toplevel(self.root)
            self.recording_window.title("Record Audio")

        self.start_button = tk.Button(self.recording_window, text="Start Recording", font=("Helvetica", 14), command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.recording_window, text="Stop Recording", font=("Helvetica", 14), state=tk.DISABLED, command=self.stop_recording)
        self.stop_button.pack(pady=10)

    def upload_audio_option(self):
        self.instructions.config(text="Select an audio file to upload.")
        self.record_button.config(state=tk.DISABLED)
        self.upload_button.config(state=tk.DISABLED)

        file_path = filedialog.askopenfilename(title="Select an Audio File", filetypes=[("Audio Files", "*.wav;*.mp3")])
        if file_path:
            try:
                audio, sr = load_audio_file(file_path, sr=16000)
                if audio is not None:
                    self.audio_data = audio
                    self.prediction_label.config(text="Audio file selected.")
                    self.show_upload_options()
                else:
                    messagebox.showerror("Error", "Error processing file.")
                    self.prediction_label.config(text="")
            except Exception as e:
                messagebox.showerror("Error", f"Error processing file: {e}")
                self.prediction_label.config(text="")

    def start_recording(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.instructions.config(text="Recording in progress...")

        self.audio_data = self.record_audio(duration=5)

        if self.audio_data is None:
            self.reset_recording()
            messagebox.showerror("Error", "Recording failed.")
            return

        self.prediction_label.config(text="Recording captured.")
        self.show_recording_options()

    def stop_recording(self):
        sd.stop()
        self.instructions.config(text="Recording stopped.")
        self.show_recording_options()

    def record_audio(self, duration=15, fs=16000):
        try:
            print("Recording...")
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
            sd.wait()
            return audio_data.flatten()
        except Exception as e:
            print(f"Error during recording: {e}")
            return None

    def show_recording_options(self):
        if not hasattr(self, 'record_again_button'):
            self.record_again_button = tk.Button(self.recording_window, text="Record Again", font=("Helvetica", 14), command=self.record_audio_option)
            self.record_again_button.pack(pady=10)

        if not hasattr(self, 'predict_button'):
            self.predict_button = tk.Button(self.recording_window, text="Predict", font=("Helvetica", 14), command=self.predict_audio)
            self.predict_button.pack(pady=10)

        if not hasattr(self, 'play_button'):
            self.play_button = tk.Button(self.recording_window, text="Play", font=("Helvetica", 14), command=self.play_audio)
            self.play_button.pack(pady=10)

        if not hasattr(self, 'stop_button'):
            self.stop_button = tk.Button(self.recording_window, text="Stop Playing", font=("Helvetica", 14), state=tk.DISABLED, command=self.stop_playing)
            self.stop_button.pack(pady=10)

    def show_upload_options(self):
        if not hasattr(self, 'upload_again_button'):
            self.upload_again_button = tk.Button(self.recording_window, text="Upload Again", font=("Helvetica", 14), command=self.upload_audio_option)
            self.upload_again_button.pack(pady=10)

        if not hasattr(self, 'predict_button'):
            self.predict_button = tk.Button(self.recording_window, text="Predict", font=("Helvetica", 14), command=self.predict_audio)
            self.predict_button.pack(pady=10)

        if not hasattr(self, 'play_button'):
            self.play_button = tk.Button(self.recording_window, text="Play", font=("Helvetica", 14), command=self.play_audio)
            self.play_button.pack(pady=10)

        if not hasattr(self, 'stop_button'):
            self.stop_button = tk.Button(self.recording_window, text="Stop Playing", font=("Helvetica", 14), state=tk.DISABLED, command=self.stop_playing)
            self.stop_button.pack(pady=10)

    def predict_audio(self):
        knn_label, log_reg_label = predict_audio_label(self.audio_data)
        self.prediction_label.config(text=f"KNN Model Prediction: {knn_label}\nLogistic Regression Prediction: {log_reg_label}")
        self.show_go_back_option()

    def play_audio(self):
        if not self.is_playing:
            audio_to_play = self.audio_data[:160000]  # 16000 samples/sec * 10 sec = 160000 samples
            sd.play(audio_to_play, 16000)
            self.is_playing = True
            self.play_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            sd.wait()

    def stop_playing(self):
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def show_go_back_option(self):
        if not hasattr(self, 'go_back_button'):
            self.go_back_button = tk.Button(self.recording_window, text="Go Back", font=("Helvetica", 14), command=self.go_back)
            self.go_back_button.pack(pady=10)         

    def go_back(self):
        if self.recording_window:
            self.recording_window.destroy()
            self.recording_window = None
        self.instructions.config(text="Select an option to start.")
        self.record_button.config(state=tk.NORMAL)
        self.upload_button.config(state=tk.NORMAL)
        self.prediction_label.config(text="")

root = tk.Tk()
app = AudioClassifierApp(root)

root.mainloop()
