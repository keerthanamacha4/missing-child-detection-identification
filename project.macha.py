import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
import numpy as np
import sqlite3
import threading
import logging
from deepface import DeepFace
from PIL import Image, ImageTk
import pyaudio
import wave
import os
import librosa
import librosa.display
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.svm import SVC  # Multiclass SVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Setup logging
logging.basicConfig(filename="app.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

class ChildIdentificationSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Missing Child Identification System")
        self.root.geometry("1280x720")
        self.root.minsize(1280, 720)

        # Image and voice paths
        self.missing_child_path = None
        self.predicted_child_path = None
        self.child_age = None
        self.voice_recording_path = "child_voice.wav"

        self.setup_ui()
        self.create_database()
        self.svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
        self.train_svm_model()

    def train_svm_model(self):
        """Train SVM model with dummy data (to be replaced with actual features)"""
        try:
            # Example feature data (Replace with actual extracted features)
            X_train = np.random.rand(10, 128)  # 10 samples, 128 feature dimensions
            #y_train = np.random.randint(0, 2, 10)  # Binary classification labels
            y_train=np.arange(len(X_train))
            self.svm_model.fit(X_train, y_train)
            messagebox.showinfo("SVM Training", "SVM model trained successfully!")
        except Exception as e:
            logging.error(f"Error training SVM model: {str(e)}")
            messagebox.showerror("Error", f"Error training SVM model: {str(e)}")

    def predict_with_svm(self, features):
        """Predict child identity using trained SVM model"""
        try:
            prediction = self.svm_model.predict([features])
            probability = self.svm_model.predict_proba([features])
            return prediction, probability
        except Exception as e:
            logging.error(f"Error in SVM prediction: {str(e)}")
            messagebox.showerror("Error", f"Error predicting with SVM model: {str(e)}")
            return None, None



    def setup_ui(self):
        """Sets up the UI components"""
        title_label = ttk.Label(self.root, text="Missing Child Identification System", font=('Helvetica', 24, 'bold'))
        title_label.pack(pady=20)

        frame = ttk.Frame(self.root)
        frame.pack()

        # Missing Child Image
        self.missing_label = ttk.Label(frame, text="No image selected", font=('Helvetica', 10))
        self.missing_label.grid(row=0, column=0, padx=20)

        missing_button = ttk.Button(frame, text="Upload Missing Child Image", command=self.upload_missing_child)
        missing_button.grid(row=1, column=0, pady=10)

        # Predicted Child Image
        self.predicted_label = ttk.Label(frame, text="No image selected", font=('Helvetica', 10))
        self.predicted_label.grid(row=0, column=1, padx=20)

        predicted_button = ttk.Button(frame, text="Upload Predicted Child Image", command=self.upload_predicted_child)
        predicted_button.grid(row=1, column=1, pady=10)

        # Age Input
        age_button = ttk.Button(self.root, text="Enter Child's Age", command=self.enter_age)
        age_button.pack(pady=10)

        # Voice Recording
        voice_button = ttk.Button(self.root, text="Record Child's Voice", command=self.record_voice)
        voice_button.pack(pady=10)

        # Voice Comparison
        compare_voice_button = ttk.Button(self.root, text="Compare Voice", command=self.compare_voice)
        compare_voice_button.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="indeterminate")
        self.progress.pack(pady=10)

        # Compare button
        compare_button = ttk.Button(self.root, text="Compare Images", command=self.threaded_compare)
        compare_button.pack(pady=10)

        # Result display
        self.result_label = ttk.Label(self.root, text="", font=('Helvetica', 16, 'bold'))
        self.result_label.pack(pady=10)

    def upload_missing_child(self):
        """Uploads and validates the missing child image"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            self.missing_child_path = file_path
            self.missing_label.config(text=os.path.basename(file_path))

    def upload_predicted_child(self):
        """Uploads and validates the predicted child image"""
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            self.predicted_child_path = file_path
            self.predicted_label.config(text=os.path.basename(file_path))

    def enter_age(self):
        """Prompts user to enter child's age"""
        self.child_age = simpledialog.askstring("Input", "Enter child's age:")
        if self.child_age:
            messagebox.showinfo("Success", f"Child's age recorded: {self.child_age}")

    def threaded_compare(self):
        """Runs image comparison in a separate thread to prevent UI freezing"""
        thread = threading.Thread(target=self.compare_images)
        thread.start()

    def compare_images(self):
        """Compares the uploaded images using DeepFace"""
        if not self.missing_child_path or not self.predicted_child_path:
            messagebox.showwarning("Warning", "Please upload both images first!")
            return

        try:
            self.progress.start()
            result = DeepFace.verify(self.missing_child_path, self.predicted_child_path, model_name="Facenet", enforce_detection=False)
            if result["verified"]:
                confidence_score = (1 - result["distance"]) * 100  # Convert distance to similarity percentage
                self.result_label.config(text=f"MATCH FOUND! Confidence: {confidence_score:.2f}%", foreground="green")
                self.progress.config(value=confidence_score)
            else:
                confidence_score = (1 - result["distance"]) * 100
                self.result_label.config(text=f"NO MATCH. Confidence: {confidence_score:.2f}%", foreground="red")
                self.progress.config(value=confidence_score)
            self.progress.stop()
        except Exception as e:
            logging.error(f"Error in comparison: {str(e)}")
            messagebox.showerror("Error", f"Error comparing images: {str(e)}")
            self.progress.stop()

    def create_database(self):
        """Creates a SQLite database for storing missing child face embeddings, age, and voice details"""
        conn = sqlite3.connect("missing_children.db")
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS children (id INTEGER PRIMARY KEY, name TEXT, age TEXT, voice TEXT, features BLOB,image_path TEXT)")
        conn.commit()
        conn.close()

    def record_voice(self):
        """Records child's voice for identification using PyAudio"""
        chunk = 1024  # Record in chunks of 1024 samples
        format = pyaudio.paInt16  # 16-bit audio format
        channels = 1  # Mono audio
        rate = 44100  # Sample rate
        record_seconds = 5  # Duration of recording
        output_filename = self.voice_recording_path

        audio = pyaudio.PyAudio()
        stream = audio.open(format=format, channels=channels,
                            rate=rate, input=True,
                            frames_per_buffer=chunk)

        messagebox.showinfo("Recording", "Recording will start now. Speak clearly.")
        frames = []

        for _ in range(0, int(rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

        messagebox.showinfo("Success", "Voice recorded successfully!")
        #messagebox.showinfo("Success", "Voice recorded successfully!")

    def extract_voice_features(self, file_path):
        """Extracts voice features using MFCC (Mel-frequency cepstral coefficients)"""
        y, sr = librosa.load(file_path, sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc, axis=1)

    def compare_voice(self):
        """Compares two recorded voices and provides a confidence score"""
        missing_voice_path = filedialog.askopenfilename(title="Select Missing Child's Voice", filetypes=[("Audio Files", "*.wav")])
        predicted_voice_path = filedialog.askopenfilename(title="Select Predicted Child's Voice", filetypes=[("Audio Files", "*.wav")])
        
        if not missing_voice_path or not predicted_voice_path:
            messagebox.showwarning("Warning", "Please select both voice files to compare.")
            return
        
        missing_features = self.extract_voice_features(missing_voice_path)
        predicted_features = self.extract_voice_features(predicted_voice_path)
        
        similarity_score = 1 - cosine(missing_features, predicted_features)
        confidence_score = similarity_score * 100  # Convert to percentage
        
        messagebox.showinfo("Voice Comparison Result", f"Voice Confidence Score: {confidence_score:.2f}%")
        
        if confidence_score > 70:  # Adjustable threshold
            self.result_label.config(text=f"Voice Match Found! Confidence: {confidence_score:.2f}%", foreground="green")
        else:
            self.result_label.config(text=f"No Voice Match. Confidence: {confidence_score:.2f}%", foreground="red")
            messagebox.showinfo("Result", "No matching voice found in database.")

def main():
    root = tk.Tk()
    app = ChildIdentificationSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()
