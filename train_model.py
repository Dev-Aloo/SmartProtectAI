import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

DATA_PATH = "/Users/ashritarawat/Downloads/Converted_Separately"   # Folder containing audio files
LABELS = ["scream", "non_scream"]  # two classes

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

features, labels = [], []

for label in LABELS:
    folder = os.path.join(DATA_PATH, label)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        features.append(extract_features(file_path))
        labels.append(label)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

# Save the model
pickle.dump(model, open("scream_model.pkl", "wb"))
