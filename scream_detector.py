import numpy as np
import sounddevice as sd
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("scream_model.pkl")

def extract_features_from_audio(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean.reshape(1, -1)

print("🎧 Listening for screams... (Ctrl+C to stop)")
duration = 2  # seconds per chunk
threshold = 0.65  # Probability threshold

while True:
    try:
        audio = sd.rec(int(22050 * duration), samplerate=22050, channels=1)
        sd.wait()
        audio = np.squeeze(audio)

        features = extract_features_from_audio(audio, 22050)
        probs = model.predict_proba(features)[0]
        pred = np.argmax(probs)
        conf = probs[pred]

        if pred == 1 and conf > threshold:
            print(f"😶 No scream (confidence: {conf:.2f})")
        else:
            print(f"🚨 SCREAM DETECTED!  ({conf:.2f})")

    except KeyboardInterrupt:
        print("\n🛑 Stopped listening.")
        break
