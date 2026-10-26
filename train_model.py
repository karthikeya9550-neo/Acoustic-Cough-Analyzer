
import os
import glob
import json
import numpy as np
import librosa
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

AUDIO = "../SPRSound-main/BioCAS2022/train2022_wav"
ANNOTATIONS = "../SPRSound-main/BioCAS2022/train2022_json"

X, y = [], []

files = glob.glob(os.path.join(ANNOTATIONS, "*.json"))
print("Reading annotations...")

for file in files:
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        label = data.get("record_annotation", "")
        if label not in ["Normal", "CAS", "DAS", "CAS & DAS"]:
            continue

        audio_file = os.path.join(
            AUDIO, os.path.splitext(os.path.basename(file))[0] + ".wav"
        )
        if not os.path.exists(audio_file):
            continue

        audio, sr = librosa.load(audio_file, sr=16000, duration=5)

        if len(audio) < 1600:
            continue

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            [np.mean(librosa.feature.zero_crossing_rate(y=audio)),
             np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))]
        ])

        X.append(features)
        y.append(label)

        if len(y) >= 300:
            break

    except Exception as e:
        print("Skipped:", os.path.basename(file), e)

print("Usable recordings:", len(y))

if len(set(y)) < 2 or len(y) < 20:
    raise SystemExit("Not enough usable examples. Check labels and audio paths.")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, "sound_model.joblib")
print("Saved model as sound_model.joblib")
