
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
import joblib

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sound_model.joblib")
model = joblib.load(MODEL_PATH)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_model_features(signal, sr):
    # MUST match the features used during training
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        [
            np.mean(librosa.feature.zero_crossing_rate(y=signal)),
            np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
        ]
    ]).reshape(1, -1)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "Please choose an audio file."}), 400

    audio = request.files["audio"]

    if not audio.filename:
        return jsonify({"error": "Please choose an audio file."}), 400

    if not allowed_file(audio.filename):
        return jsonify({"error": "Unsupported audio format."}), 400

    filename = secure_filename(audio.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    audio.save(path)

    try:
        signal, sr = librosa.load(
            path, sr=16000, mono=True, duration=5
        )

        if len(signal) < 1600:
            return jsonify({"error": "Audio is too short. Use at least 0.1 seconds."}), 400

        # Extract the same features used for model training
        model_features = extract_model_features(signal, sr)
        prediction = str(model.predict(model_features)[0])

        probabilities = {}
        if hasattr(model, "predict_proba"):
            probabilities = {
                str(label): round(float(prob) * 100, 1)
                for label, prob in zip(
                    model.classes_,
                    model.predict_proba(model_features)[0]
                )
            }

        # Extra audio features for the website
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        mel = librosa.feature.melspectrogram(
            y=signal, sr=sr, n_mels=64
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(signal)

        return jsonify({
            "status": "success",
            "filename": filename,
            "duration_seconds": round(len(signal) / sr, 2),
            "sample_rate": sr,
            "audio_samples": len(signal),
            "prediction": prediction,
            "probabilities": probabilities,
            "features": {
                "mfcc_coefficients": int(mfcc.shape[0]),
                "mel_bands": int(mel_db.shape[0]),
                "spectral_centroid_hz": round(float(np.mean(centroid)), 2),
                "spectral_bandwidth_hz": round(float(np.mean(bandwidth)), 2),
                "zero_crossing_rate": round(float(np.mean(zcr)), 4)
            },
            "message": (
                "Experimental sound-category prediction only. "
                "This is not a medical diagnosis."
            )
        })

    except Exception as e:
        app.logger.exception("Audio analysis failed")
        return jsonify({
            "error": "Could not process this audio. Try a valid WAV file."
        }), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        "error": "File is too large. Maximum size is 20 MB."
    }), 413


if __name__ == "__main__":
    app.run(debug=True)
