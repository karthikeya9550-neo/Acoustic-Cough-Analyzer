# Acoustic Cough Analyzer — starter prototype

A web interface for an experimental pediatric cough-audio research project. Audio can be recorded in the browser or selected from a local file.

## Requirements
- Python 3.10+
- pip

## Run locally
1. Open a terminal in this project folder.
2. Create a virtual environment (recommended):
   - Windows: `py -m venv .venv`
   - Activate: `.venv\\Scripts\\activate`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Start the app:
   - `python app.py`
5. Open `http://127.0.0.1:5000` in your browser.

## Audio input
Use **Start recording** to capture audio with your microphone, or choose a WAV, MP3, OGG, FLAC, or M4A file (up to 20 MB). The selected recording or file can then be previewed and analyzed.

## Important
This prototype generates experimental sound-category predictions; it does **not** diagnose a condition. Before relying on model output, use appropriately consented, clinically labeled data; split evaluation by child to avoid data leakage; and validate preprocessing and model outputs. Do not use this prototype for clinical decisions.

Uploaded files are stored in `uploads/`; delete them when no longer needed. This development server is not configured for public deployment or sensitive health data.
