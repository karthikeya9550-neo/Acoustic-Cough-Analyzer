
import librosa
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification

print("Loading AST model...")

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = ASTForAudioClassification.from_pretrained(model_name)

print("AST model loaded successfully!")

audio, sr = librosa.load(
    "sample_test_audio.wav",
    sr=16000,
    mono=True
)

inputs = extractor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

print("Test completed!")
print("Detected audio category:", model.config.id2label[prediction])
