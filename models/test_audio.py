import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/audio_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- AUDIO → SPECTROGRAM ----------
def audio_to_image(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)


    y = y[:22050*2]

    spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)

    plt.figure(figsize=(3,3))
    librosa.display.specshow(spec_db, sr=sr)
    plt.axis('off')

    temp_path = "temp.png"
    plt.savefig(temp_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = cv2.imread(temp_path)
    os.remove(temp_path)

    return img

# ---------- TEST ----------
audio_path = r"data\for-dataset\for-original\for-original\testing\fake\file36.wav"

print("Testing:", audio_path)

img = audio_to_image(audio_path)

if img is None:
    print("❌ Failed to load spectrogram")
    exit()

img = transform(img).unsqueeze(0).to(device)

# ---------- PREDICTION ----------
with torch.no_grad():
    output = model(img)
    probs = torch.softmax(output, dim=1)

fake_prob = probs[0][0].item()
real_prob = probs[0][1].item()

confidence = max(fake_prob, real_prob)

# ---------- SMART DECISION ----------
if fake_prob > 0.7:
    label = "FAKE"
elif real_prob > 0.7:
    label = "REAL"
else:
    label = "UNCERTAIN"

# ---------- RESULT ----------
print("Prediction:", label)
print(f"Confidence: {confidence*100:.2f}%")
print(f"Fake Prob: {fake_prob:.2f}, Real Prob: {real_prob:.2f}")