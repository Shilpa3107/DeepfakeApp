import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ---------- LOAD MODEL ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/deepfake_model.pth", map_location=device))
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

# ---------- LOAD FACE DETECTOR ----------
protoPath = "models/face_detector/deploy.prototxt"
modelPath = "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# ---------- VIDEO ----------
video_path = "realtime/vid3.mp4"
cap = cv2.VideoCapture(video_path)

fake_count = 0
real_count = 0

frame_skip = 10
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            img = transform(face).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img)
                _, pred = torch.max(output, 1)

            if pred.item() == 0:
                fake_count += 1
            else:
                real_count += 1

    frame_id += 1

cap.release()

# ---------- RESULT ----------
print("Real frames:", real_count)
print("Fake frames:", fake_count)

total = real_count + fake_count

if total == 0:
    print("❌ No faces detected!")
else:
    fake_ratio = fake_count / total
    print(f"Fake Ratio: {fake_ratio:.2f}")

    if fake_ratio < 0.3:
        print("LOW RISK")
    elif fake_ratio < 0.7:
        print("MEDIUM RISK")
    else:
        print("HIGH RISK")