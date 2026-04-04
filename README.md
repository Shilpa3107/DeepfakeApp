# 🛡️ DeepGuard — Real-Time Video Calling with Deepfake Detection

A full-stack application combining **WebRTC peer-to-peer video calling**, **AI-powered deepfake detection**, and a **real-time alert system**.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Browser A                         Browser B            │
│  ┌──────────┐  WebRTC P2P stream  ┌──────────┐         │
│  │ React UI │◄───────────────────►│ React UI │         │
│  └────┬─────┘                     └──────────┘         │
│       │ frames (base64)                                  │
│       ▼                                                  │
│  ┌─────────────────┐    signaling    ┌───────────────┐  │
│  │ FastAPI :8000   │                 │ Node.js :3001 │  │
│  │  /predict       │                 │  Socket.io    │  │
│  │  /feedback      │                 │  WebRTC relay │  │
│  └─────────────────┘                 └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
deepfake-detector/
├── frontend/               # React app
│   ├── src/
│   │   ├── App.js          # Main UI, call + feedback screens
│   │   ├── App.css         # Dark cyberpunk theme
│   │   └── hooks/
│   │       ├── useWebRTC.js             # WebRTC peer connection
│   │       └── useDeepfakeDetection.js  # Frame capture + API polling
│   ├── public/index.html
│   └── .env                # REACT_APP_SIGNAL_URL, REACT_APP_FASTAPI_URL
│
├── signaling-server/       # Node.js + Socket.io
│   ├── server.js
│   └── package.json
│
└── fastapi-server/         # Python deepfake inference
    ├── main.py
    └── requirements.txt
```

---

## Quick Start

### 1. FastAPI (Deepfake Detection)

```bash
cd fastapi-server
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Test it:
```bash
curl http://localhost:8000/health
```

### 2. Signaling Server (Node.js)

```bash
cd signaling-server
npm install
npm start          # or: npm run dev   (hot reload with nodemon)
```

### 3. React Frontend

```bash
cd frontend
npm install
npm start          # Opens http://localhost:3000
```

---

## Using the App

1. Open **two browser tabs** (or two devices on the same network) at `http://localhost:3000`
2. In both tabs, enter the **same Room ID** (e.g. `ROOM1`) and click **Join**
3. Grant camera/mic permissions
4. The call starts automatically when both peers join
5. **Detection Active** badge appears on the local video panel
6. Every 500 ms a frame is sent to FastAPI — the badge shows `✓ REAL` or `⚠ FAKE`
7. The bottom row of colored dots shows the last 10 predictions (majority vote is displayed on the remote video)
8. If fake confidence > 70%, an alert popup appears with `Continue Call` / `End Call`
9. After the call ends, a **feedback form** collects accuracy ratings

---

## Integrating a Real Deepfake Model

Open `fastapi-server/main.py` and replace the `predict_deepfake()` function body.

### Example — PyTorch EfficientNet

```python
from PIL import Image
import torch, torchvision.transforms as T

MODEL_PATH = "efficientnet_deepfake.pth"

# Load once at startup
_model = torch.jit.load(MODEL_PATH).eval()
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def predict_deepfake(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    fake_prob = probs[1].item()
    return {
        "prediction": "fake" if fake_prob > 0.5 else "real",
        "confidence": round(max(fake_prob, 1-fake_prob), 3),
        "reason": "Neural artifacts detected" if fake_prob > 0.5 else "No manipulation found"
    }
```

### Example — ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

_sess = ort.InferenceSession("deepfake_model.onnx")

def predict_deepfake(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32)[None] / 255.0
    out = _sess.run(None, {"input": arr.transpose(0,3,1,2)})[0]
    fake_prob = float(out[0][1])
    return {
        "prediction": "fake" if fake_prob > 0.5 else "real",
        "confidence": round(max(fake_prob, 1-fake_prob), 3),
        "reason": "ONNX model inference complete"
    }
```

---

## API Reference

### `POST /predict`

**Request:**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQ...",
  "room_id": "ROOM1",
  "user_id": "optional"
}
```

**Response:**
```json
{
  "prediction": "real",
  "confidence": 0.872,
  "reason": "Natural facial micro-movements detected",
  "processing_time_ms": 52.4
}
```

### `POST /feedback`

```json
{
  "room_id": "ROOM1",
  "accurate": true,
  "comment": "Worked perfectly",
  "prediction_history": [...],
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### `GET /feedback`

Returns last 50 feedback entries.

---

## Environment Variables

### Frontend (`frontend/.env`)
| Variable | Default | Description |
|---|---|---|
| `REACT_APP_SIGNAL_URL` | `http://localhost:3001` | Signaling server URL |
| `REACT_APP_FASTAPI_URL` | `http://localhost:8000` | FastAPI detection URL |

---

## Detection Logic

| Feature | Detail |
|---|---|
| Capture interval | Every 500 ms |
| Frame size | 320×240 JPEG at 70% quality |
| Alert threshold | `prediction == "fake"` AND `confidence > 0.7` |
| Majority voting | Last 10 predictions; >50% fake = fake verdict |
| API timeout | 4 000 ms per request (skipped on timeout) |
| Non-blocking | `processingRef` prevents concurrent frames |

---

## Performance Tips

- **GPU inference**: Add `device="cuda"` to your PyTorch model for 10-50× speedup
- **Model quantization**: Use INT8 ONNX models to reduce latency by ~3×
- **Frame skipping**: Reduce `CAPTURE_INTERVAL_MS` in `useDeepfakeDetection.js`
- **Production signaling**: Use a TURN server (e.g. Coturn) for peers behind NAT

---

## Feedback Storage

Feedback is written to `fastapi-server/feedback.json`. For production, replace with:
- **PostgreSQL** via SQLAlchemy
- **MongoDB** via Motor (async)
- **Firebase Firestore**

---

## License

MIT
