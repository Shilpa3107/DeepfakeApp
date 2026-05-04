# 🛡️ DeepGuard: Real-Time Deepfake Detection System
## Comprehensive Project Documentation

---

### 1. Introduction and Core Motivation

The rapid advancement of generative Artificial Intelligence has democratized the creation of highly realistic synthetic media, commonly known as "deepfakes." While this technology has legitimate applications in entertainment and art, it has also given rise to a severe, unprecedented threat vector in cybersecurity. 

The core reason for developing **DeepGuard** is to combat this emerging threat by providing a robust, real-time deepfake detection mechanism specifically tailored for live video communications. As remote work, telehealth, and digital socializing become standard, video calls have become the primary medium of human interaction. The risk of impersonation and fraud using real-time deepfake technology has multiplied exponentially. Existing deepfake detection tools are often offline, requiring users to upload a video for analysis. DeepGuard was built to act as an active, invisible shield that operates *during* the call, ensuring the authenticity of the person on the other end of the screen without disrupting the user experience.

---

### 2. Societal Impact and Problems Solved

Deepfakes present a myriad of societal challenges. DeepGuard aims to mitigate several critical issues:

*   **Financial Fraud and Identity Theft:** Malicious actors use deepfakes to impersonate executives, politicians, or family members to authorize fraudulent wire transfers or steal highly sensitive intellectual property. DeepGuard provides immediate verification to prevent these targeted social engineering attacks.
*   **Misinformation and Propaganda:** Deepfakes can be used to create convincing fake statements by public figures. If broadcasted live or used in critical virtual meetings, they can manipulate public opinion or cause financial panic.
*   **Cyberbullying and Harassment:** The platform protects individuals from interacting with malicious entities utilizing non-consensual deepfake generation during live interactions.
*   **Restoring Trust in Digital Communication:** Ultimately, DeepGuard restores trust in remote communications by providing cryptographically and visually verified interactions, assuring users that they are talking to a genuine human whose voice and face have not been artificially synthesized.

---

### 3. The DeepGuard Solution

DeepGuard provides a seamless, real-time video calling platform equipped with an integrated, dual-modality AI verification system. Its key features include:

*   **Secure WebRTC Environment:** A customized, peer-to-peer video communication setup that ensures end-to-end privacy for the video streams.
*   **Continuous Background Monitoring:** The system samples video frames and audio chunks continuously in the background. It does not wait for a call to end to provide a verdict.
*   **Dual-Modality Analysis:** It analyzes both visual anomalies (such as unnatural blending artifacts, facial boundaries) and audio anomalies (synthetic voice patterns, unnatural frequencies).
*   **Intelligent Alerting System:** If the system detects a high probability of a deepfake, it immediately warns the user with a visual overlay alert and a confidence score, giving them the option to terminate the call instantly.
*   **Rolling History & Majority Voting:** To prevent false positives caused by brief network glitches or lighting changes, the system uses a rolling history of the last 10 predictions and relies on a majority verdict.

---

### 4. Technology Stack

The project utilizes a modern, decoupled architecture split across three main domains:

#### Frontend (User Interface & Capture)
*   **React.js:** Used for building a reactive, component-based user interface.
*   **WebRTC API:** For establishing low-latency, peer-to-peer live video and audio streaming.
*   **Socket.io-client:** For real-time connection signaling.
*   **Vanilla CSS:** Used to craft a highly customized, cyberpunk-inspired dark theme with dynamic micro-animations.

#### Signaling Server (Connection Management)
*   **Node.js & Express:** Lightweight backend to host the signaling infrastructure.
*   **Socket.io:** Facilitates the WebRTC connection handshake (exchange of SDP offers, answers, and ICE candidates) between peers.

#### AI Backend (Inference & Analysis)
*   **Python & FastAPI:** A high-performance asynchronous API server designed to handle high-frequency detection requests from the frontend.
*   **PyTorch & Torchvision:** Deep learning frameworks used for training and running the classification models.
*   **OpenCV:** Used for frame processing, image manipulation, and running the Caffe-based Face Detector.
*   **Librosa:** Advanced audio processing library used to generate Mel-spectrograms from incoming audio chunks.

#### Machine Learning Models
*   **Face Detection:** OpenCV DNN module utilizing a Caffe SSD Model (`res10_300x300_ssd_iter_140000.caffemodel`).
*   **Visual Detection:** A ResNet18 Convolutional Neural Network (CNN) fine-tuned specifically for detecting visual deepfake artifacts.
*   **Audio Detection:** A secondary ResNet18 CNN fine-tuned to classify images of audio Mel-spectrograms to differentiate human voices from AI-generated speech.

---

### 5. Methodology and Pipeline

The development of DeepGuard followed a systematic approach combining computer vision, audio processing, and real-time networking. The operational pipeline works as follows:

1.  **Call Initiation:** Users join a room via the React UI. The Node.js signaling server coordinates the WebRTC peer-to-peer connection.
2.  **Data Capture Strategy:**
    *   The React frontend hooks into the *remote peer's* media stream (ensuring it only analyzes the incoming data, not the local user).
    *   **Video:** Frames are captured from an invisible HTML canvas every 500 milliseconds and compressed to Base64 JPEG formats to optimize network payload.
    *   **Audio:** Audio is recorded using the `MediaRecorder` API in 2-second chunks and converted to Base64 strings.
3.  **Visual Inference Pipeline:**
    *   The FastAPI server receives the Base64 frame and decodes it into a NumPy array.
    *   The OpenCV Caffe model detects faces within the frame.
    *   Detected faces are cropped, resized to 224x224 pixels, and normalized.
    *   The PyTorch ResNet18 model evaluates the facial tensor and outputs probability logits for "real" and "fake" classes.
4.  **Audio Inference Pipeline:**
    *   Base64 audio chunks are decoded and processed by Librosa.
    *   Librosa generates a Mel-spectrogram, mapping the frequency domain over time.
    *   This spectrogram is plotted as an image and passed to the secondary PyTorch ResNet18 model to classify real vs. AI-synthesized audio patterns.
5.  **Decision & Aggregation:**
    *   The UI receives the API responses and maintains a rolling window of the last 10 predictions.
    *   A majority voting algorithm determines the overall status badge (Real/Fake) to smooth out erratic predictions.
    *   A critical threshold (e.g., confidence > 70%) triggers an immediate blocking UI alert overlay.

---

### 6. Challenges Faced During Development

Building a real-time AI inference system in the browser presented several significant hurdles:

*   **Real-Time Latency vs. Model Accuracy:** Running complex deep learning models on every single frame introduces massive latency, making the call unwatchable. 
    *   *Solution:* We implemented frame throttling (capturing every 500ms instead of 30fps) and utilized lightweight architectures like ResNet18 rather than heavier models (e.g., EfficientNet-B7 or Vision Transformers), balancing speed and accuracy.
*   **Audio-Video Synchronization in Browser Capture:** Extracting synchronized audio chunks from a remote WebRTC stream was highly challenging due to strict browser security policies and the asynchronous nature of the `MediaRecorder` API.
    *   *Solution:* We decoupled the video and audio polling loops in the `useDeepfakeDetection` React hook, allowing them to run on separate intervals and report to the UI independently without blocking one another.
*   **Model Overfitting and Generalization:** Initially, the visual model overfitted to specific backgrounds or lighting conditions found in the training dataset, failing in real-world webcam conditions.
    *   *Solution:* Heavy data augmentation techniques (Random Horizontal Flips, Rotations, Color Jittering) were introduced in the PyTorch `train_model.py` pipeline to ensure the model focuses strictly on facial artifacts rather than environmental noise.
*   **Spectrogram Generation Overhead:** Converting audio to Mel-spectrograms using Matplotlib inside an API route was slow and CPU-intensive.
    *   *Solution:* We optimized the plot rendering by disabling axes, using `Agg` backend for Matplotlib, and directly piping bytes in memory via `io.BytesIO()` instead of writing to disk.

---

### 7. Conclusion and Future Scope

DeepGuard successfully demonstrates a functional, highly responsive protective layer against digital impersonation. By utilizing a dual-modality approach (analyzing both audio and video independently), it significantly reduces the chances of sophisticated deepfakes bypassing the system. The project proves that complex AI verification can be integrated into consumer-facing applications without ruining the user experience.

**Future Enhancements could include:**
1.  **WebAssembly (WASM) Integration:** Porting the ONNX models directly into the browser via WebAssembly to completely eliminate network API latency and privacy concerns (all data stays client-side).
2.  **Temporal Analysis:** Incorporating recurrent architectures like LSTMs or 3D-CNNs to detect unnatural micro-movements or temporal flickering across multiple consecutive frames.
3.  **Enterprise Integrations:** Creating browser extensions or plugins for unified communication platforms like Zoom, Google Meet, and Microsoft Teams to provide universal deepfake protection.
