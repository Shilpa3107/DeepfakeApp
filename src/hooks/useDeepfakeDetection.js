import { useRef, useState, useCallback, useEffect } from "react";

const FASTAPI_URL = process.env.REACT_APP_FASTAPI_URL || "http://localhost:8000";
const CAPTURE_INTERVAL_MS = 500;
const HISTORY_SIZE = 10;

export function useDeepfakeDetection(localVideoRef, remoteVideoRef, isCallActive) {
  const [detectionStatus, setDetectionStatus] = useState(null);
  const [audioStatus, setAudioStatus] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [majorityVerdict, setMajorityVerdict] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [showAlert, setShowAlert] = useState(false);
  const [alertData, setAlertData] = useState(null);

  const intervalRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const processingRef = useRef(false);
  const audioProcessingRef = useRef(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const captureFrame = useCallback(() => {
    // Only capture from the remote peer's video!
    const videoEl = remoteVideoRef?.current;
    if (!videoEl || !videoEl.videoWidth) return null;

    const canvas = canvasRef.current;
    canvas.width = 320;
    canvas.height = 240;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.7);
  }, [localVideoRef, remoteVideoRef]);

  const runDetection = useCallback(async () => {
    if (processingRef.current) return;
    processingRef.current = true;

    const frame = captureFrame();
    if (!frame) {
      processingRef.current = false;
      return;
    }

    try {
      const res = await fetch(`${FASTAPI_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frame }),
        signal: AbortSignal.timeout(4000),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setDetectionStatus(data);

      setPredictionHistory((prev) => {
        const next = [...prev, data].slice(-HISTORY_SIZE);

        // Majority voting
        const fakeCount = next.filter((p) => p.prediction === "fake").length;
        const verdict = fakeCount > next.length / 2 ? "fake" : "real";
        const avgConf =
          next.reduce((s, p) => s + p.confidence, 0) / next.length;

        setMajorityVerdict({ verdict, fakeCount, total: next.length, avgConf });

        // Trigger alert
        if (data.prediction === "fake" && data.confidence > 0.7) {
          setAlertData(data);
          setShowAlert(true);
        }

        return next;
      });
    } catch (err) {
      // Silently ignore individual frame failures
    } finally {
      processingRef.current = false;
    }
  }, [captureFrame]);

  // Start/stop detection loop
  useEffect(() => {
    let audioAttachInterval;

    if (isCallActive) {
      setIsDetecting(true);
      intervalRef.current = setInterval(runDetection, CAPTURE_INTERVAL_MS);
      
      // Periodically check for the remote person's stream so we only analyze THEM
      audioAttachInterval = setInterval(() => {
        const remoteVideoEl = remoteVideoRef?.current;
        if (remoteVideoEl && remoteVideoEl.srcObject && !mediaRecorderRef.current) {
           try {
               const audioStream = new MediaStream(remoteVideoEl.srcObject.getAudioTracks());
               if (audioStream.getTracks().length > 0) {
                   mediaRecorderRef.current = new MediaRecorder(audioStream, { mimeType: 'audio/webm' });
                   mediaRecorderRef.current.ondataavailable = async (e) => {
                       if (e.data.size > 0) {
                           const reader = new FileReader();
                           reader.readAsDataURL(e.data);
                           reader.onloadend = async () => {
                               if (audioProcessingRef.current) return;
                               audioProcessingRef.current = true;
                               const base64Audio = reader.result;
                               try {
                                   const res = await fetch(`${FASTAPI_URL}/predict-audio`, {
                                       method: "POST",
                                       headers: { "Content-Type": "application/json" },
                                       body: JSON.stringify({ audio: base64Audio }),
                                       signal: AbortSignal.timeout(4000),
                                   });
                                   if (res.ok) {
                                       const data = await res.json();
                                       setAudioStatus(data);
                                       console.log("Audio prediction:", data);
                                       if (data.prediction === "fake" && data.confidence > 0.6) {
                                           setAlertData({ ...data, type: "audio" });
                                           setShowAlert(true);
                                       }
                                   }
                               } catch (err) {}
                               finally { audioProcessingRef.current = false; }
                           };
                       }
                   };
                   mediaRecorderRef.current.start(2000); // 2 seconds chunks
                   clearInterval(audioAttachInterval);
               }
           } catch(e) { console.error("Audio recorder error:", e); }
        }
      }, 1000);

    } else {
      setIsDetecting(false);
      clearInterval(intervalRef.current);
      clearInterval(audioAttachInterval);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
      }
      mediaRecorderRef.current = null;
    }

    return () => {
      clearInterval(intervalRef.current);
      clearInterval(audioAttachInterval);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
          mediaRecorderRef.current.stop();
      }
    };
  }, [isCallActive, runDetection, remoteVideoRef]);

  const dismissAlert = useCallback(() => setShowAlert(false), []);
  const resetHistory = useCallback(() => {
    setPredictionHistory([]);
    setMajorityVerdict(null);
    setDetectionStatus(null);
    setShowAlert(false);
  }, []);

  return {
    detectionStatus,
    audioStatus,
    predictionHistory,
    majorityVerdict,
    isDetecting,
    showAlert,
    alertData,
    dismissAlert,
    resetHistory,
  };
}
