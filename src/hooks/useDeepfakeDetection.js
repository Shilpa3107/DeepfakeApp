import { useRef, useState, useCallback, useEffect } from "react";

const FASTAPI_URL = process.env.REACT_APP_FASTAPI_URL || "http://localhost:8000";
const CAPTURE_INTERVAL_MS = 500;
const HISTORY_SIZE = 10;

export function useDeepfakeDetection(localVideoRef, remoteVideoRef, isCallActive) {
  const [detectionStatus, setDetectionStatus] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [majorityVerdict, setMajorityVerdict] = useState(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [showAlert, setShowAlert] = useState(false);
  const [alertData, setAlertData] = useState(null);

  const intervalRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));
  const processingRef = useRef(false);

  const captureFrame = useCallback(() => {
    const videoEl =
      remoteVideoRef?.current?.videoWidth
        ? remoteVideoRef.current
        : localVideoRef?.current?.videoWidth
        ? localVideoRef.current
        : null;
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
    if (isCallActive) {
      setIsDetecting(true);
      intervalRef.current = setInterval(runDetection, CAPTURE_INTERVAL_MS);
    } else {
      setIsDetecting(false);
      clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [isCallActive, runDetection]);

  const dismissAlert = useCallback(() => setShowAlert(false), []);
  const resetHistory = useCallback(() => {
    setPredictionHistory([]);
    setMajorityVerdict(null);
    setDetectionStatus(null);
    setShowAlert(false);
  }, []);

  return {
    detectionStatus,
    predictionHistory,
    majorityVerdict,
    isDetecting,
    showAlert,
    alertData,
    dismissAlert,
    resetHistory,
  };
}
