import React, { useState, useEffect, useCallback, useRef } from "react";
import { io } from "socket.io-client";
import { useWebRTC } from "./hooks/useWebRTC";
import { useDeepfakeDetection } from "./hooks/useDeepfakeDetection";
import "./App.css";

const HOSTNAME = typeof window !== "undefined" ? window.location.hostname : "localhost";
const SIGNAL_URL = process.env.REACT_APP_SIGNAL_URL || `http://${HOSTNAME}:3001`;
const FASTAPI_URL = process.env.REACT_APP_FASTAPI_URL || `http://${HOSTNAME}:8000`;

// ── Sub-components ────────────────────────────────────────────────────────────

function DeepfakeAlert({ data, onContinue, onEnd }) {
  const pct = Math.round((data?.confidence ?? 0) * 100);
  return (
    <div className="alert-overlay">
      <div className="alert-card">
        <div className="alert-icon">⚠️</div>
        <h2 className="alert-title">Possible Deepfake Detected</h2>
        <div className="alert-confidence">
          <span className="conf-label">Confidence</span>
          <div className="conf-bar-wrap">
            <div className="conf-bar" style={{ width: `${pct}%` }} />
          </div>
          <span className="conf-pct">{pct}%</span>
        </div>
        <p className="alert-reason">"{data?.reason}"</p>
        <div className="alert-actions">
          <button className="btn-continue" onClick={onContinue}>
            Continue Call
          </button>
          <button className="btn-end" onClick={onEnd}>
            End Call
          </button>
        </div>
      </div>
    </div>
  );
}

function DetectionBadge({ status, audioStatus, majorityVerdict, isDetecting }) {
  if (!isDetecting) return null;

  const verdict = majorityVerdict?.verdict ?? status?.prediction ?? "…";
  const conf = status ? Math.round(status.confidence * 100) : null;

  if (verdict === "…") {
    return (
      <div className="detection-badge badge-real" style={{ opacity: 0.7 }}>
        <span className="badge-dot" style={{ background: "grey", boxShadow: "none" }} />
        <span className="badge-label" style={{ color: "#aaa" }}>Waiting for feed...</span>
        <span className="badge-sub">Detection Active</span>
      </div>
    );
  }

  const isFake = verdict === "fake";

  return (
    <div className={`detection-badge ${isFake ? "badge-fake" : "badge-real"}`} style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', padding: '12px' }}>
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
        <span className="badge-dot" />
        <span className="badge-label">
          {isFake ? "⚠ OVERALL: FAKE" : "✓ OVERALL: REAL"}
          {conf !== null && <em> {conf}%</em>}
        </span>
      </div>
      
      {status?.reason && (
        <div style={{ fontSize: '11.5px', color: '#fff', opacity: 0.95, marginBottom: '6px', lineHeight: 1.3, textAlign: 'left' }}>
          <strong>📹 Video Reason:</strong> {status.reason}
        </div>
      )}
      
      {audioStatus?.reason && (
        <div style={{ fontSize: '11.5px', color: '#fff', opacity: 0.95, lineHeight: 1.3, textAlign: 'left' }}>
          <strong>🎙️ Audio Reason:</strong> {audioStatus.reason} 
          <em> (Conf: {Math.round(audioStatus.confidence * 100)}%)</em>
        </div>
      )}
      
      <span className="badge-sub" style={{ marginTop: '10px' }}>Tracking active across {audioStatus ? 'Audio & Video' : 'Video'} streams</span>
    </div>
  );
}

function PredictionHistory({ history }) {
  if (!history.length) return null;
  return (
    <div className="pred-history">
      {history.map((p, i) => (
        <div
          key={i}
          className={`pred-dot ${p.prediction === "fake" ? "dot-fake" : "dot-real"}`}
          title={`${p.prediction} ${Math.round(p.confidence * 100)}%`}
        />
      ))}
    </div>
  );
}

function MajorityVerdict({ data }) {
  if (!data) return null;
  const isFake = data.verdict === "fake";
  return (
    <div className={`majority-verdict ${isFake ? "mv-fake" : "mv-real"}`}>
      <span className="mv-icon">{isFake ? "🚨" : "✅"}</span>
      <span className="mv-text">
        Majority: <strong>{data.verdict.toUpperCase()}</strong>
        <em> ({data.fakeCount}/{data.total} fake)</em>
      </span>
    </div>
  );
}

function FeedbackForm({ onSubmit, onSkip }) {
  const [accurate, setAccurate] = useState(null);
  const [comment, setComment] = useState("");

  return (
    <div className="feedback-overlay">
      <div className="feedback-card">
        <h2>Call Ended — Share Feedback</h2>
        <p>Was the deepfake detection accurate?</p>
        <div className="fb-choices">
          <button
            className={`fb-btn ${accurate === true ? "fb-active" : ""}`}
            onClick={() => setAccurate(true)}
          >
            👍 Yes
          </button>
          <button
            className={`fb-btn ${accurate === false ? "fb-active fb-no" : ""}`}
            onClick={() => setAccurate(false)}
          >
            👎 No
          </button>
        </div>
        <textarea
          className="fb-comment"
          placeholder="Optional comments…"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
        <div className="fb-actions">
          <button
            className="fb-submit"
            disabled={accurate === null}
            onClick={() => onSubmit({ accurate, comment })}
          >
            Submit Feedback
          </button>
          <button className="fb-skip" onClick={onSkip}>
            Skip
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────

export default function App() {
  const [socket, setSocket] = useState(null);
  const [screen, setScreen] = useState("lobby"); // lobby | call | feedback
  const [roomId, setRoomId] = useState("");
  const [inputRoom, setInputRoom] = useState("");
  const [roomFull, setRoomFull] = useState(false);
  const [peerCount, setPeerCount] = useState(0);
  const [feedbackDone, setFeedbackDone] = useState(false);
  const historySnap = useRef([]);

  // Init socket
  useEffect(() => {
    const s = io(SIGNAL_URL, { transports: ["polling", "websocket"] });
    s.on("connect", () => {
      console.log("Socket connected:", s.id);
    });
    s.on("connect_error", (err) => {
      console.warn("Socket connect error:", err);
    });
    s.on("room-joined", ({ roomId: r, peers }) => {
      setPeerCount(peers.length);
    });
    s.on("user-joined", () => setPeerCount((c) => c + 1));
    s.on("user-left", () => setPeerCount((c) => Math.max(0, c - 1)));
    s.on("room-full", () => setRoomFull(true));
    setSocket(s);
    return () => s.disconnect();
  }, []);

  const {
    localVideoRef,
    remoteVideoRef,
    localStreamRef,
    callState,
    error: rtcError,
    startLocalStream,
    endCall,
  } = useWebRTC(socket, roomId);

  const isCallActive = screen === "call" && callState !== "ended";

  const {
    detectionStatus,
    audioStatus,
    predictionHistory,
    majorityVerdict,
    isDetecting,
    showAlert,
    alertData,
    dismissAlert,
    resetHistory,
  } = useDeepfakeDetection(localVideoRef, remoteVideoRef, isCallActive);

  // Snapshot history on call end for feedback
  useEffect(() => {
    if (callState === "ended" && screen === "call") {
      historySnap.current = predictionHistory;
      setScreen("feedback");
    }
  }, [callState, screen, predictionHistory]);

  const joinRoom = useCallback(async () => {
    const id = inputRoom.trim() || Math.random().toString(36).slice(2, 8).toUpperCase();
    setRoomId(id);
    setRoomFull(false);
    setScreen("call");
    try {
      await new Promise((resolve) => requestAnimationFrame(resolve));
      await startLocalStream();
      socket.emit("join-room", { roomId: id, userId: socket.id });
    } catch (err) {
      alert("Could not access camera/mic: " + (err.message || "Please check browser permissions or secure context."));
      setScreen("lobby");
    }
  }, [inputRoom, socket, startLocalStream]);

  const handleEndCall = useCallback(() => {
    historySnap.current = predictionHistory;
    endCall();
    setScreen("feedback");
  }, [endCall, predictionHistory]);

  const submitFeedback = useCallback(
    async ({ accurate, comment }) => {
      try {
        await fetch(`${FASTAPI_URL}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            room_id: roomId,
            accurate,
            comment,
            prediction_history: historySnap.current,
            timestamp: new Date().toISOString(),
          }),
        });
      } catch (_) {}
      setFeedbackDone(true);
      setTimeout(() => {
        setFeedbackDone(false);
        resetHistory();
        setScreen("lobby");
        setRoomId("");
        setInputRoom("");
      }, 2000);
    },
    [roomId, resetHistory]
  );

  // ── Lobby ─────────────────────────────────────────────────────────────────
  if (screen === "lobby") {
    return (
      <div className="app lobby-screen">
        <div className="lobby-bg" />
        <div className="lobby-card">
          <div className="logo">
            <span className="logo-icon">🛡️</span>
            <div>
              <h1>DeepGuard</h1>
              <p>Secure Video Calling with Deepfake Detection</p>
            </div>
          </div>
          <div className="lobby-form">
            <label>Room ID</label>
            <div className="input-row">
              <input
                type="text"
                placeholder="Enter or leave blank to auto-generate"
                value={inputRoom}
                onChange={(e) => setInputRoom(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === "Enter" && joinRoom()}
              />
              <button className="btn-join" onClick={joinRoom}>
                Join
              </button>
            </div>
            {roomFull && (
              <p className="error-msg">Room is full (max 2 participants).</p>
            )}
          </div>
          <div className="lobby-features">
            <div className="feature">
              <span>🎥</span> WebRTC P2P
            </div>
            <div className="feature">
              <span>🤖</span> AI Detection
            </div>
            <div className="feature">
              <span>⚡</span> Real-time Alerts
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ── Feedback ───────────────────────────────────────────────────────────────
  if (screen === "feedback") {
    if (feedbackDone) {
      return (
        <div className="app feedback-thanks">
          <div>✅ Thank you for your feedback!</div>
        </div>
      );
    }
    return (
      <div className="app">
        <FeedbackForm
          onSubmit={submitFeedback}
          onSkip={() => {
            resetHistory();
            setScreen("lobby");
          }}
        />
      </div>
    );
  }

  // ── Call screen ────────────────────────────────────────────────────────────
  return (
    <div className="app call-screen">
      {/* Header */}
      <header className="call-header">
        <span className="header-logo">🛡️ DeepGuard</span>
        <div className="header-room">
          Room: <strong>{roomId}</strong>
          <button 
            onClick={() => {
              navigator.clipboard.writeText(roomId);
              alert("Room ID copied to clipboard!");
            }} 
            style={{ 
              marginLeft: '10px', 
              padding: '4px 8px', 
              cursor: 'pointer', 
              fontSize: '12px',
              borderRadius: '4px',
              border: 'none',
              background: '#4CAF50',
              color: 'white'
            }}
          >
            📋 Copy ID
          </button>
          <span className={`peer-dot ${peerCount > 0 ? "peer-on" : ""}`} />
          <span className="peer-label">
            {peerCount > 0 ? `${peerCount} peer` : "Waiting…"}
          </span>
        </div>
        <button className="btn-endcall" onClick={handleEndCall}>
          ⏹ End Call
        </button>
      </header>

      {/* Videos */}
      <main className="videos-grid">
        {/* Remote */}
        <div className="video-panel panel-remote">
          <video
            id="remote-video"
            ref={remoteVideoRef}
            autoPlay
            playsInline
            className="video-el"
          />
          <span className="panel-label">Remote</span>
          <DetectionBadge
            status={detectionStatus}
            audioStatus={audioStatus}
            majorityVerdict={majorityVerdict}
            isDetecting={isDetecting}
          />
          <PredictionHistory history={predictionHistory} />
          <MajorityVerdict data={majorityVerdict} />
        </div>

        {/* Local */}
        <div className="video-panel panel-local">
          <video
            id="local-video"
            ref={localVideoRef}
            autoPlay
            playsInline
            muted
            className="video-el"
          />
          <span className="panel-label">You</span>
        </div>
      </main>

      {/* Status bar */}
      <div className="status-bar">
        <span className={`call-state-badge state-${callState}`}>
          {callState === "connected"
            ? "🟢 Connected"
            : callState === "connecting"
            ? "🟡 Connecting…"
            : "⚪ Idle"}
        </span>
        {isDetecting && (
          <span className="detection-active-tag">
            <span className="pulse" /> Detection Active
          </span>
        )}
        {rtcError && <span className="err-tag">⚠ {rtcError}</span>}
      </div>

      {/* Deepfake alert popup */}
      {showAlert && alertData && (
        <DeepfakeAlert
          data={alertData}
          onContinue={dismissAlert}
          onEnd={handleEndCall}
        />
      )}
    </div>
  );
}
