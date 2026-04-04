import { useRef, useState, useCallback, useEffect } from "react";

const ICE_SERVERS = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "stun:stun1.l.google.com:19302" },
  ],
};

export function useWebRTC(socket, roomId) {
  const localVideoRef = useRef(null);
  const remoteVideoRef = useRef(null);
  const peerConnectionRef = useRef(null);
  const localStreamRef = useRef(null);

  const [callState, setCallState] = useState("idle"); // idle | connecting | connected | ended
  const [remotePeerId, setRemotePeerId] = useState(null);
  const [error, setError] = useState(null);

  const createPeerConnection = useCallback(
    (targetId) => {
      const pc = new RTCPeerConnection(ICE_SERVERS);

      pc.onicecandidate = ({ candidate }) => {
        if (candidate && targetId) {
          socket.emit("ice-candidate", { to: targetId, candidate });
        }
      };

      pc.ontrack = (event) => {
        if (remoteVideoRef.current && event.streams[0]) {
          remoteVideoRef.current.srcObject = event.streams[0];
        }
      };

      pc.onconnectionstatechange = () => {
        const s = pc.connectionState;
        if (s === "connected") setCallState("connected");
        if (s === "disconnected" || s === "failed" || s === "closed")
          setCallState("ended");
      };

      // Add local tracks
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach((track) => {
          pc.addTrack(track, localStreamRef.current);
        });
      }

      peerConnectionRef.current = pc;
      return pc;
    },
    [socket]
  );

  // Start local media
  const startLocalStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
        audio: true,
      });
      localStreamRef.current = stream;
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }
      return stream;
    } catch (err) {
      setError("Camera/mic access denied: " + err.message);
      throw err;
    }
  }, []);

  // Socket listeners for signaling
  useEffect(() => {
    if (!socket) return;

    const onUserJoined = async ({ socketId }) => {
      setRemotePeerId(socketId);
      setCallState("connecting");
      const pc = createPeerConnection(socketId);
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      socket.emit("offer", { to: socketId, offer });
    };

    const onOffer = async ({ from, offer }) => {
      setRemotePeerId(from);
      setCallState("connecting");
      const pc = createPeerConnection(from);
      await pc.setRemoteDescription(new RTCSessionDescription(offer));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit("answer", { to: from, answer });
    };

    const onAnswer = async ({ answer }) => {
      const pc = peerConnectionRef.current;
      if (pc && pc.signalingState !== "stable") {
        await pc.setRemoteDescription(new RTCSessionDescription(answer));
      }
    };

    const onIce = async ({ candidate }) => {
      const pc = peerConnectionRef.current;
      if (pc && candidate) {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(candidate));
        } catch (_) {}
      }
    };

    const onUserLeft = () => {
      setCallState("ended");
      setRemotePeerId(null);
      if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
    };

    socket.on("user-joined", onUserJoined);
    socket.on("offer", onOffer);
    socket.on("answer", onAnswer);
    socket.on("ice-candidate", onIce);
    socket.on("user-left", onUserLeft);

    return () => {
      socket.off("user-joined", onUserJoined);
      socket.off("offer", onOffer);
      socket.off("answer", onAnswer);
      socket.off("ice-candidate", onIce);
      socket.off("user-left", onUserLeft);
    };
  }, [socket, createPeerConnection]);

  const endCall = useCallback(() => {
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
      peerConnectionRef.current = null;
    }
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((t) => t.stop());
      localStreamRef.current = null;
    }
    if (localVideoRef.current) localVideoRef.current.srcObject = null;
    if (remoteVideoRef.current) remoteVideoRef.current.srcObject = null;
    setCallState("ended");
    if (socket && roomId) socket.emit("leave-room", { roomId });
  }, [socket, roomId]);

  return {
    localVideoRef,
    remoteVideoRef,
    localStreamRef,
    callState,
    remotePeerId,
    error,
    startLocalStream,
    endCall,
  };
}
