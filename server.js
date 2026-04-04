const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] },
  pingTimeout: 60000,
  pingInterval: 25000,
});

// room_id → Set<socket.id>
const rooms = new Map();
// socket.id → room_id
const socketRoom = new Map();

// ── Health ────────────────────────────────────────────────────────────────────
app.get("/health", (req, res) => res.json({ status: "ok", rooms: rooms.size }));

// ── Socket.io signaling ───────────────────────────────────────────────────────
io.on("connection", (socket) => {
  console.log(`[+] Connected: ${socket.id}`);

  // ── join-room ──────────────────────────────────────────────────────────────
  socket.on("join-room", ({ roomId, userId }) => {
    if (!roomId) return;

    // Leave any previous room
    const prevRoom = socketRoom.get(socket.id);
    if (prevRoom && prevRoom !== roomId) leaveRoom(socket, prevRoom);

    if (!rooms.has(roomId)) rooms.set(roomId, new Set());
    const room = rooms.get(roomId);

    if (room.size >= 2) {
      socket.emit("room-full", { roomId });
      return;
    }

    room.add(socket.id);
    socketRoom.set(socket.id, roomId);
    socket.join(roomId);

    const peers = [...room].filter((id) => id !== socket.id);
    socket.emit("room-joined", { roomId, peers, userId });

    // Notify existing peers
    socket.to(roomId).emit("user-joined", { userId, socketId: socket.id });

    console.log(`  Room "${roomId}": ${room.size} peer(s)`);
  });

  // ── WebRTC signaling: offer / answer / ice-candidate ──────────────────────
  socket.on("offer", ({ to, offer }) => {
    io.to(to).emit("offer", { from: socket.id, offer });
  });

  socket.on("answer", ({ to, answer }) => {
    io.to(to).emit("answer", { from: socket.id, answer });
  });

  socket.on("ice-candidate", ({ to, candidate }) => {
    io.to(to).emit("ice-candidate", { from: socket.id, candidate });
  });

  // ── Chat / misc relay ─────────────────────────────────────────────────────
  socket.on("chat-message", ({ roomId, message, userId }) => {
    socket.to(roomId).emit("chat-message", { message, userId, socketId: socket.id });
  });

  socket.on("deepfake-alert", ({ roomId, result }) => {
    // Optionally broadcast detection result to the remote peer for logging
    socket.to(roomId).emit("peer-deepfake-alert", { result });
  });

  // ── Disconnect ─────────────────────────────────────────────────────────────
  socket.on("disconnecting", () => {
    const roomId = socketRoom.get(socket.id);
    if (roomId) leaveRoom(socket, roomId);
  });

  socket.on("disconnect", () => {
    console.log(`[-] Disconnected: ${socket.id}`);
  });

  socket.on("leave-room", ({ roomId }) => {
    leaveRoom(socket, roomId);
  });
});

function leaveRoom(socket, roomId) {
  const room = rooms.get(roomId);
  if (!room) return;
  room.delete(socket.id);
  socketRoom.delete(socket.id);
  socket.leave(roomId);
  socket.to(roomId).emit("user-left", { socketId: socket.id });
  if (room.size === 0) rooms.delete(roomId);
  console.log(`  Room "${roomId}": ${rooms.get(roomId)?.size ?? 0} peer(s) remaining`);
}

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => console.log(`Signaling server on http://localhost:${PORT}`));
