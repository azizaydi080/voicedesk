"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ChunkPlayer, Downsampler } from "./audio";
import { type ServerMessage, type Turn, WS_URL } from "./protocol";

export type CallState = "offline" | "connecting" | "idle" | "listening" | "thinking" | "speaking";

interface Live {
  ws: WebSocket;
  ctx: AudioContext;
  player: ChunkPlayer;
  stream?: MediaStream;
  node?: AudioWorkletNode;
  pendingAudio: Extract<ServerMessage, { type: "audio" }> | null;
  endOfInput: number; // performance.now() when the caller stopped talking / sent text
  firstAudioTurn: string | null;
}

export function useVoiceSession() {
  const [state, setState] = useState<CallState>("offline");
  const [micOn, setMicOn] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [partial, setPartial] = useState("");
  const [providers, setProviders] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const live = useRef<Live | null>(null);

  const patchTurn = useCallback((id: string, fn: (t: Turn) => Turn) => {
    setTurns((ts) => {
      const i = ts.findIndex((t) => t.id === id);
      if (i < 0) return ts;
      const copy = ts.slice();
      copy[i] = fn(ts[i]!);
      return copy;
    });
  }, []);

  const onMessage = useCallback(
    (msg: ServerMessage) => {
      const l = live.current;
      if (!l) return;
      switch (msg.type) {
        case "ready":
          setProviders(msg.providers);
          setState("idle");
          break;
        case "vad":
          if (msg.event === "speech_start") {
            l.player.stop(); // barge-in: silence the agent immediately, server cancels the turn
            setState("listening");
          } else {
            l.endOfInput = performance.now();
            setState("thinking");
          }
          break;
        case "transcript":
          if (!msg.final) {
            setPartial(msg.text);
            break;
          }
          setPartial("");
          setTurns((ts) => [
            ...ts,
            {
              id: msg.turn_id!,
              user: msg.text,
              language: msg.language,
              languageMethod: msg.language_method,
              agent: "",
              tools: [],
              interrupted: false,
            },
          ]);
          break;
        case "state":
          setState(msg.state);
          break;
        case "agent_text":
          patchTurn(msg.turn_id, (t) => ({ ...t, agent: t.agent + msg.delta }));
          break;
        case "tool_call":
          patchTurn(msg.turn_id, (t) => ({
            ...t,
            tools: [...t.tools, { id: msg.id, name: msg.name, arguments: msg.arguments, at: Date.now() }],
          }));
          break;
        case "tool_result":
          patchTurn(msg.turn_id, (t) => ({
            ...t,
            tools: t.tools.map((x) =>
              x.id === msg.id ? { ...x, ok: msg.ok, output: msg.output, duration_ms: msg.duration_ms } : x,
            ),
          }));
          break;
        case "audio":
          l.pendingAudio = msg;
          break;
        case "interrupt":
          l.player.stop();
          setTurns((ts) => (ts.length ? [...ts.slice(0, -1), { ...ts[ts.length - 1]!, interrupted: true }] : ts));
          break;
        case "turn_end":
          if (msg.metrics) patchTurn(msg.turn_id, (t) => ({ ...t, metrics: msg.metrics, interrupted: msg.metrics!.interrupted }));
          if (!l.player.playing) setState("idle");
          break;
        case "error":
          setError(msg.message);
          break;
      }
    },
    [patchTurn],
  );

  const onAudio = useCallback(async (data: ArrayBuffer) => {
    const l = live.current;
    const meta = l?.pendingAudio;
    if (!l || !meta) return;
    l.pendingAudio = null;
    const startsAt = await l.player.enqueue(data);
    if (startsAt !== null && meta.seq === 0 && l.firstAudioTurn !== meta.turn_id) {
      l.firstAudioTurn = meta.turn_id;
      const clientTtfa = Math.round(startsAt - l.endOfInput);
      l.ws.send(JSON.stringify({ type: "playback_started", turn_id: meta.turn_id, client_ttfa_ms: clientTtfa }));
    }
  }, []);

  const connect = useCallback(async () => {
    if (live.current) return;
    setError(null);
    setState("connecting");
    const ctx = new AudioContext();
    await ctx.resume();
    const ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";
    const player = new ChunkPlayer(ctx, () => {
      ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: "playback_ended" }));
      setState((s) => (s === "speaking" ? "idle" : s));
    });
    live.current = { ws, ctx, player, pendingAudio: null, endOfInput: performance.now(), firstAudioTurn: null };
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") onMessage(JSON.parse(ev.data) as ServerMessage);
      else void onAudio(ev.data as ArrayBuffer);
    };
    ws.onerror = () => setError(`Cannot reach ${WS_URL}. Is the backend running?`);
    ws.onclose = () => {
      live.current?.stream?.getTracks().forEach((t) => t.stop());
      void live.current?.ctx.close();
      live.current = null;
      setMicOn(false);
      setState("offline");
    };
  }, [onAudio, onMessage]);

  const startMic = useCallback(async () => {
    await connect();
    const l = live.current;
    if (!l || l.stream) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    await l.ctx.audioWorklet.addModule("/pcm-worklet.js");
    const source = l.ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(l.ctx, "pcm-capture");
    const down = new Downsampler(l.ctx.sampleRate);
    let pending: Int16Array[] = [];
    let pendingLen = 0;
    node.port.onmessage = (ev: MessageEvent<Float32Array>) => {
      const pcm = down.push(ev.data);
      pending.push(pcm);
      pendingLen += pcm.length;
      if (pendingLen >= 320 && l.ws.readyState === WebSocket.OPEN) {
        // ~20 ms frames keep the server's VAD and barge-in responsive.
        const frame = new Int16Array(pendingLen);
        let o = 0;
        for (const p of pending) (frame.set(p, o), (o += p.length));
        l.ws.send(frame.buffer);
        pending = [];
        pendingLen = 0;
      }
    };
    source.connect(node);
    l.stream = stream;
    l.node = node;
    setMicOn(true);
  }, [connect]);

  const stopMic = useCallback(() => {
    const l = live.current;
    if (!l) return;
    l.stream?.getTracks().forEach((t) => t.stop());
    l.node?.disconnect();
    l.stream = undefined;
    l.node = undefined;
    setMicOn(false);
  }, []);

  const sendText = useCallback(
    async (text: string) => {
      await connect();
      const l = live.current;
      if (!l) return;
      const send = () => {
        l.player.stop();
        l.endOfInput = performance.now();
        l.ws.send(JSON.stringify({ type: "text", text }));
        setState("thinking");
      };
      if (l.ws.readyState === WebSocket.OPEN) send();
      else l.ws.addEventListener("open", send, { once: true });
    },
    [connect],
  );

  const hangUp = useCallback(() => live.current?.ws.close(), []);

  useEffect(() => () => live.current?.ws.close(), []);

  return { state, micOn, turns, partial, providers, error, startMic, stopMic, sendText, hangUp, connect };
}
