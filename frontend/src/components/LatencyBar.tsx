import type { TurnMetrics } from "@/lib/protocol";

/** Decomposes time-to-first-audio into STT, agent (LLM + tools) and first TTS chunk. */
export function latencyParts(m: TurnMetrics) {
  const stt = m.stt_ms ?? 0;
  const tts = m.tts_first_chunk_ms ?? 0;
  const ttfa = m.ttfa_ms ?? 0;
  return { endpoint: m.endpoint_silence_ms ?? 0, stt, agent: Math.max(0, ttfa - stt - tts), tts, ttfa };
}

export function LatencyBar({ m, scale }: { m: TurnMetrics; scale: number }) {
  const p = latencyParts(m);
  const pct = (v: number) => `${(100 * v) / scale}%`;
  return (
    <div className="lat">
      <div className="lat-bar" role="img" aria-label={`time to first audio ${Math.round(p.ttfa)} ms`}>
        <span className="seg seg-endpoint" style={{ width: pct(p.endpoint) }} title={`endpointing wait ${p.endpoint} ms`} />
        <span className="seg seg-stt" style={{ width: pct(p.stt) }} title={`STT ${Math.round(p.stt)} ms`} />
        <span className="seg seg-agent" style={{ width: pct(p.agent) }} title={`LLM + tools ${Math.round(p.agent)} ms`} />
        <span className="seg seg-tts" style={{ width: pct(p.tts) }} title={`first TTS chunk ${Math.round(p.tts)} ms`} />
      </div>
      <span className="lat-total">{m.ttfa_ms == null ? "–" : `${(m.ttfa_ms / 1000).toFixed(2)} s`}</span>
    </div>
  );
}

export function LatencyLegend() {
  return (
    <div className="legend">
      <span><i className="seg-endpoint" /> endpointing</span>
      <span><i className="seg-stt" /> STT</span>
      <span><i className="seg-agent" /> LLM + tools</span>
      <span><i className="seg-tts" /> first TTS chunk</span>
    </div>
  );
}
