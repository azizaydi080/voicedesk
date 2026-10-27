"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { LatencyBar, LatencyLegend } from "@/components/LatencyBar";
import { BACKEND_URL, type TurnMetrics } from "@/lib/protocol";

type Stat = { n: number; p50: number | null; p90: number | null; p95: number | null };
interface MetricsResponse {
  summary: Record<string, Stat> & { turns: number; interrupted: number };
  turns: TurnMetrics[];
}

const STAGES: [string, string][] = [
  ["ttfa_ms", "Time to first audio (server)"],
  ["client_ttfa_ms", "Time to first audio (browser)"],
  ["stt_ms", "STT (after end of speech)"],
  ["llm_ttft_ms", "LLM time to first token / tool call"],
  ["tts_ttfb_ms", "TTS time to first byte"],
  ["turn_ms", "Whole turn"],
];

const fmt = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v)} ms`);

export default function Dashboard() {
  const [data, setData] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () =>
      fetch(`${BACKEND_URL}/api/metrics?limit=300`)
        .then((r) => r.json() as Promise<MetricsResponse>)
        .then((d) => alive && (setData(d), setError(null)))
        .catch(() => alive && setError(`Cannot reach ${BACKEND_URL}`));
    void load();
    const id = setInterval(load, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const turns = data?.turns.filter((t) => t.ttfa_ms != null) ?? [];
  const scale = Math.max(1000, ...turns.map((t) => (t.ttfa_ms ?? 0) + t.endpoint_silence_ms));

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>Latency</h1>
          <p className="muted">
            {data ? `${data.summary.turns} turns · ${data.summary.interrupted} interrupted` : "Loading…"} · refreshes
            every 5 s
          </p>
        </div>
        <nav>
          <Link href="/">← Back to the call</Link>
        </nav>
      </header>
      {error && <p className="error">{error}</p>}
      <main className="dash">
        <section className="card">
          <h2>Percentiles</h2>
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>n</th>
                <th>p50</th>
                <th>p90</th>
                <th>p95</th>
              </tr>
            </thead>
            <tbody>
              {STAGES.map(([key, label]) => {
                const st = data?.summary[key] as Stat | undefined;
                return (
                  <tr key={key}>
                    <td>{label}</td>
                    <td>{st?.n ?? "–"}</td>
                    <td>{fmt(st?.p50)}</td>
                    <td>{fmt(st?.p90)}</td>
                    <td>{fmt(st?.p95)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="muted small">
            Server TTFA runs from the moment the endpointer decides the caller has finished to the first audio byte
            sent. Perceived latency adds the endpointing silence window (shown hatched below).
          </p>
        </section>
        <section className="card">
          <h2>Recent turns</h2>
          <LatencyLegend />
          {turns.slice(0, 40).map((t) => (
            <div key={t.turn_id} className="lat-row wide">
              <span className="lat-label" dir="auto" title={t.transcript}>
                <b>{t.language ?? "?"}</b> {t.transcript.slice(0, 40)}
              </span>
              <LatencyBar m={t} scale={scale} />
              <span className="muted small">
                {t.tools.join(", ") || "no tool"}
                {t.cached_tokens != null && t.prompt_tokens ? ` · cache ${t.cached_tokens}/${t.prompt_tokens}` : ""}
              </span>
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}
