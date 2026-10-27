"use client";

import Link from "next/link";
import { useState } from "react";
import { Conversation } from "@/components/Conversation";
import { LatencyBar, LatencyLegend } from "@/components/LatencyBar";
import { ToolTimeline } from "@/components/ToolTimeline";
import { useVoiceSession } from "@/lib/useVoiceSession";

const STATE_LABEL = {
  offline: "Not connected",
  connecting: "Connecting…",
  idle: "Your turn",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
} as const;

export default function Home() {
  const s = useVoiceSession();
  const [text, setText] = useState("");
  const measured = s.turns.filter((t) => t.metrics?.ttfa_ms != null);
  const scale = Math.max(
    1000,
    ...measured.map((t) => (t.metrics!.ttfa_ms ?? 0) + (t.metrics!.endpoint_silence_ms ?? 0)),
  );

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>
            VoiceDesk
          </h1>
          <p className="muted">Ennour Clinic receptionist · العربية · Français · English</p>
        </div>
        <nav>
          {Object.entries(s.providers).map(([k, v]) => (
            <span key={k} className="provider" title={k}>
              {k}: {v}
            </span>
          ))}
          <Link href="/dashboard">Latency dashboard →</Link>
        </nav>
      </header>

      <main className="grid">
        <section className="card call">
          <div className="controls">
            <button
              className={`mic ${s.micOn ? "on" : ""}`}
              onClick={() => (s.micOn ? s.stopMic() : void s.startMic())}
              aria-pressed={s.micOn}
            >
              {s.micOn ? "Mute" : "Start talking"}
            </button>
            <span className={`state state-${s.state}`}>{STATE_LABEL[s.state]}</span>
            {s.state !== "offline" && (
              <button className="ghost" onClick={s.hangUp}>
                Hang up
              </button>
            )}
          </div>
          {s.error && <p className="error">{s.error}</p>}
          <Conversation turns={s.turns} partial={s.partial} />
          <form
            className="typed"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim()) void s.sendText(text.trim());
              setText("");
            }}
          >
            <input
              dir="auto"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="…or type a message"
              aria-label="Type a message"
            />
            <button type="submit">Send</button>
          </form>
        </section>

        <aside className="side">
          <section className="card">
            <h2>Tool calls</h2>
            <ToolTimeline turns={s.turns} />
          </section>
          <section className="card">
            <h2>Time to first audio</h2>
            <LatencyLegend />
            {measured.length === 0 && <p className="muted">Per-turn breakdown appears after the first answer.</p>}
            {measured
              .slice(-8)
              .reverse()
              .map((t) => (
                <div key={t.id} className="lat-row">
                  <span className="lat-label" dir="auto">{t.user.slice(0, 28)}</span>
                  <LatencyBar m={t.metrics!} scale={scale} />
                </div>
              ))}
          </section>
        </aside>
      </main>
    </div>
  );
}
