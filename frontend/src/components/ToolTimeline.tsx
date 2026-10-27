import type { Turn } from "@/lib/protocol";

function summarize(output: Record<string, unknown> | undefined): string {
  if (!output) return "running…";
  if ("error" in output) return String(output.error);
  if (Array.isArray(output.available_times)) return `free: ${output.available_times.join(", ") || "none"}`;
  if ("booking_ref" in output) return `#${output.booking_ref} · ${output.status} · ${output.date} ${output.time}`;
  return JSON.stringify(output);
}

export function ToolTimeline({ turns }: { turns: Turn[] }) {
  const events = turns.flatMap((t) => t.tools).reverse();
  if (!events.length) return <p className="muted">Tool calls will appear here.</p>;
  return (
    <ol className="timeline">
      {events.map((e) => (
        <li key={e.id} className={e.ok === false ? "fail" : e.ok ? "ok" : "pending"}>
          <div className="tl-head">
            <code>{e.name}</code>
            {e.duration_ms !== undefined && <span className="muted">{e.duration_ms.toFixed(1)} ms</span>}
          </div>
          <code className="args" dir="auto">{JSON.stringify(e.arguments)}</code>
          <div className="tl-out" dir="auto">{summarize(e.output)}</div>
        </li>
      ))}
    </ol>
  );
}
