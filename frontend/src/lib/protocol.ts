// Mirrors the server's WebSocket messages (backend/src/voicedesk/api/app.py).

export type Lang = "ar" | "fr" | "en";

export interface TurnMetrics {
  turn_id: string;
  input_mode: "voice" | "text";
  providers: Record<"stt" | "llm" | "tts", string>;
  started_at: number;
  endpoint_silence_ms: number;
  audio_seconds: number | null;
  transcript: string;
  language: Lang | null;
  language_method: string | null;
  reply: string;
  tools: string[];
  llm_rounds: number;
  prompt_tokens: number | null;
  cached_tokens: number | null;
  stt_ms: number | null;
  llm_ttft_ms: number | null;
  llm_total_ms: number | null;
  tool_ms: number | null;
  tts_ttfb_ms: number | null;
  tts_first_chunk_ms: number | null;
  ttfa_ms: number | null;
  client_ttfa_ms: number | null;
  turn_ms: number | null;
  interrupted: boolean;
  error: string | null;
}

export type ServerMessage =
  | { type: "ready"; session_id: string; providers: Record<string, string>; today: string }
  | { type: "vad"; event: "speech_start" | "speech_end" }
  | {
      type: "transcript";
      final: boolean;
      text: string;
      turn_id?: string;
      language?: Lang;
      language_method?: string;
      stt_language?: string | null;
    }
  | { type: "state"; state: "thinking" | "speaking"; turn_id: string }
  | { type: "agent_text"; turn_id: string; delta: string }
  | { type: "tool_call"; turn_id: string; id: string; name: string; arguments: Record<string, unknown> }
  | {
      type: "tool_result";
      turn_id: string;
      id: string;
      ok: boolean;
      output: Record<string, unknown>;
      duration_ms: number;
    }
  | { type: "audio"; turn_id: string; seq: number; format: string; text: string; bytes: number }
  | { type: "interrupt"; reason: string }
  | { type: "turn_end"; turn_id: string; empty?: boolean; metrics?: TurnMetrics }
  | { type: "error"; turn_id?: string; message: string };

export interface ToolEvent {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  ok?: boolean;
  output?: Record<string, unknown>;
  duration_ms?: number;
  at: number;
}

export interface Turn {
  id: string;
  user: string;
  language?: Lang;
  languageMethod?: string;
  agent: string;
  tools: ToolEvent[];
  metrics?: TurnMetrics;
  interrupted: boolean;
}

export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";
export const WS_URL = BACKEND_URL.replace(/^http/, "ws") + "/ws";
