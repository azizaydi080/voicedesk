"""Tool definitions exposed to the model, and their dispatcher."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from voicedesk.domain.clinic import SPECIALTIES, ClinicStore, ToolError
from voicedesk.providers.base import Language, ToolCall, ToolSpec

_SPECIALTY = {"type": "string", "enum": list(SPECIALTIES), "description": "Medical specialty."}
_DATE = {"type": "string", "description": "YYYY-MM-DD, or 'today' / 'tomorrow'."}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "find_slots",
        "List free appointment times for a specialty on a given date.",
        {
            "type": "object",
            "properties": {
                "specialty": _SPECIALTY,
                "date": _DATE,
                "part_of_day": {"type": "string", "enum": ["morning", "afternoon"]},
            },
            "required": ["specialty", "date"],
        },
    ),
    ToolSpec(
        "book_appointment",
        "Book an appointment. Only call once you know the time, the patient's name and phone.",
        {
            "type": "object",
            "properties": {
                "specialty": _SPECIALTY,
                "date": _DATE,
                "time": {"type": "string", "description": "HH:MM, 24-hour clock."},
                "patient_name": {"type": "string"},
                "phone": {"type": "string"},
            },
            "required": ["specialty", "date", "time", "patient_name", "phone"],
        },
    ),
    ToolSpec(
        "get_appointment",
        "Look up an existing booking by its reference number.",
        {
            "type": "object",
            "properties": {"booking_ref": {"type": "string"}},
            "required": ["booking_ref"],
        },
    ),
    ToolSpec(
        "cancel_appointment",
        "Cancel an existing booking by its reference number.",
        {
            "type": "object",
            "properties": {"booking_ref": {"type": "string"}},
            "required": ["booking_ref"],
        },
    ),
]


@dataclass(frozen=True, slots=True)
class ToolResult:
    call: ToolCall
    ok: bool
    output: dict[str, Any]
    duration_ms: float

    def as_content(self) -> str:
        return json.dumps(
            self.output if self.ok else {"error": self.output["error"]}, ensure_ascii=False
        )


class ToolRunner:
    def __init__(self, store: ClinicStore) -> None:
        self.store = store

    def run(self, call: ToolCall, language: Language) -> ToolResult:
        t0 = time.perf_counter()
        a = call.arguments
        try:
            match call.name:
                case "find_slots":
                    out = self.store.find_slots(
                        str(a["specialty"]), str(a.get("date", "today")), a.get("part_of_day")
                    )
                case "book_appointment":
                    out = self.store.book(
                        str(a["specialty"]),
                        str(a["date"]),
                        str(a["time"]),
                        str(a["patient_name"]),
                        str(a["phone"]),
                    )
                case "get_appointment":
                    out = self.store.get(str(a["booking_ref"]))
                case "cancel_appointment":
                    out = self.store.cancel(str(a["booking_ref"]))
                case _:
                    raise ToolError(f"Unknown tool '{call.name}'.")
            ok = True
        except KeyError as exc:
            out, ok = {"error": f"Missing argument {exc}."}, False
        except ToolError as exc:
            out, ok = {"error": str(exc)}, False
        return ToolResult(call, ok, out, (time.perf_counter() - t0) * 1000)
