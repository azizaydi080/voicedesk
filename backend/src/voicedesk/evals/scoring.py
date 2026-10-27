"""Scenario schema and scoring rules for the conversation evals."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from voicedesk.domain.clinic import (
    ClinicStore,
    ToolError,
    normalize_ref,
    normalize_specialty,
    parse_date,
    parse_time,
)
from voicedesk.providers.base import Language, ToolCall

SIDE_EFFECT_TOOLS = frozenset({"book_appointment", "cancel_appointment"})
FIRST_NEW_REF = 1004  # refs below this are seeded bookings


class ExpectedCall(BaseModel):
    name: str
    args: dict[str, str | list[str]] = Field(default_factory=dict)


class Turn(BaseModel):
    user: str
    lang: Language
    voice: str | None = None
    tools: list[ExpectedCall] | None = None
    any_of_tools: list[str] | None = None
    answer_any: list[str] | None = None  # reply must mention one of these
    answer_none: list[str] | None = None  # ... and none of these


class Scenario(BaseModel):
    id: str
    turns: list[Turn]
    success: dict[str, Any] = Field(default_factory=dict)

    @property
    def group(self) -> str:
        return self.id.split("_", 1)[0]


class Suite(BaseModel):
    today: date
    scenarios: list[Scenario]


def load_suite(path: Path) -> Suite:
    return Suite.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ----------------------------------------------------------------- tool calls


def _norm_arg(key: str, value: object, today: date) -> str:
    raw = str(value)
    try:
        if key == "specialty":
            return normalize_specialty(raw)
        if key == "date":
            return parse_date(raw, today).isoformat()
        if key == "time":
            return parse_time(raw).strftime("%H:%M")
        if key == "booking_ref":
            return normalize_ref(raw)
    except ToolError:
        pass
    return raw.strip().lower()


def call_matches(expected: ExpectedCall, actual: ToolCall, today: date) -> bool:
    if expected.name != actual.name:
        return False
    for key, want in expected.args.items():
        if key not in actual.arguments:
            return False
        got = _norm_arg(key, actual.arguments[key], today)
        options = want if isinstance(want, list) else [want]
        if got not in {_norm_arg(key, o, today) for o in options}:
            return False
    return True


def score_tools(turn: Turn, calls: list[ToolCall], today: date) -> bool | None:
    """True/False if the turn has a tool expectation, None if it is not checked."""
    if turn.any_of_tools is not None:
        return any(c.name in turn.any_of_tools for c in calls)
    if turn.tools is None:
        return None
    if not turn.tools:
        return not calls
    expected_names = {e.name for e in turn.tools}
    if any(c.name in SIDE_EFFECT_TOOLS and c.name not in expected_names for c in calls):
        return False  # e.g. booking when only a lookup was asked for
    return all(any(call_matches(e, c, today) for c in calls) for e in turn.tools)


def score_answer(turn: Turn, reply: str) -> bool | None:
    """Keyword grading of the spoken answer (crude, but deterministic and free)."""
    if turn.answer_any is None and turn.answer_none is None:
        return None
    text = f" {normalize_for_cer(reply)} "
    hit = any(normalize_for_cer(k) in text for k in turn.answer_any or [""])
    miss = not any(normalize_for_cer(k) in text for k in turn.answer_none or [])
    return hit and miss


def score_success(success: dict[str, Any], store: ClinicStore) -> bool:
    new = [b for ref, b in store.bookings.items() if int(ref) >= FIRST_NEW_REF]
    if "booking" in success:
        want = success["booking"]
        if not any(
            b.status == "confirmed"
            and want.get("name_contains", "").lower() in b.patient_name.lower()
            and ("specialty" not in want or b.specialty == want["specialty"])
            and ("time" not in want or b.start.strftime("%H:%M") == want["time"])
            for b in new
        ):
            return False
    if "cancelled" in success:
        ref = str(success["cancelled"])
        if ref not in store.bookings or store.bookings[ref].status != "cancelled":
            return False
    return not (success.get("no_new_booking") and new)


# -------------------------------------------------------------------- STT CER

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي"})
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_for_cer(text: str) -> str:
    t = _DIACRITICS.sub("", text.lower()).translate(_ALEF).translate(_DIGITS)
    t = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in t)
    t = re.sub(r"(?<=\d) (?=\d)", "", t)  # "055 512" == "055512"
    return " ".join(t.split())


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)
