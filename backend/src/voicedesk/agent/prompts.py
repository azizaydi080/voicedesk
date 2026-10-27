"""System prompt and few-shot priming.

Everything here is static for a given day, so it sits in the LLM's KV / prompt cache:
after the first turn of a call, clinic facts and examples cost no prefill time. Static
knowledge therefore lives in the prompt; tools are reserved for dynamic state (free
slots, bookings), which keeps FAQ answers to a single LLM round.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

from voicedesk.domain.clinic import CLINIC_NAME
from voicedesk.domain.faq import FAQ
from voicedesk.providers.base import Language, Message, ToolCall

LANGUAGE_NAMES: dict[Language, str] = {
    "ar": "Arabic (simple Modern Standard Arabic that Gulf and Maghrebi callers understand)",
    "fr": "French",
    "en": "English",
}

# Tunisian Derja words a 1-3B model will not reliably know.
_DERJA = "نبطّل / نفسخ = cancel · غدوة = tomorrow · العشية = afternoon · الصباح = morning · النومرو = number · السنان = teeth · وقتاش = when · باهي = fine · نحب = I want"


def system_prompt(today: date) -> str:
    tomorrow = today + timedelta(days=1)
    facts = "\n".join(
        f"- {e.topic}: {e.answers['en']} | {e.answers['fr']} | {e.answers['ar']}" for e in FAQ
    )
    return f"""You are the phone receptionist of {CLINIC_NAME}, a clinic in Tunis.
Today is {today:%A} {today.isoformat()}; tomorrow is {tomorrow:%A} {tomorrow.isoformat()}.

Clinic facts (answer these directly, without a tool):
{facts}

Use a tool for anything about appointments; never invent free times or bookings.
- Free times with a doctor -> find_slots. Specialties: general, dentist, pediatrics, dermatology.
- Booking -> book_appointment as soon as you know specialty, date, time, full name and phone;
  ask only for what is missing.
- Checking a booking number -> get_appointment. Cancelling one -> cancel_appointment.
- If the request has nothing to do with the clinic, say you can only help with appointments
  and clinic information.

Tunisian words: {_DERJA}.

Your replies are spoken by a text-to-speech voice: one or two short sentences, no lists,
no markdown, no emojis. Reply in the language you are told the caller is using."""


def language_note(language: Language) -> str:
    return f"The caller is speaking {LANGUAGE_NAMES[language]}. Reply only in that language."


def few_shot_messages(today: date) -> list[Message]:
    """Worked tool-use exchanges, fenced off from the live call. Small models follow
    demonstrated calls far more reliably than rules. Phrasings, specialties and booking
    numbers deliberately differ from the eval scenarios."""
    tomorrow = (today + timedelta(days=1)).isoformat()

    def exchange(
        user: str,
        lang: Language,
        name: str,
        args: dict[str, str],
        result: dict[str, object],
        reply: str,
    ) -> list[Message]:
        call = ToolCall(f"example_{name}", name, args)
        return [
            Message("user", f"{user}\n\n({language_note(lang)})"),
            Message("assistant", "", tool_calls=[call]),
            Message(
                "tool", json.dumps(result, ensure_ascii=False), tool_call_id=call.id, name=name
            ),
            Message("assistant", reply),
        ]

    return [
        Message("system", "Example exchanges with earlier callers, showing how to use the tools:"),
        *exchange(
            "Vous avez un dermatologue de libre demain matin ?",
            "fr",
            "find_slots",
            {"specialty": "dermatology", "date": tomorrow, "part_of_day": "morning"},
            {"specialty": "dermatology", "date": tomorrow, "available_times": ["09:00", "09:30"]},
            "Oui, demain à neuf heures ou neuf heures et demie. Laquelle vous convient ?",
        ),
        *exchange(
            "I want to cancel appointment 2041",
            "en",
            "cancel_appointment",
            {"booking_ref": "2041"},
            {"error": "No booking found with reference 2041."},
            "I couldn't find booking 2041. Could you repeat the number?",
        ),
        *exchange(
            "شنوة تفاصيل الحجز ٣٠٥٧؟",
            "ar",
            "get_appointment",
            {"booking_ref": "3057"},
            {"error": "No booking found with reference 3057."},
            "لم أجد حجزًا بهذا الرقم. هل يمكنك تكراره؟",
        ),
        Message("system", "End of examples. A new caller is now on the line."),
    ]
