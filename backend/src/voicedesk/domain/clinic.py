"""A small, deterministic mock business: appointment booking for a multilingual clinic.

Designed for voice: callers say times and booking numbers out loud, so tools take
``specialty + date + time`` instead of opaque slot ids, and booking references are short
numbers that survive Arabic-Indic digits ("١٠٠٢") or a spoken prefix ("ENR 1002").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

CLINIC_NAME = "Ennour Clinic"

SPECIALTIES: dict[str, tuple[str, ...]] = {
    "general": (
        "general",
        "gp",
        "généraliste",
        "generaliste",
        "médecin généraliste",
        "doctor",
        "طبيب عام",
        "عام",
        "طب عام",
        "family",
    ),
    "dentist": (
        "dentist",
        "dental",
        "dentiste",
        "dentaire",
        "أسنان",
        "اسنان",
        "طبيب أسنان",
        "طبيب الأسنان",
        "طب الأسنان",
        "teeth",
    ),
    "pediatrics": (
        "pediatrics",
        "pediatrician",
        "paediatrics",
        "pédiatre",
        "pediatre",
        "pédiatrie",
        "أطفال",
        "اطفال",
        "طبيب أطفال",
        "طب الأطفال",
        "child",
        "children",
    ),
    "dermatology": (
        "dermatology",
        "dermatologist",
        "dermatologue",
        "dermatologie",
        "skin",
        "peau",
        "جلدية",
        "الجلدية",
        "جلد",
        "طبيب جلدية",
        "الأمراض الجلدية",
    ),
}

_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_SLOT_MINUTES = 30
_DAY_START, _DAY_END = time(9, 0), time(17, 0)


class ToolError(ValueError):
    """Raised for invalid tool input; the message is returned to the model verbatim."""


def normalize_specialty(value: str) -> str:
    v = value.strip().lower().translate(_ARABIC_INDIC)
    for key, synonyms in SPECIALTIES.items():
        if v == key or v in synonyms:
            return key
    for key, synonyms in SPECIALTIES.items():
        if any(s in v for s in synonyms if len(s) > 3):
            return key
    raise ToolError(f"Unknown specialty '{value}'. Valid: {', '.join(SPECIALTIES)}.")


def normalize_ref(value: str | int) -> str:
    digits = re.sub(r"\D", "", str(value).translate(_ARABIC_INDIC))
    if not digits:
        raise ToolError(f"Booking reference '{value}' has no digits.")
    return digits


def parse_date(value: str, today: date) -> date:
    v = value.strip().lower().translate(_ARABIC_INDIC)
    relative = {
        "today": 0,
        "aujourd'hui": 0,
        "اليوم": 0,
        "lyoum": 0,
        "tomorrow": 1,
        "demain": 1,
        "غدا": 1,
        "غدًا": 1,
        "غدوة": 1,
        "بكرة": 1,
        "bukra": 1,
    }
    if v in relative:
        return today + timedelta(days=relative[v])
    try:
        return date.fromisoformat(v[:10])
    except ValueError as exc:
        raise ToolError(f"Date '{value}' must be YYYY-MM-DD, 'today' or 'tomorrow'.") from exc


def parse_time(value: str) -> time:
    v = value.strip().lower().translate(_ARABIC_INDIC).replace("h", ":").replace(".", ":")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", v)
    if not m:
        raise ToolError(f"Time '{value}' must look like HH:MM.")
    hour, minute = int(m.group(1)), int(m.group(2) or 0)
    if (m.group(3) == "pm" and hour < 12) or hour < 8:
        hour += 12
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ToolError(f"Time '{value}' is not a valid time.")
    return time(hour, minute)


@dataclass(slots=True)
class Booking:
    ref: str
    specialty: str
    start: datetime
    patient_name: str
    phone: str
    status: str = "confirmed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "booking_ref": self.ref,
            "specialty": self.specialty,
            "date": self.start.date().isoformat(),
            "time": self.start.strftime("%H:%M"),
            "patient_name": self.patient_name,
            "status": self.status,
        }


@dataclass
class ClinicStore:
    """In-memory store, seeded deterministically from ``today`` so evals are reproducible."""

    today: date
    bookings: dict[str, Booking] = field(default_factory=dict)
    _next_ref: int = 1004

    def __post_init__(self) -> None:
        tomorrow = self.today + timedelta(days=1)
        seed = [
            ("1001", "dermatology", tomorrow, time(10, 0), "Amina Ben Salah", "+21620111222"),
            ("1002", "dentist", tomorrow, time(11, 30), "Jean Dupont", "+33612345678"),
            (
                "1003",
                "general",
                self.today + timedelta(days=2),
                time(15, 0),
                "John Smith",
                "+971501234567",
            ),
        ]
        for ref, spec, d, t, name, phone in seed:
            self.bookings[ref] = Booking(ref, spec, datetime.combine(d, t), name, phone)

    # -- availability ---------------------------------------------------------

    def _is_open(self, specialty: str, start: datetime) -> bool:
        if start.weekday() == 6:  # closed on Sundays
            return False
        if not (_DAY_START <= start.time() < _DAY_END) or start.minute % _SLOT_MINUTES:
            return False
        # Deterministic "busy" pattern: about a third of slots are taken by other patients.
        digest = hashlib.sha256(f"{specialty}{start.isoformat()}".encode()).digest()
        if digest[0] % 3 == 0:
            return False
        return not any(
            b.status == "confirmed" and b.specialty == specialty and b.start == start
            for b in self.bookings.values()
        )

    def find_slots(
        self, specialty: str, day: str, part_of_day: str | None = None
    ) -> dict[str, Any]:
        spec = normalize_specialty(specialty)
        d = parse_date(day, self.today)
        if d < self.today:
            raise ToolError("That date is in the past.")
        lo, hi = _DAY_START, _DAY_END
        if part_of_day in {"morning", "matin", "صباح", "الصباح"}:
            hi = time(12, 0)
        elif part_of_day in {"afternoon", "après-midi", "apres-midi", "مساء", "بعد الظهر"}:
            lo = time(12, 0)
        slots: list[str] = []
        cursor = datetime.combine(d, lo)
        while cursor.time() < hi and len(slots) < 4:
            if self._is_open(spec, cursor):
                slots.append(cursor.strftime("%H:%M"))
            cursor += timedelta(minutes=_SLOT_MINUTES)
        return {
            "specialty": spec,
            "date": d.isoformat(),
            "weekday": d.strftime("%A"),
            "available_times": slots,
            "note": "Clinic is closed on Sundays." if d.weekday() == 6 else None,
        }

    # -- bookings -------------------------------------------------------------

    def book(
        self, specialty: str, day: str, at: str, patient_name: str, phone: str
    ) -> dict[str, Any]:
        spec = normalize_specialty(specialty)
        start = datetime.combine(parse_date(day, self.today), parse_time(at))
        if not patient_name.strip():
            raise ToolError("patient_name is required.")
        if len(re.sub(r"\D", "", phone.translate(_ARABIC_INDIC))) < 6:
            raise ToolError("A phone number with at least 6 digits is required.")
        if not self._is_open(spec, start):
            alt = self.find_slots(spec, start.date().isoformat())
            raise ToolError(
                f"{start:%H:%M} on {start:%Y-%m-%d} is not available for {spec}. "
                f"Free times that day: {', '.join(alt['available_times']) or 'none'}."
            )
        ref = str(self._next_ref)
        self._next_ref += 1
        booking = Booking(ref, spec, start, patient_name.strip(), phone.strip())
        self.bookings[ref] = booking
        return booking.as_dict()

    def get(self, booking_ref: str) -> dict[str, Any]:
        ref = normalize_ref(booking_ref)
        if ref not in self.bookings:
            raise ToolError(f"No booking found with reference {ref}.")
        return self.bookings[ref].as_dict()

    def cancel(self, booking_ref: str) -> dict[str, Any]:
        ref = normalize_ref(booking_ref)
        if ref not in self.bookings:
            raise ToolError(f"No booking found with reference {ref}.")
        self.bookings[ref].status = "cancelled"
        return self.bookings[ref].as_dict()
