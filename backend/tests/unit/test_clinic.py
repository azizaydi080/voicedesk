import pytest

from tests.conftest import TODAY
from voicedesk.agent.prompts import few_shot_messages, system_prompt
from voicedesk.domain.clinic import (
    ClinicStore,
    ToolError,
    normalize_ref,
    normalize_specialty,
    parse_time,
)
from voicedesk.domain.faq import FAQ


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dentist", "dentist"),
        ("Dentiste", "dentist"),
        ("طبيب الأسنان", "dentist"),
        ("pédiatre", "pediatrics"),
        ("الجلدية", "dermatology"),
        ("médecin généraliste", "general"),
    ],
)
def test_specialty_synonyms(raw: str, expected: str) -> None:
    assert normalize_specialty(raw) == expected


def test_refs_accept_spoken_and_arabic_indic_forms() -> None:
    assert normalize_ref("ENR-1002") == normalize_ref("١٠٠٢") == normalize_ref(1002) == "1002"
    with pytest.raises(ToolError):
        normalize_ref("none")


@pytest.mark.parametrize(
    ("raw", "hh_mm"),
    [("10:30", "10:30"), ("10h30", "10:30"), ("2", "14:00"), ("3 pm", "15:00"), ("١١:٠٠", "11:00")],
)
def test_parse_time(raw: str, hh_mm: str) -> None:
    assert parse_time(raw).strftime("%H:%M") == hh_mm


def test_slots_are_deterministic_and_bookable() -> None:
    a, b = ClinicStore(TODAY), ClinicStore(TODAY)
    slots = a.find_slots("dentist", "tomorrow")
    assert slots == b.find_slots("dentist", "tomorrow")
    first = slots["available_times"][0]
    booked = a.book("dentist", "tomorrow", first, "Sara Haddad", "+971 55 512 3456")
    assert booked["status"] == "confirmed"
    assert first not in a.find_slots("dentist", "tomorrow")["available_times"]
    with pytest.raises(ToolError, match="not available"):
        a.book("dentist", "tomorrow", first, "Someone Else", "0612345678")


def test_closed_on_sunday_and_past_dates() -> None:
    store = ClinicStore(TODAY)
    sunday = store.find_slots("general", "2026-11-01")
    assert sunday["available_times"] == [] and sunday["note"]
    with pytest.raises(ToolError, match="past"):
        store.find_slots("general", "2026-10-01")


def test_cancel_and_lookup() -> None:
    store = ClinicStore(TODAY)
    assert store.get("1001")["patient_name"] == "Amina Ben Salah"
    assert store.cancel("ENR 1001")["status"] == "cancelled"
    with pytest.raises(ToolError):
        store.get("9999")


def test_system_prompt_carries_every_fact_in_three_languages() -> None:
    prompt = system_prompt(TODAY)
    for entry in FAQ:
        for lang in ("en", "fr", "ar"):
            assert entry.answers[lang] in prompt  # type: ignore[index]
    assert "2026-10-28" in prompt  # tomorrow is spelled out for date resolution


def test_few_shot_examples_do_not_leak_eval_booking_refs() -> None:
    text = " ".join(m.content + str(m.tool_calls) for m in few_shot_messages(TODAY))
    assert not any(ref in text for ref in ("1001", "1002", "1003"))
