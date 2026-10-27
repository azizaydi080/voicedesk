import pytest

from voicedesk.language import detect_language


@pytest.mark.parametrize(
    ("text", "expected", "method"),
    [
        ("أريد موعدا عند طبيب الأسنان غدا", "ar", "script"),
        ("نحب ناخذ رنديفو غدوة", "ar", "script"),
        ("Je voudrais prendre un rendez-vous demain", "fr", "stopwords"),
        ("I would like to book an appointment tomorrow", "en", "stopwords"),
        ("nheb rendez-vous ghodwa 3and tbib", "ar", "arabizi"),
        ("Est-ce que vous êtes ouverts le samedi ?", "fr", "stopwords"),
    ],
)
def test_text_signals(text: str, expected: str, method: str) -> None:
    guess = detect_language(text)
    assert (guess.language, guess.method) == (expected, method)


def test_short_utterance_falls_back_to_stt_then_sticky() -> None:
    assert detect_language("OK", stt_language="fr", stt_probability=0.9).language == "fr"
    assert (
        detect_language("OK", stt_language="mt", stt_probability=0.9, previous="ar").language
        == "ar"
    )
    assert detect_language("OK", previous="en").method == "sticky"


def test_unsupported_stt_language_is_ignored() -> None:
    guess = detect_language("Salam", stt_language="fa", stt_probability=0.95)
    assert guess.language in {"ar", "fr", "en"}
