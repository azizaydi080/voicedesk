"""Language identification for short, spoken utterances.

Whisper's own language ID is unreliable on 1-3 second clips and on Maghrebi Arabic,
which it often labels as French or Maltese. We combine three signals:

1. the script (Arabic script is unambiguous),
2. a stop-word vote between French and English,
3. the STT provider's guess, used only when the text itself is inconclusive.

Short acknowledgements ("ok", "merci", "yes") keep the conversation's current language.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from voicedesk.providers.base import SUPPORTED_LANGUAGES, Language

_FR_WORDS = frozenset(
    """le la les un une des du de et est je tu il elle nous vous ils pour pas que qui
quoi avec mon ma mes votre vos rendez-vous rendez vous bonjour merci oui non
demain aujourd'hui est-ce c'est j'ai voudrais veux prendre médecin dentiste
heure heures quelle quel combien annuler réservation au aux sur dans peut
pouvez s'il plaît""".split()  # noqa: SIM905
)
_EN_WORDS = frozenset(
    """the a an and is i you he she we they to of for not that what with my your me
appointment hello hi thanks thank yes no tomorrow today can could would like
book doctor dentist time when how much cancel booking at on in please do does
open""".split()  # noqa: SIM905
)
# Tunisian / Maghrebi Arabic written in Latin script with digits ("Arabizi").
_ARABIZI_WORDS = frozenset(
    """chnowa chniya nheb n7eb nhebb barcha 3andi 3andek famma fama mta3 ya3tik sahit
saha inchallah inshallah wa9t waqt ghodwa ghudwa lyoum tbib""".split()  # noqa: SIM905
)
_FR_ELISIONS = frozenset({"j", "c", "l", "d", "s", "qu", "n", "m"})
_ARABIZI_TOKEN = re.compile(r"^(?=.*[a-z])(?=.*[235679])[a-z0-9']+$")
_WORD = re.compile(r"[\w'-]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class LanguageGuess:
    language: Language
    confidence: float
    method: str


def _arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if "ARABIC" in unicodedata.name(c, ""))
    return arabic / len(letters)


def detect_language(
    text: str,
    stt_language: str | None = None,
    stt_probability: float | None = None,
    previous: Language | None = None,
) -> LanguageGuess:
    ratio = _arabic_ratio(text)
    if ratio >= 0.5:
        return LanguageGuess("ar", min(1.0, 0.6 + ratio * 0.4), "script")

    tokens = [t.lower() for t in _WORD.findall(text)]
    arabizi = sum(1 for t in tokens if t in _ARABIZI_WORDS or _ARABIZI_TOKEN.match(t))
    fr = sum(1 for t in tokens if t in _FR_WORDS or ("'" in t and t.split("'")[0] in _FR_ELISIONS))
    en = sum(1 for t in tokens if t in _EN_WORDS)
    accents = sum(1 for c in text.lower() if c in "éèêàâçùûôîœ")
    fr += min(accents, 2)

    if arabizi >= 2 and arabizi >= max(fr, en):
        return LanguageGuess("ar", 0.7, "arabizi")

    total = fr + en
    if total >= 2 and fr != en:
        winner: Language = "fr" if fr > en else "en"
        margin = abs(fr - en) / total
        return LanguageGuess(winner, round(0.5 + 0.5 * margin, 3), "stopwords")

    if stt_language in SUPPORTED_LANGUAGES and (stt_probability or 0.0) >= 0.5:
        lang: Language = stt_language
        return LanguageGuess(lang, float(stt_probability or 0.5), "stt")

    if previous is not None:
        return LanguageGuess(previous, 0.5, "sticky")

    if total and fr != en:
        return LanguageGuess("fr" if fr > en else "en", 0.5, "stopwords-weak")
    if stt_language in SUPPORTED_LANGUAGES:
        return LanguageGuess(stt_language, float(stt_probability or 0.3), "stt-weak")
    return LanguageGuess("en", 0.2, "default")
