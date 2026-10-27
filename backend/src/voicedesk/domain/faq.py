"""The clinic's static facts, in all three languages.

They are small enough to live in the (cached) system prompt, so FAQ answers need no tool
round trip.
"""

from __future__ import annotations

from dataclasses import dataclass

from voicedesk.providers.base import Language


@dataclass(frozen=True, slots=True)
class FaqEntry:
    topic: str
    answers: dict[Language, str]


FAQ: tuple[FaqEntry, ...] = (
    FaqEntry(
        "hours",
        {
            "en": "Open Monday to Saturday, 9 am to 5 pm. Closed on Sundays.",
            "fr": "Ouvert du lundi au samedi, de 9 h à 17 h. Fermé le dimanche.",
            "ar": "مفتوحون من الاثنين إلى السبت، من التاسعة صباحًا حتى الخامسة مساءً. مغلق يوم الأحد.",
        },
    ),
    FaqEntry(
        "address",
        {
            "en": "We are at 12 Avenue Habib Bourguiba, Tunis, next to the central pharmacy.",
            "fr": "Nous sommes au 12 avenue Habib Bourguiba, à Tunis, à côté de la pharmacie centrale.",
            "ar": "عنواننا ١٢ شارع الحبيب بورقيبة، تونس، بجانب الصيدلية المركزية.",
        },
    ),
    FaqEntry(
        "prices",
        {
            "en": "A general consultation costs 60 dinars; specialists cost 90 dinars.",
            "fr": "Une consultation générale coûte 60 dinars ; un spécialiste 90 dinars.",
            "ar": "الكشف العام بستين دينارًا، وعند الأخصائي بتسعين دينارًا.",
        },
    ),
    FaqEntry(
        "insurance",
        {
            "en": "We accept CNAM and most private insurers; bring your insurance card.",
            "fr": "Nous acceptons la CNAM et la plupart des mutuelles ; apportez votre carte.",
            "ar": "نقبل الكنام وأغلب شركات التأمين الخاصة؛ أحضر بطاقة التأمين.",
        },
    ),
    FaqEntry(
        "documents",
        {
            "en": "Please bring an ID, your insurance card and any previous test results.",
            "fr": "Apportez une pièce d'identité, votre carte d'assurance et vos anciens résultats.",
            "ar": "أحضر بطاقة الهوية وبطاقة التأمين وأي نتائج تحاليل سابقة.",
        },
    ),
    FaqEntry(
        "parking",
        {
            "en": "Free parking is available behind the building.",
            "fr": "Un parking gratuit est disponible derrière le bâtiment.",
            "ar": "يوجد موقف سيارات مجاني خلف المبنى.",
        },
    ),
)
