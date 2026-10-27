"""Tiny trilingual FAQ with keyword retrieval.

Retrieval-augmented generation is not the point of this project (a sibling repo covers
it), so this is a deliberately small lexical retriever. It returns the answer in the
caller's language so the model only has to rephrase, not translate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from voicedesk.providers.base import Language


@dataclass(frozen=True, slots=True)
class FaqEntry:
    topic: str
    keywords: frozenset[str]
    answers: dict[Language, str]


FAQ: tuple[FaqEntry, ...] = (
    FaqEntry(
        "hours",
        frozenset(
            """hours open opening close closing time horaires heures ouvert ouverture fermé
fermeture dimanche sunday saturday samedi مواعيد ساعات العمل مفتوح تفتح تغلق
الأحد الدوام""".split()  # noqa: SIM905
        ),
        {
            "en": "Open Monday to Saturday, 9 am to 5 pm. Closed on Sundays.",
            "fr": "Ouvert du lundi au samedi, de 9 h à 17 h. Fermé le dimanche.",
            "ar": "مفتوحون من الاثنين إلى السبت، من التاسعة صباحًا حتى الخامسة مساءً. مغلق يوم الأحد.",
        },
    ),
    FaqEntry(
        "address",
        frozenset(
            """address where location located directions adresse où situé trouve العنوان وين
أين موقع مكان""".split()  # noqa: SIM905
        ),
        {
            "en": "We are at 12 Avenue Habib Bourguiba, Tunis, next to the central pharmacy.",
            "fr": "Nous sommes au 12 avenue Habib Bourguiba, à Tunis, à côté de la pharmacie centrale.",
            "ar": "عنواننا ١٢ شارع الحبيب بورقيبة، تونس، بجانب الصيدلية المركزية.",
        },
    ),
    FaqEntry(
        "prices",
        frozenset(
            """price prices cost fee fees how much consultation prix tarif tarifs coûte
combien سعر الأسعار كم تكلفة الكشف ثمن""".split()  # noqa: SIM905
        ),
        {
            "en": "A general consultation costs 60 dinars; specialists cost 90 dinars.",
            "fr": "Une consultation générale coûte 60 dinars ; un spécialiste 90 dinars.",
            "ar": "الكشف العام بستين دينارًا، وعند الأخصائي بتسعين دينارًا.",
        },
    ),
    FaqEntry(
        "insurance",
        frozenset(
            """insurance cnam mutuelle assurance covered coverage remboursement reimburse
تأمين التأمين الضمان cnam تغطية""".split()  # noqa: SIM905
        ),
        {
            "en": "We accept CNAM and most private insurers; bring your insurance card.",
            "fr": "Nous acceptons la CNAM et la plupart des mutuelles ; apportez votre carte.",
            "ar": "نقبل الكنام وأغلب شركات التأمين الخاصة؛ أحضر بطاقة التأمين.",
        },
    ),
    FaqEntry(
        "documents",
        frozenset(
            """bring documents id card papers apporter documents pièce identité carte أحضر
وثائق بطاقة هوية أوراق""".split()  # noqa: SIM905
        ),
        {
            "en": "Please bring an ID, your insurance card and any previous test results.",
            "fr": "Apportez une pièce d'identité, votre carte d'assurance et vos anciens résultats.",
            "ar": "أحضر بطاقة الهوية وبطاقة التأمين وأي نتائج تحاليل سابقة.",
        },
    ),
    FaqEntry(
        "parking",
        frozenset(
            """parking park car garer voiture stationnement موقف سيارات سيارة ركن""".split()  # noqa: SIM905
        ),
        {
            "en": "Free parking is available behind the building.",
            "fr": "Un parking gratuit est disponible derrière le bâtiment.",
            "ar": "يوجد موقف سيارات مجاني خلف المبنى.",
        },
    ),
)

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    out = set()
    for tok in _TOKEN.findall(text.lower()):
        out.add(tok)
        if tok.startswith("ال") and len(tok) > 3:  # Arabic definite article
            out.add(tok[2:])
    return out


def search_faq(question: str, language: Language) -> dict[str, Any]:
    q = _tokens(question)
    scored = sorted(
        ((len(q & {k.lower() for k in e.keywords}) + (2 if e.topic in q else 0), e) for e in FAQ),
        key=lambda pair: pair[0],
        reverse=True,
    )
    best_score, best = scored[0]
    if best_score == 0:
        return {"found": False, "topics": [e.topic for e in FAQ]}
    return {"found": True, "topic": best.topic, "answer": best.answers[language]}
