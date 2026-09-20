"""Explicit period and version markers, for telling a cross-document update from a contradiction.

Used only by the multi-document mode (multidoc.py). A figure in a Q2 report that differs from the Q1
report is usually an update, not a contradiction; NLI cannot know that. This module finds EXPLICIT
markers only -- it does not infer periods from context:
  quarters   "Q2 2026", "second quarter of 2026", "2026 Q2", and "first quarter" / "Q1" without a year
  fiscal     "fiscal 2026", "fiscal year 2026", "FY2026", "FY 2026"
  months     "March 2026", "March 31, 2026"
  versions   "version 2", "v2.1", "Amendment No. 3"
Keys are comparable tuples: ("Q", 2026, 2), ("FY", 2026, None), ("M", 2026, 3), ("V", "2.1", None).
A marker without a year takes the year of the document it appears in.
"""

from __future__ import annotations

import re
from collections import Counter

_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                       "september", "october", "november", "december"], 1)}
_PATTERNS = [
    (re.compile(r"\bQ([1-4])\s*(?:of\s+)?(?:FY\s*)?((?:19|20)\d\d)?\b", re.I), "Q_short"),
    (re.compile(r"\b((?:19|20)\d\d)\s*Q([1-4])\b", re.I), "Q_year_first"),
    (re.compile(r"\b(first|second|third|fourth)[\s-]+quarter(?:\s+(?:of\s+)?(?:fiscal\s+)?((?:19|20)\d\d))?", re.I), "Q_words"),
    (re.compile(r"\b(?:fiscal(?:\s+year)?|FY)\s*((?:19|20)\d\d)\b", re.I), "FY"),
    (re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+(?:\d{1,2},\s+)?((?:19|20)\d\d)\b", re.I), "M"),
    (re.compile(r"\b(?:version|v)\s*(\d+(?:\.\d+)*)\b", re.I), "V"),
    (re.compile(r"\bAmendment\s+No\.?\s*(\d+)\b", re.I), "V"),
]


def markers(text: str, default_year: int | None = None) -> list[tuple]:
    """All explicit period/version markers in ``text``, in order of appearance."""
    found: list[tuple[int, tuple]] = []
    for pattern, kind in _PATTERNS:
        for m in pattern.finditer(text):
            if kind == "Q_short":
                year = int(m.group(2)) if m.group(2) else default_year
                key = ("Q", year, int(m.group(1)))
            elif kind == "Q_year_first":
                key = ("Q", int(m.group(1)), int(m.group(2)))
            elif kind == "Q_words":
                year = int(m.group(2)) if m.group(2) else default_year
                key = ("Q", year, _ORD[m.group(1).lower()])
            elif kind == "FY":
                key = ("FY", int(m.group(1)), None)
            elif kind == "M":
                key = ("M", int(m.group(2)), _MONTHS[m.group(1).lower()])
            else:
                key = ("V", m.group(1), None)
            found.append((m.start(), key))
    return [key for _, key in sorted(found, key=lambda t: t[0])]


def document_period(text: str) -> tuple | None:
    """The document's own period: the most frequent fully specified marker (ties: the earliest).

    Quarter and fiscal-year markers are preferred over months and versions, because a report's reporting
    period is usually stated that way; a year-less marker cannot define a document's period.
    """
    keys = [k for k in markers(text) if k[1] is not None]
    if not keys:
        return None
    for kinds in (("Q",), ("FY",), ("M",), ("V",)):
        pool = [k for k in keys if k[0] in kinds]
        if pool:
            counts = Counter(pool)
            best = max(counts.values())
            return next(k for k in pool if counts[k] == best)
    return None


def describe(key: tuple | None) -> str:
    if key is None:
        return "no period marker"
    kind, a, b = key
    if kind == "Q":
        return f"Q{b} {a}" if a else f"Q{b}"
    if kind == "FY":
        return f"FY{a}"
    if kind == "M":
        return f"{a}-{b:02d}"
    return f"version {a}"


def claim_period(claim: str, doc_period: tuple | None) -> tuple | None:
    """The period a claim is about: its own explicit marker (year from its document), else its document's."""
    year = doc_period[1] if doc_period and isinstance(doc_period[1], int) else None
    own = markers(claim, default_year=year)
    return own[0] if own else doc_period
