"""Sentence splitting and small text helpers."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk

_BULLET = re.compile(r"^\s*(?:[-*•]|\(?[a-z0-9]{1,2}\))\s+", re.IGNORECASE)
_WS = re.compile(r"\s+")
_COMPLETE_END = re.compile(r"[.!?][\"'”)\]]*$")


@lru_cache(maxsize=1)
def ensure_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


@lru_cache(maxsize=1)
def ensure_tagger() -> None:
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; each bullet line starts its own paragraph."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current))
                current = []
        elif _BULLET.match(line):
            if current:
                paragraphs.append(" ".join(current))
            current = [_BULLET.sub("", line, count=1)]
        else:
            current.append(line.strip())
    if current:
        paragraphs.append(" ".join(current))
    return [p for p in (normalize_ws(p) for p in paragraphs) if p]


def split_sentences(text: str) -> list[str]:
    ensure_nltk()
    sentences: list[str] = []
    for paragraph in split_paragraphs(text):
        sentences.extend(s.strip() for s in nltk.sent_tokenize(paragraph) if s.strip())
    return sentences


def is_complete_sentence(sentence: str) -> bool:
    return bool(_COMPLETE_END.search(sentence.strip()))


def drop_incomplete_tail(sentences: list[str]) -> tuple[list[str], list[str]]:
    """Drop a final sentence cut off by the generation length limit (no terminal punctuation).

    The only sentence is never dropped, so a summary cannot become empty this way.
    """
    if len(sentences) > 1 and not is_complete_sentence(sentences[-1]):
        return sentences[:-1], sentences[-1:]
    return sentences, []


# Small built-in stopword list (avoids an extra nltk corpus download).
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our
ours out over own same she should so some such than that the their them then there these they this those through
to too under until up very was we were what when where which while who whom why will with would you your yours
thank thanks please okay ok yes hello hi welcome good afternoon morning evening everyone
""".split())
_SPEAKER_LABEL = re.compile(r"^\s*[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,4}(?:,\s*[^:]{2,60})?:\s+(?=\S)")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'%$.-]*")


def strip_speaker_label(sentence: str) -> str:
    """'Jane Doe, Chief Financial Officer: Revenue grew.' -> 'Revenue grew.' (transcript speaker labels)."""
    return _SPEAKER_LABEL.sub("", sentence, count=1)


# Reporting / evidential frames that wrap a claim without changing WHAT is claimed. As the NLI
# hypothesis, "We also note that X" is read as a claim about what the authors note, which the premise
# does not address, so P(contradiction) collapses (0.005-0.63 on four measured pairs, FINDINGS 12.1).
# Deliberately EXCLUDED: belief and expectation hedges ("we believe", "we expect", "results suggest",
# "we hypothesize") -- stripping those would turn a stated belief into an assertion -- and discourse
# adverbs ("Moreover,"), which were not measured to matter.
_REPORTING_FRAME = re.compile(
    r"^(?:"
    r"(?:we|i)\s+(?:(?:also|further|additionally|again|now|then|first|finally|consistently)\s+)?"
    r"(?:note|noted|observe|observed|find|found|show|showed|demonstrate|demonstrated|see|saw|report|reported|"
    r"confirm|confirmed|verify|verified|notice|noticed)\s+that\s+"
    r"|it\s+(?:is|was)\s+(?:also\s+)?(?:worth\s+noting|important\s+to\s+note|interesting\s+to\s+note|"
    r"notable|noteworthy)\s+that\s+"
    r"|it\s+should\s+(?:also\s+)?be\s+noted\s+that\s+"
    r"|(?:note|notice)\s+(?:also\s+)?that\s+"
    r"|(?:our|the|these)\s+(?:results|findings|experiments|analyses|analysis)\s+(?:also\s+)?"
    r"(?:show|shows|showed|demonstrate|demonstrates|demonstrated|confirm|confirms|confirmed)\s+that\s+"
    r")",
    re.IGNORECASE)


def strip_reporting_frame(sentence: str) -> str:
    """'We also note that the dataset spans 24 months.' -> 'The dataset spans 24 months.'"""
    stripped = sentence
    for _ in range(2):                      # "We also note that, ... we find that X": at most two frames
        m = _REPORTING_FRAME.match(stripped.lstrip())
        if not m:
            break
        rest = stripped.lstrip()[m.end():].lstrip(", ")
        stripped = rest[:1].upper() + rest[1:]
    return stripped if stripped.strip() else sentence


def content_word_count(sentence: str) -> int:
    return sum(1 for w in _WORD.findall(sentence) if w.lower().strip(".'") not in STOPWORDS)


@lru_cache(maxsize=65536)
def has_verb_or_number(sentence: str) -> bool:
    """True if the sentence has a verb (incl. modals) or a number (digits or spelled out).

    A fragment with neither asserts nothing that could be contradicted: "Jeanna Smialek, New York
    Times." or "Elizabeth Schulze with ABC News." (reporter introductions in a press-conference
    transcript, which scored P(contradiction) ~1.00 against almost anything). Headline-style facts
    without a verb keep their number and so survive: "Revenue up 12% year over year."
    Uses nltk's averaged-perceptron POS tagger; a mis-tag errs in either direction. The tagger
    labels some spelled-out numbers as adjectives ("roughly sixty enterprise accounts": sixty/JJ),
    so number words are also matched directly.
    """
    ensure_nltk()
    ensure_tagger()
    letters = [c for c in sentence if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        sentence = sentence.lower()   # ALL-CAPS legal text: the tagger reads SHALL/BE as proper nouns
    # Curly apostrophes stop the tokenizer splitting contractions ("isn’t" stays one unknown token).
    words = nltk.word_tokenize(sentence.replace("’", "'"))
    if any(c.isdigit() for c in sentence) or any(w.lower().split("-")[0] in _NUMBER_WORDS for w in words):
        return True
    return any(tag.startswith("VB") or tag in ("MD", "CD") for _, tag in nltk.pos_tag(words))


_NUMBER_WORDS = frozenset("""
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
million billion trillion dozen half percent
""".split())


def _content_token(word: str) -> str | None:
    lowered = word.lower().strip(".'%$-\u2019").replace(",", "")
    if lowered.endswith(("'s", "\u2019s")):
        lowered = lowered[:-2]
    if lowered in STOPWORDS:
        return None
    if any(c.isdigit() for c in lowered):
        return lowered
    if len(lowered) < 3:
        return None
    if len(lowered) > 4 and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > 3 and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def content_tokens(sentence: str) -> set[str]:
    """Lower-cased non-stopword tokens, crudely de-pluralised ("rates" -> "rate"), for topic overlap.

    Unlike ``entity_tokens`` this keeps ordinary nouns ("labor market", "mortgage rates"), which is
    what most contradictions in minutes, transcripts and contracts are about (FINDINGS 10.2).
    """
    out = set()
    for word in _WORD.findall(strip_speaker_label(sentence)):
        token = _content_token(word)
        if token:
            out.add(token)
    return out


# Hyphenated compounds stay ONE token on purpose: "twenty-24" (a real DistilBART corruption of
# "twenty-four (24)") must not tokenize into "twenty" + "24", which both occur in the source.
_SURFACE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9%$'\u2019.-]*")


def surface_tokens(text: str) -> list[str]:
    return [t.lower().strip(".'\u2019") for t in _SURFACE_TOKEN.findall(text)]


def novel_tokens(sentence: str, source_vocabulary: set[str], min_length: int = 5) -> list[str]:
    """Tokens of ``sentence`` that occur nowhere in the source.

    Kept deliberately simple: a token is reported when it contains a digit (fabricated or corrupted
    numbers) or is at least ``min_length`` characters (corrupted/invented words). Short function
    words are ignored, and legitimate paraphrase will also show up here -- these are flags for a
    human, not errors.
    """
    out: list[str] = []
    for token in surface_tokens(sentence):
        if token in source_vocabulary or token in STOPWORDS:
            continue
        if any(c.isdigit() for c in token) or len(token) >= min_length:
            if token not in out:
                out.append(token)
    return out


# Entity-like tokens, without a NER model: capitalised words away from the sentence start,
# acronyms, and anything containing a digit. This is a cheap proxy for "these two sentences talk
# about the same thing", used as an optional Stage 3 pre-filter.
_ENTITYISH = re.compile(r"[A-Za-z][\w-]*|\d[\w.,%$-]*")


def entity_tokens(sentence: str) -> set[str]:
    """Approximate the named entities / figures in a sentence.

    Deliberately crude, and its limits matter: a sentence written entirely in lower case with
    spelled-out numbers ("roughly sixty enterprise accounts are still served...") yields NOTHING,
    so an entity filter would treat it as related to nothing at all.
    """
    words = _ENTITYISH.findall(strip_speaker_label(sentence))
    out: set[str] = set()
    for position, word in enumerate(words):
        if any(ch.isdigit() for ch in word):
            out.add(word.lower().strip(".,"))
        elif word.isupper() and len(word) >= 2:
            out.add(word.lower())
        elif position > 0 and word[:1].isupper() and word.lower() not in STOPWORDS:
            out.add(word.lower())
    return out


# Phrases that frame a sentence as a COMPARISON with something else rather than an assertion about
# the sentence's own subject. NLI treats "system A does X" vs "the baseline does Y instead" as a
# contradiction, because its training data assumes both sentences describe one situation.
_COMPARATIVE_MARKERS = (
    "baseline", "unlike", "in contrast", "by contrast", "compared with", "compared to",
    "whereas", "versus", " vs ", "relative to", "instead of", "rather than", "as opposed to",
    "outperform", "better than", "worse than", "faster than", "slower than", "higher than",
    "lower than", "more than", "less than",
)


def comparative_framing(sentence: str) -> str | None:
    """Return the comparison phrase that frames this sentence, or None."""
    lowered = f" {sentence.lower()} "
    for marker in _COMPARATIVE_MARKERS:
        if marker in lowered:
            return marker.strip()
    return None
