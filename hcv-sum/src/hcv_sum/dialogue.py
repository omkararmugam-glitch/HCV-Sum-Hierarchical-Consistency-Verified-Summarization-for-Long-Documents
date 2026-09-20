"""Optional preprocessing: rewrite speaker-labelled dialogue as descriptive prose before segmentation.

Inspired by NexusSum's dialogue-to-description step (Kim & Kim, ACL 2025), which uses an LLM agent to
turn dialogue into narrative before summarising. This is NOT that method: it is a small set of rules,
chosen because it runs on CPU in milliseconds and every rewrite can be traced to a rule. It formalises
what the pipeline already did informally (speaker labels stripped before NLI, reporter introductions
filtered in Stage 3).

Rules, applied to each paragraph that starts with a speaker label ("Name[, Title]: utterance"):
1. The utterance is split into sentences. Sentences that assert nothing are dropped:
   (a) short ones -- fewer than four content words, or a short fragment with no verb and no number
       (the Stage 3 non-claim rule): "Thank you, operator.", "Please go ahead.";
   (b) non-substantive speech acts of ANY length, recognised by their form (``speech_act``): greetings
       and welcomes, thanks, closings, call procedure ("our first question comes from...", "listen-only
       mode", "turn the call over to..."), speaker introductions ("Joining me today are...") and
       safe-harbour boilerplate ("forward-looking statements", "reconciliation of non-GAAP measures").
       Rule (a) alone missed these when they were long: "Good afternoon, and welcome to the Helix Cloud
       Systems third quarter 2026 earnings conference call" has nine content words, so it was kept and
       rewritten as "The conference operator said that good afternoon, and welcome..." (FINDINGS 16).
   Dropped rather than kept in their original form: the rewrite already drops short pleasantries, and
   keeping these as "Operator: Good afternoon..." would put speaker labels and pleasantries back into
   the summaries and back into Stage 3 as claim candidates. Every dropped sentence is recorded, with its
   category, in the counts returned (and in the pipeline stats), so nothing is removed silently.
2. First-person pronouns become third person. Plural (we / us / our / ours) -> the configured
   organisation reference ("the company", "the company's"); singular (I / me / my) -> the speaker's name.
   A present-tense verb right after the new subject is put in the third person singular ("we plan" ->
   "the company plans", "I think" -> "Daniel Okafor thinks"; contractions "we're", "we've", "we'll",
   "I'm" are expanded first). Verbs are found with nltk's part-of-speech tagger.
3. The first kept STATEMENT of a turn is attributed: "Name, Title, said that <sentence>" (the operator
   becomes "The conference operator"). A question is never put into "said that" form: it is kept as
   "Name, Title, asked: <question>", with its wording unchanged (no pronoun rewriting, since an
   analyst's "we"/"our" is not the company's). Later sentences of the same turn are not re-attributed.
Everything else in the document is left untouched.

Known limits of the rules: second-person "you" is kept as is; a plural "we" that does not mean the
organisation (e.g. "we" = the analysts on the call) is still rewritten to it; the verb rule only handles
the verb directly after the pronoun; quoted speech inside an utterance is rewritten like the rest.
"""

from __future__ import annotations

import re

import nltk

from .text_utils import content_word_count, ensure_nltk, ensure_tagger, has_verb_or_number, split_sentences

_SPEAKER_TURN = re.compile(r"^(?P<name>[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,4})(?:,\s*(?P<title>[^:\n]{2,60}))?:\s+(?P<text>\S.*)$",
                           re.DOTALL)
_CONTRACTIONS = [
    (re.compile(r"\b([Ww]e)['’]re\b"), r"\1 are"), (re.compile(r"\b([Ww]e)['’]ve\b"), r"\1 have"),
    (re.compile(r"\b([Ww]e)['’]ll\b"), r"\1 will"), (re.compile(r"\b([Ww]e)['’]d\b"), r"\1 would"),
    (re.compile(r"\bI['’]m\b"), "I am"), (re.compile(r"\bI['’]ve\b"), "I have"),
    (re.compile(r"\bI['’]ll\b"), "I will"), (re.compile(r"\bI['’]d\b"), "I would"),
]
_IRREGULAR = {"are": "is", "am": "is", "have": "has", "do": "does", "go": "goes", "were": "was"}


def _third_person(verb: str) -> str:
    lower = verb.lower()
    if lower in _IRREGULAR:
        out = _IRREGULAR[lower]
    elif re.search(r"(s|sh|ch|x|z|o)$", lower):
        out = lower + "es"
    elif re.search(r"[^aeiou]y$", lower):
        out = lower[:-1] + "ies"
    else:
        out = lower + "s"
    return out[0].upper() + out[1:] if verb[:1].isupper() else out


def _is_pleasantry(sentence: str) -> bool:
    return content_word_count(sentence) < 4 or (len(sentence.split()) < 10 and not has_verb_or_number(sentence))


# Non-substantive speech acts, recognised by FORM, whatever their length. Anchored patterns ("^") only
# match at the start of the sentence, so "Thanks to strong demand, revenue grew" is not a thank-you.
_SPEECH_ACTS = [
    ("greeting", re.compile(r"^(?:good (?:morning|afternoon|evening|day)|hello|hi|hey|greetings|welcome(?: back)? to"
                            r"|ladies and gentlemen)\b", re.I)),
    ("thanks", re.compile(r"^(?:thank you|thanks|many thanks)\b(?!\s+to\b)", re.I)),
    ("closing", re.compile(r"^(?:that|this) concludes\b|^(?:goodbye|have a (?:good|great|nice))\b", re.I)),
    ("procedure", re.compile(r"\b(?:listen-only mode|question-and-answer session|(?:this|the) (?:call|conference) is being recorded"
                             r"|(?:turn|hand)(?:ing)? (?:the call|it|things)(?: back)? over to"
                             r"|(?:our|the) (?:first|next|last|final) question (?:comes|is) from"
                             r"|please (?:go ahead|stand by|hold)|(?:we|i) will now (?:begin|open|take|turn|move))\b", re.I)),
    ("introduction", re.compile(r"^(?:joining (?:me|us)|with (?:me|us) (?:today|on the call)|on the call with me)\b", re.I)),
    ("boilerplate", re.compile(r"\b(?:forward-looking statements?|safe harbou?r|reconciliation of non-gaap)\b", re.I)),
]


def speech_act(sentence: str) -> str | None:
    """The category of a non-substantive speech act (greeting, thanks, closing, procedure, introduction,
    boilerplate), or None if the sentence may assert something."""
    s = sentence.strip()
    for name, pattern in _SPEECH_ACTS:
        if pattern.search(s):
            return name
    return None


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().rstrip('"\u201d\'').endswith("?")


def rewrite_sentence(sentence: str, speaker: str, organisation: str) -> str:
    """First person -> third person, with the verb right after the new subject agreed."""
    for pattern, repl in _CONTRACTIONS:
        sentence = pattern.sub(repl, sentence)
    tokens = nltk.word_tokenize(sentence)
    tags = [t for _, t in nltk.pos_tag(tokens)]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok, low = tokens[i], tokens[i].lower()
        subject = None
        if low == "we":
            subject = organisation
        elif tok == "I":
            subject = speaker
        if subject is not None:
            out.append(subject)
            if i + 1 < len(tokens) and tags[i + 1] in ("VBP", "VB") and tokens[i + 1].lower() not in ("will", "would", "can", "could", "should", "may", "might", "must"):
                out.append(_third_person(tokens[i + 1]))
                i += 2
                continue
            if i + 1 < len(tokens) and tokens[i + 1].lower() == "were":
                out.append("was")
                i += 2
                continue
        elif low == "us":
            out.append(organisation)
        elif low in ("our", "ours"):
            out.append(f"{organisation}'s")
        elif low in ("me",) and tok == "me":
            out.append(speaker)
        elif low == "my":
            out.append(f"{speaker}'s")
        else:
            out.append(tok)
        i += 1
    text = " ".join(out)
    text = re.sub(r"\s+([.,;:!?%)\]])", r"\1", text)          # detokenize punctuation
    text = re.sub(r"([($\[])\s+", r"\1", text)
    text = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d)\b", r"\1", text)
    text = text.replace("`` ", '"').replace(" ''", '"')
    return text[:1].upper() + text[1:]


def _lower_first(sentence: str) -> str:
    """Lower-case the first word for 'X said that ...' unless it looks like a proper noun or acronym."""
    first = sentence.split(" ", 1)[0]
    if first.isupper() or (len(first) > 1 and any(c.isupper() for c in first[1:])):
        return sentence
    tag = nltk.pos_tag([first])[0][1] if first else ""
    if tag in ("NNP", "NNPS") and first.lower() not in ("the", "total", "overall"):
        return sentence
    return sentence[:1].lower() + sentence[1:]


def dialogue_to_description(text: str, organisation: str = "the company") -> tuple[str, dict]:
    """Rewrite speaker-labelled paragraphs as descriptive prose. Returns (text, counts)."""
    ensure_nltk()
    ensure_tagger()
    counts = {"turns": 0, "sentences_kept": 0, "sentences_dropped": 0, "dropped_by_category": {},
              "dropped_sentences": []}
    paragraphs = re.split(r"(\n\s*\n)", text)
    out = []
    for para in paragraphs:
        m = _SPEAKER_TURN.match(para.strip()) if para.strip() else None
        if not m:
            out.append(para)
            continue
        counts["turns"] += 1
        name, title = m.group("name").strip(), (m.group("title") or "").strip()
        speaker = "The conference operator" if name.lower() == "operator" else name
        kept: list[tuple[str, bool]] = []          # (sentence, is_question)
        for sentence in split_sentences(m.group("text")):
            category = speech_act(sentence) or ("pleasantry" if _is_pleasantry(sentence) else None)
            if category:
                counts["sentences_dropped"] += 1
                counts["dropped_by_category"][category] = counts["dropped_by_category"].get(category, 0) + 1
                counts["dropped_sentences"].append(f"[{category}] {name}: {sentence}")
                continue
            if _is_question(sentence):
                kept.append((sentence, True))
            else:
                kept.append((rewrite_sentence(sentence, speaker, organisation), False))
        counts["sentences_kept"] += len(kept)
        if not kept:
            out.append("")
            continue
        who = f"{speaker}, {title}," if title else speaker
        rendered, attributed = [], False
        for sentence, is_question in kept:
            if is_question:
                rendered.append(f"{who} asked: {sentence}")
            elif not attributed:
                rendered.append(f"{who} said that {_lower_first(sentence)}")
                attributed = True
            else:
                rendered.append(sentence)
        out.append(" ".join(rendered))
    return "".join(out), counts


_ATTRIBUTION_FRAME = re.compile(r"^(?:[A-Z][\w.'-]*(?:\s+[\w.'-]+){0,5}?)(?:,\s*[^,]{2,60},)?\s+said\s+that\s+")


def strip_attribution_frame(sentence: str) -> str:
    """'Mei Lin Zhou, Chief Financial Officer, said that revenue grew.' -> 'Revenue grew.' (NLI input only)."""
    m = _ATTRIBUTION_FRAME.match(sentence)
    if not m or m.end() >= len(sentence):
        return sentence
    rest = sentence[m.end():]
    return rest[:1].upper() + rest[1:]
