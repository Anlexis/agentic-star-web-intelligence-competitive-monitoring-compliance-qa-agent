"""AgentCore Platform v1.0"""

# Caller-request contract for the competitive-intelligence compliance Q&A agent.
#
# One place validates everything a caller can send, so there is exactly one
# answer to "what is accepted?" — the pre_process node calls into here and
# nothing downstream re-parses raw request data.
#
# Two request channels reach this module:
#   * the free-text question (the `input` field of a request), optionally
#     wrapped in a small JSON envelope so a caller can send a question and its
#     filters in one string;
#   * `input_context`, the structured invocation parameter carried by the
#     framework, which may additionally carry the caller's own intelligence
#     passages.
#
# Rules that hold for every field:
#   * numbers are parsed by a finite + bounded parser. NaN and the infinities
#     survive float() and every comparison against them is False, so an
#     unchecked non-finite threshold silently keeps or drops everything;
#   * strings that end up in the rendered answer are whitespace-collapsed and
#     length-capped, so caller text cannot open a new Markdown block and forge
#     answer structure;
#   * labels and identifiers are restricted to inert alphabets;
#   * a value that fails any check REFUSES the request, naming the field but
#     never repeating the value;
#   * absent data is not an error — the agent falls back to the seeded corpus.

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ── Bounds ────────────────────────────────────────────────────────────────────
MAX_QUERY_CHARS = 2000
MAX_DOCUMENTS = 20
MAX_TITLE_CHARS = 200
MAX_CONTENT_CHARS = 5000
MAX_URL_CHARS = 500
MAX_TAGS = 20
MAX_CONTEXT_DEPTH = 6

TOP_K_MIN = 1
TOP_K_MAX = 20
SCORE_THRESHOLD_MIN = 0.0
SCORE_THRESHOLD_MAX = 1.0

# ── Inert alphabets ───────────────────────────────────────────────────────────
# Labels (category, channel, tags) and document identifiers are rendered into
# the answer and into audit payloads, so they are restricted rather than
# escaped. The identifier alphabet includes "-" because corpus identifiers are
# written that way (kb-001).
_LABEL_RE = re.compile(r"^[a-z0-9_]{1,32}$")
_DOCUMENT_ID_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_URL_RE = re.compile(r"^https?://[^\s<>\"'\\]{3,%d}$" % (MAX_URL_CHARS - 8))

# Field names are caller-controlled too. One is repeated back in a refusal only
# when it is short, inert, and carries no disallowed pattern of its own.
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

_WHITESPACE_RE = re.compile(r"\s+")
# Zero-width and bidi controls: invisible in a rendered answer, so they can hide
# a directive from a human reviewer while an LLM still reads it.
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

# ── Personal-data shapes ──────────────────────────────────────────────────────
# ONE definition, used by both directions of the personal-data guarantee: the
# inbound strip that rewrites these shapes out of caller text, and the outbound
# gate that refuses to release anything still matching them. Two lists would
# drift, and the drift would always favour the leak.
#
# The dialling forms are separate patterns on purpose. A single "digits and
# separators" pattern either misses the international form (no leading zero) or
# matches ordinary figures; splitting them keeps each one anchored to a shape
# a phone number actually has.
PERSONAL_DATA_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("long_id_number", re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{2,11}\b")),
    ("international_phone_number", re.compile(r"\+\d{1,3}[-\s]\d{1,4}[-\s]\d{2,4}[-\s]\d{3,4}\b")),
    ("phone_number", re.compile(r"\b0\d{1,4}[-\s]\d{2,4}[-\s]\d{4}\b")),
)
REDACTION_STUB = "[REDACTED]"


def strip_direct_identifiers(text: str) -> str:
    """Replace direct-identifier shapes in free text with a fixed stub.

    Applied to every free-text field before it is stored, whichever channel it
    arrived on: the framework's own masking covers the question field only, so
    passages arriving on the structured channel would otherwise skip it.
    """
    for _name, pattern in PERSONAL_DATA_PATTERNS:
        text = pattern.sub(REDACTION_STUB, text)
    return text


def find_personal_data(text: str) -> Optional[str]:
    """Name the first personal-data shape present in a string, or None."""
    for name, pattern in PERSONAL_DATA_PATTERNS:
        if pattern.search(text):
            return name
    return None


# ── Disallowed-instruction screen ─────────────────────────────────────────────
# Chat-template control tokens are screened as a class. They are how a payload
# forges a turn boundary, and they carry no meaning in ordinary business prose,
# so matching them cannot block real work.
_CONTROL_TOKEN_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("chat_template_token", re.compile(r"<\|[^<>|]{0,64}\|>")),
    ("instruction_token", re.compile(r"\[/?INST\]", re.IGNORECASE)),
    ("system_token", re.compile(r"<</?SYS>>", re.IGNORECASE)),
)

# Instruction-shaped phrases. Every pattern requires a verb AND its object, so
# the surrounding prose has to actually be an instruction: a passage that
# merely mentions rules, prompts or an assistant does not match. Competitive
# intelligence quotes marketing copy verbatim, and a screen that fires on it
# refuses genuine work, which is the more damaging failure of the two.
_DIRECTIVE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "override_directive",
        re.compile(
            r"\b(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|the\s+|your\s+|these\s+|those\s+)*"
            r"(?:previous|prior|above|earlier|preceding|system|initial|safety)\s+"
            r"(?:instruction|rule|prompt|direction|guardrail|guideline)s?",
            re.IGNORECASE,
        ),
    ),
    (
        "role_reassignment",
        re.compile(
            r"\b(?:you\s+are\s+now|pretend\s+to\s+be|act\s+as|behave\s+as|roleplay\s+as)\s+"
            r"(?:a|an|the)\s+"
            r"(?:system|assistant|language\s+model|ai\s+model|unrestricted|jailbroken|"
            r"admin(?:istrator)?|developer\s+mode)",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"\b(?:reveal|print|repeat|show|output|display|disclose)\s+(?:me\s+)?"
            r"(?:your|the)\s+(?:system|initial|original|hidden|full|exact)\s+"
            r"(?:prompt|instruction|rule)s?",
            re.IGNORECASE,
        ),
    ),
)

_ALL_SCREEN_PATTERNS = _CONTROL_TOKEN_PATTERNS + _DIRECTIVE_PATTERNS


class CallerDataError(ValueError):
    """A caller field failed its contract. Carries a field reference, never a value."""


def screen_text(text: str) -> Optional[str]:
    """Name the first disallowed pattern in one string, or None.

    Screens the string as received AND after the identifier strip. Both passes
    are needed and neither subsumes the other: control tokens have to be seen
    before any rewrite could consume them, and a directive split by an
    identifier-shaped run only reads as a directive once the strip has
    collapsed that run.
    """
    for candidate in (text, strip_direct_identifiers(text), _INVISIBLE_RE.sub("", text)):
        for name, pattern in _ALL_SCREEN_PATTERNS:
            if pattern.search(candidate):
                return name
    return None


def _reference(parent: str, name: object, index: int) -> str:
    """Render a caller-supplied field name safe to repeat in a refusal."""
    if isinstance(name, str) and _SAFE_NAME_RE.match(name) and screen_text(name) is None:
        return f"{parent}.{name}"
    return f"{parent}[field #{index}]"


def screen_payload(value: object, reference: str = "input_context", depth: int = 0) -> Optional[Tuple[str, str]]:
    """Depth-first screen of a parsed payload; returns (pattern, field) or None.

    Mapping KEYS are screened as well as values: a payload delivered as JSON can
    write any pattern into a key, and \\u escapes make a scan of the raw request
    text unreliable — only a scan after parsing sees what the reader will see.
    Nesting is bounded so a pathologically nested payload cannot exhaust the
    stack before the per-field checks run.
    """
    if depth > MAX_CONTEXT_DEPTH:
        return ("nesting_depth", reference)
    if isinstance(value, str):
        hit = screen_text(value)
        return (hit, reference) if hit else None
    if isinstance(value, Mapping):
        for index, (key, item) in enumerate(value.items(), start=1):
            child = _reference(reference, key, index)
            if isinstance(key, str):
                hit = screen_text(key)
                if hit:
                    return (hit, child)
            found = screen_payload(item, child, depth + 1)
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value, start=1):
            found = screen_payload(item, f"{reference}[{index}]", depth + 1)
            if found:
                return found
        return None
    return None


# ── Field parsers (every failure refuses the request) ─────────────────────────
def parse_number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
    integer: bool = False,
) -> float:
    """Parse a caller number, or refuse.

    Rejects booleans (True is an int in Python), non-numeric text, NaN and the
    infinities, and anything outside the stated range. A non-finite value that
    reaches a comparison never raises — it simply makes every comparison False,
    which turns a relevance floor into "keep nothing" or "keep everything"
    without a single error in the log.
    """
    if isinstance(value, bool) or value is None:
        raise CallerDataError(f"{field} must be a number")
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError):
            raise CallerDataError(f"{field} must be a number") from None
    elif isinstance(value, (int, float)):
        number = float(value)
    else:
        raise CallerDataError(f"{field} must be a number")
    if not math.isfinite(number):
        raise CallerDataError(f"{field} must be a finite number")
    if integer and number != int(number):
        raise CallerDataError(f"{field} must be a whole number")
    if not minimum <= number <= maximum:
        raise CallerDataError(f"{field} must be between {minimum} and {maximum}")
    return float(int(number)) if integer else number


def parse_label(value: object, *, field: str) -> str:
    """Parse an inert label (category, channel, tag), or refuse."""
    if not isinstance(value, str):
        raise CallerDataError(f"{field} must be text")
    label = value.strip().lower()
    if not _LABEL_RE.match(label):
        raise CallerDataError(f"{field} must be 1-32 characters of lowercase letters, digits or underscores")
    return label


def parse_render_text(value: object, *, field: str, limit: int, required: bool = False) -> str:
    """Parse free text that will be rendered into the answer, or refuse.

    Whitespace is collapsed to single spaces. That is the whole defence against
    caller text forging answer structure: the answer is Markdown, and a block
    element only starts at the beginning of a line, so text that cannot contain
    a newline cannot open a heading, a list item or a fenced block.
    """
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise CallerDataError(f"{field} must be text")
    text = _WHITESPACE_RE.sub(" ", _INVISIBLE_RE.sub("", value)).strip()
    if not text:
        if required:
            raise CallerDataError(f"{field} must not be empty")
        return ""
    if len(text) > limit:
        raise CallerDataError(f"{field} must be at most {limit} characters")
    return strip_direct_identifiers(text)


def _parse_url(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise CallerDataError(f"{field} must be text")
    url = value.strip()
    if not url:
        return ""
    if len(url) > MAX_URL_CHARS or not _URL_RE.match(url):
        raise CallerDataError(f"{field} must be an http(s) URL of at most {MAX_URL_CHARS} characters")
    return url


def _parse_date(value: object, *, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise CallerDataError(f"{field} must be text")
    date = value.strip()
    if not date:
        return ""
    if not _DATE_RE.match(date):
        raise CallerDataError(f"{field} must be a YYYY-MM-DD date")
    return date


def _parse_tags(value: object, *, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise CallerDataError(f"{field} must be a list of labels")
    if len(value) > MAX_TAGS:
        raise CallerDataError(f"{field} accepts at most {MAX_TAGS} labels")
    return [parse_label(item, field=f"{field}[{index}]") for index, item in enumerate(value, start=1)]


def parse_documents(value: object, *, field: str) -> List[Dict[str, Any]]:
    """Parse the caller's own intelligence passages, or refuse.

    Each passage becomes a retrievable corpus entry with the same shape as a
    seeded one, so the pipeline scores, cites and compliance-screens caller
    material exactly as it does its own.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise CallerDataError(f"{field} must be a list of documents")
    if len(value) > MAX_DOCUMENTS:
        raise CallerDataError(f"{field} accepts at most {MAX_DOCUMENTS} documents")

    documents: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(value, start=1):
        ref = f"{field}[{index}]"
        if not isinstance(entry, Mapping):
            raise CallerDataError(f"{ref} must be an object")
        document_id = entry.get("id")
        if not isinstance(document_id, str) or not _DOCUMENT_ID_RE.match(document_id.strip().lower()):
            raise CallerDataError(
                f"{ref}.id must be 1-64 characters of lowercase letters, digits, underscores or hyphens"
            )
        document_id = document_id.strip().lower()
        if document_id in seen:
            raise CallerDataError(f"{ref}.id repeats an identifier already used in this request")
        seen.add(document_id)

        content = parse_render_text(
            entry.get("content"), field=f"{ref}.content", limit=MAX_CONTENT_CHARS, required=True
        )
        title = parse_render_text(entry.get("title"), field=f"{ref}.title", limit=MAX_TITLE_CHARS)
        category = (
            parse_label(entry.get("category"), field=f"{ref}.category") if entry.get("category") is not None else ""
        )
        documents.append(
            {
                "id": document_id,
                "title": title or document_id,
                "category": category,
                "source_url": _parse_url(entry.get("source_url"), field=f"{ref}.source_url"),
                "retrieved_at": _parse_date(entry.get("retrieved_at"), field=f"{ref}.retrieved_at"),
                "tags": _parse_tags(entry.get("tags"), field=f"{ref}.tags"),
                "content": content,
            }
        )
    return documents


# ── The whole contract ────────────────────────────────────────────────────────
_ENVELOPE_QUERY_KEYS: Sequence[str] = ("query", "question")
# Keys the caller may set on either channel. Anything else on input_context is
# screened and then ignored rather than refused: the hosting platform puts its
# own material there (conversation history, routing metadata), and refusing
# unknown keys would break every hosted deployment.
_CONTEXT_KEYS: Sequence[str] = ("channel", "category", "top_k", "score_threshold", "documents")


def build_caller_contract(user_input: object, input_context: object) -> Dict[str, Any]:
    """Screen, validate and normalise everything the caller sent.

    Returns the validated contract. Raises CallerDataError naming the offending
    field — and only the field — when anything fails.
    """
    context: Mapping[str, Any] = input_context if isinstance(input_context, Mapping) else {}

    found = screen_payload(context, "input_context")
    if found:
        raise CallerDataError(f"{found[1]} contains a disallowed instruction pattern")

    text = user_input.strip() if isinstance(user_input, str) else ""
    if not text:
        raise CallerDataError("input must not be empty")
    if len(text) > MAX_QUERY_CHARS * 4:
        raise CallerDataError(f"input must be at most {MAX_QUERY_CHARS * 4} characters")

    hit = screen_text(text)
    if hit:
        raise CallerDataError("input contains a disallowed instruction pattern")

    # A caller may wrap the question and its filters in a small JSON envelope.
    # Parsing it here means the envelope's fields go through the same checks as
    # the structured channel instead of being re-parsed, unchecked, downstream.
    envelope: Mapping[str, Any] = {}
    if text.startswith("{"):
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            raise CallerDataError("input looks like JSON but did not parse") from None
        if not isinstance(decoded, Mapping):
            raise CallerDataError("input must be a question or a JSON object")
        envelope = decoded
        found = screen_payload(envelope, "input")
        if found:
            raise CallerDataError(f"{found[1]} contains a disallowed instruction pattern")
        raw_query: object = None
        for key in _ENVELOPE_QUERY_KEYS:
            if envelope.get(key) is not None:
                raw_query = envelope.get(key)
                break
        query = parse_render_text(raw_query, field="input.query", limit=MAX_QUERY_CHARS, required=True)
    else:
        query = parse_render_text(text, field="input", limit=MAX_QUERY_CHARS, required=True)

    def pick(key: str) -> object:
        """input_context wins over the envelope — one documented precedence."""
        if key in context and context.get(key) is not None:
            return context.get(key)
        return envelope.get(key)

    contract: Dict[str, Any] = {
        "query": query,
        "channel": "",
        "category": "",
        "top_k": None,
        "score_threshold": None,
        "documents": [],
    }

    channel = pick("channel")
    if channel is not None:
        contract["channel"] = parse_label(channel, field="input_context.channel")

    category = pick("category")
    if category is not None:
        contract["category"] = parse_label(category, field="input_context.category")

    top_k = pick("top_k")
    if top_k is not None:
        contract["top_k"] = int(
            parse_number(top_k, field="input_context.top_k", minimum=TOP_K_MIN, maximum=TOP_K_MAX, integer=True)
        )

    score_threshold = pick("score_threshold")
    if score_threshold is not None:
        contract["score_threshold"] = parse_number(
            score_threshold,
            field="input_context.score_threshold",
            minimum=SCORE_THRESHOLD_MIN,
            maximum=SCORE_THRESHOLD_MAX,
        )

    contract["documents"] = parse_documents(pick("documents"), field="input_context.documents")
    return contract
