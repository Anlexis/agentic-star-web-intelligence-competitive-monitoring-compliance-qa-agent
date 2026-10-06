"""AgentCore Platform v1.0"""

# GenerateAnswerNode
# Domain node 4: assemble the grounded answer from the ranked passages and
# produce the structured compliance outputs (citations, compliance_flags) that
# make this template's product structured rather than a block of prose.
#
# Answer assembly is DETERMINISTIC — no model call. A lead sentence plus one
# cited point per passage, each carrying a numbered marker [n] with its
# provenance (source_url / retrieved_at / origin). Nothing outside the ranked
# passages reaches the answer body, so the answer is grounded by construction.
# The synthesis seam for a model-generated answer is described in
# docs/02_design.md and config/prompts/answer_synthesis_prompt.md.
#
# Compliance lens — rule-based keyword and pattern matching only. It is never a
# model judgement and never a legal conclusion; see docs/02_design.md
# "Compliance-Lens Layer":
#   keihyoho (景品表示法) - unsubstantiated superlative or absolute claim terms
#                          reported in a cited passage
#   appi                 - personal-data shapes found in a cited passage, which
#                          are ALSO redacted inline before the excerpt is
#                          written into the answer, ahead of the output gate
#   eu_ai_act_gpai       - AI / machine-learning capability terms referenced in
#                          a cited passage
# All three regimes are always present in compliance_flags (triggered=false
# when no basis was found) — the lens applies to every answer.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).
# execute(self, state) -> dict only.

from typing import Any, ClassVar, Dict, List, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json
from src.services.caller_contract import PERSONAL_DATA_PATTERNS, REDACTION_STUB

# Answer body used when no corpus passage cleared the relevance threshold.
_NO_COVERAGE_ANSWER = (
    "The competitive-intelligence corpus does not contain sufficient coverage "
    "to answer this question. Rephrase the query with more specific "
    "competitor, product, or campaign terms, or escalate to the "
    "competitive-monitoring team for a manual review."
)

# Cited excerpt length per passage inside the answer body.
_POINT_EXCERPT_CHARS = 240

# ── 景品表示法 lens: unsubstantiated superlative / absolute claim terms ──────
# Deliberately literal substrings matched against passage content
# (case-insensitive) — a simple, deterministic, testable lens. Never a model
# judgement; every flag is framed as a prompt for human review.
_KEIHYOHO_TERMS: Tuple[str, ...] = (
    "guaranteed",
    "guarantee",
    "best value",
    "best in",
    "number one",
    "no. 1",
    "#1",
    "zero recorded defects",
    "unbeatable",
    "lowest price",
    "risk-free",
    "100 percent",
    "100%",
    "money-back",
    "completely safe",
)

# ── EU AI Act GPAI lens: AI / machine-learning / generative-AI capability ──
_AI_TERMS: Tuple[str, ...] = (
    "ai-powered",
    "artificial intelligence",
    "machine learning",
    "generative ai",
    "gpai",
    "foundation model",
    "ai system",
    "language model",
    "ai chatbot",
)

# ── APPI 2026 lens: personal-data shapes (detect AND redact) ───────────────
# The pattern set is imported, not restated: the request boundary strips these
# shapes on the way in and the output gate refuses them on the way out, and all
# three layers have to agree on what the shapes ARE or the disagreement becomes
# the leak.


def _scan_terms(text: str, terms: Tuple[str, ...]) -> List[str]:
    """Case-insensitive substring scan. Returns the matched terms, in order."""
    lowered = text.lower()
    return [t for t in terms if t in lowered]


def _redact_personal_data(text: str) -> Tuple[str, List[str]]:
    """Redact personal-data shapes from a source excerpt.

    Returns (redacted_text, matched_pattern_names). Applied to every cited
    excerpt BEFORE it is written into the answer — inline redaction, so the
    answer never carries the value even for the moment before the output gate
    would have caught it.
    """
    redacted = text
    hits: List[str] = []
    for name, pattern in PERSONAL_DATA_PATTERNS:
        if pattern.search(redacted):
            hits.append(name)
            redacted = pattern.sub(REDACTION_STUB, redacted)
    return redacted, hits


def _first_sentences(text: str, limit: int) -> str:
    """Trim an excerpt at a sentence boundary where possible, else hard-cap."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    period = cut.rfind(". ")
    if period > limit // 2:
        return cut[: period + 1]
    return cut.rstrip() + "..."


def _regime_flag(regime: str, triggered: bool, basis: List[str], note: str) -> Dict[str, Any]:
    return {"regime": regime, "triggered": triggered, "basis": basis, "note": note}


class GenerateAnswerNode(FunctionNode):
    """Rule-based grounded answer assembly + compliance-lens flags.

    Input state keys:
        ranked_documents: JSON list of surviving passages (from RerankFilterNode)
        search_query:     normalised query (for the lead sentence)

    Output state keys (partial dict):
        grounded_answer:  answer body with [n] citation markers, redacted
        citations:        JSON list [{ref, id, title, source_url, retrieved_at, origin}]
        compliance_flags: JSON list [{regime, triggered, basis, note}] - always
                          all three regimes (keihyoho / appi / eu_ai_act_gpai)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        ranked: List[Dict[str, Any]] = from_json(state.get("ranked_documents"), []) or []
        query = state.get("search_query") or ""

        citations: List[Dict[str, Any]] = []
        keihyoho_hits: List[str] = []
        ai_hits: List[str] = []
        appi_hits: List[str] = []
        redacted_count = 0

        if not ranked:
            grounded_answer = _NO_COVERAGE_ANSWER
        else:
            lines: List[str] = []
            if query:
                lines.append(
                    f"Based on the seeded competitive-intelligence corpus, the "
                    f'following passages answer the question: "{query}"'
                )
            else:
                lines.append("Based on the seeded competitive-intelligence corpus, the " "most relevant passages are:")
            lines.append("")
            for ref, doc in enumerate(ranked, start=1):
                if not isinstance(doc, dict):
                    continue
                title = str(doc.get("title", "")).strip()
                raw_excerpt = str(doc.get("excerpt", ""))
                scan_text = f"{title} {raw_excerpt}"

                for term in _scan_terms(scan_text, _KEIHYOHO_TERMS):
                    if term not in keihyoho_hits:
                        keihyoho_hits.append(term)
                for term in _scan_terms(scan_text, _AI_TERMS):
                    if term not in ai_hits:
                        ai_hits.append(term)

                redacted_excerpt, pd_hits = _redact_personal_data(raw_excerpt)
                if pd_hits:
                    redacted_count += 1
                    for hit in pd_hits:
                        if hit not in appi_hits:
                            appi_hits.append(hit)

                excerpt = _first_sentences(redacted_excerpt, _POINT_EXCERPT_CHARS)
                lines.append(f"[{ref}] {title}: {excerpt}")
                citations.append(
                    {
                        "ref": ref,
                        "id": str(doc.get("id", "")),
                        "title": title,
                        "source_url": str(doc.get("source_url", "")),
                        "retrieved_at": str(doc.get("retrieved_at", "")),
                        "origin": str(doc.get("origin", "seeded_corpus")),
                    }
                )
            grounded_answer = "\n".join(lines)

        compliance_flags = [
            _regime_flag(
                "keihyoho",
                bool(keihyoho_hits),
                keihyoho_hits,
                (
                    "Reported source claim(s) may need a 有利誤認/優良誤認 substantiation "
                    "review under 景品表示法 (Act against Unjustifiable Premiums and "
                    "Misleading Representations) - a lens over the competitor's reported "
                    "claim, not a claim made by this agent."
                )
                if keihyoho_hits
                else "No unsubstantiated superlative/absolute claim pattern found in the cited source material.",
            ),
            _regime_flag(
                "appi",
                bool(appi_hits),
                appi_hits,
                (
                    f"Personal/sensitive-data pattern(s) detected in {redacted_count} cited "
                    "source excerpt(s) and redacted before inclusion in the answer; verify "
                    "the source's APPI 2026 personal-data handling basis before further use."
                )
                if appi_hits
                else "No personal/sensitive-data pattern found in the cited source material.",
            ),
            _regime_flag(
                "eu_ai_act_gpai",
                bool(ai_hits),
                ai_hits,
                (
                    "Cited source material references an AI / machine-learning capability "
                    "claim; confirm the competitor's transparency obligations (e.g. EU AI "
                    "Act Article 50 GPAI provisions) are met in the primary source."
                )
                if ai_hits
                else "No AI / machine-learning capability claim found in the cited source material.",
            ),
        ]

        # Domain audit: answer assembled and compliance lens applied.
        emit_trace_event(
            "generate_answer_complete",
            {
                "citation_count": len(citations),
                "answer_chars": len(grounded_answer),
                "no_coverage": not ranked,
                "compliance_flags_triggered": sum(1 for f in compliance_flags if f["triggered"]),
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
            "compliance_flags": to_json(compliance_flags),
        }
