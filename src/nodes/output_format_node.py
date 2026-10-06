"""AgentCore Platform v1.0"""

# OutputFormatNode
# Domain node 5 (terminal): compose the final answer - the grounded answer
# body, the Sources list with provenance (source_url, retrieved_at, origin),
# the Compliance Notes section rendered for a human reader, and the standing
# advisory disclaimer. The disclaimer belongs to THIS node's output contract,
# not to the outer post_process slot, which gates and does not compose. This
# node never bypasses or weakens the output gate - it only assembles the
# content the gate then inspects.
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + citations + compliance_flags + status to
# the outer merge_output().
# Returns only changed state keys (partial dict).
# execute(self, state) -> dict only.

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json

# Standing disclaimer - appended to EVERY answer this template emits.
_ADVISORY_DISCLAIMER = (
    "This answer is generated from the seeded competitive-intelligence corpus for "
    "informational purposes only and does not constitute legal advice. The "
    "compliance-lens flags above are rule-based prompts for human review, not legal "
    "conclusions. Verify against the primary source and the primary regulatory text "
    "for 景品表示法, APPI 2026, and the EU AI Act before acting, and consult "
    "legal/compliance counsel."
)

# Regime code -> human-readable label for the Compliance Notes section.
_REGIME_LABELS: Dict[str, str] = {
    "keihyoho": "景品表示法 (Premiums & Representations Act)",
    "appi": "APPI 2026 (personal-data handling)",
    "eu_ai_act_gpai": "EU AI Act GPAI (transparency)",
}


class OutputFormatNode(FunctionNode):
    """Compose the final answer: body + sources + compliance notes + disclaimer.

    Input state keys:
        grounded_answer:  answer body with [n] citation markers (PII-redacted)
        citations:        JSON list [{ref, id, title, source_url, retrieved_at}]
        compliance_flags: JSON list [{regime, triggered, basis, note}]

    Output state keys (partial dict):
        formatted_answer: final rendered answer string
        status:           AgentStatus.SUCCESS.value (the plain string — never
                          the bare enum, which is not serializable into state)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        grounded_answer = state.get("grounded_answer") or ("No answer is available for this request.")
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []
        compliance_flags: List[Dict[str, Any]] = from_json(state.get("compliance_flags"), []) or []

        lines: List[str] = []
        lines.append("# Competitive Intelligence Compliance Q&A Result")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                title = str(citation.get("title", "")).strip()
                source_url = str(citation.get("source_url", "")).strip()
                retrieved_at = str(citation.get("retrieved_at", "")).strip()
                provenance = " — ".join(
                    p for p in (source_url, f"retrieved: {retrieved_at}" if retrieved_at else "") if p
                )
                suffix = f" ({provenance})" if provenance else ""
                lines.append(f"- [{ref}] {title}{suffix}")
        else:
            lines.append("- none (no corpus passage cleared the relevance threshold)")
        lines.append("")
        lines.append("## Compliance Notes")
        if compliance_flags:
            for flag in compliance_flags:
                if not isinstance(flag, dict):
                    continue
                regime = str(flag.get("regime", ""))
                label = _REGIME_LABELS.get(regime, regime or "unknown regime")
                marker = "FLAGGED" if flag.get("triggered") else "CLEAR"
                note = str(flag.get("note", "")).strip()
                lines.append(f"- [{marker}] {label}: {note}")
        else:
            lines.append("- compliance-lens produced no flags for this answer")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*{_ADVISORY_DISCLAIMER}*")

        formatted_answer = "\n".join(lines)

        # Domain audit: final answer composed, with sources, notes and disclaimer.
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
                "compliance_flags_triggered": sum(
                    1 for f in compliance_flags if isinstance(f, dict) and f.get("triggered")
                ),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
