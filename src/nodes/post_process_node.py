"""AgentCore Platform v1.0"""

# PostProcessNode — the output boundary of the agent.
#
# The stated output invariant of this template is: nothing that looks like a
# credential, and nothing that looks like personal or sensitive data, leaves
# the agent. This node enforces it on the whole released surface — the rendered
# answer AND the structured payload (citations, compliance flags) — with the
# module-level scan below, called from execute().
#
# The scan is RECURSIVE. Caller text can ride inside a nested mapping, and a
# gate that only looked at top-level strings would report zero findings on a
# payload whose leak sits one level down.
#
# Two independent layers, each with its own audit event:
#   * inbound — the request boundary rewrites personal-data shapes out of
#     caller text before it is ever stored (src/services/caller_contract.py);
#     the answer assembler redacts them again in every cited excerpt;
#   * outbound — this gate refuses to release anything still matching. Both
#     directions read ONE pattern definition, so they cannot drift apart.
#
# Containment on violation: returning an error is not enough on its own. The
# framework's output envelope falls back to state["result"] whatever the
# status, so a gate that raised — or that set an error status without clearing
# the fields — would still ship the un-gated answer inside the error envelope.
# This node therefore CLEARS every output-bearing field as it blocks.
#
# No _extra_security_gate_input/_output instance methods are defined here: the
# framework auto-wraps such hooks, and defining them would change the node's
# call pipeline.

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json
from src.services.caller_contract import PERSONAL_DATA_PATTERNS

logger = logging.getLogger(__name__)

# Credential shapes. The framework scans node results for its own credential
# set; this list is the template's own boundary, so a deployment where the
# framework scan is absent or configured off is still covered.
_DISALLOWED_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{16,}", re.IGNORECASE)),
    ("aws_key", re.compile(r"\bAKIA[A-Z0-9]{16}\b")),
    ("connection_string", re.compile(r"(?:postgresql|mysql|mongodb|redis)://\S{8,}", re.IGNORECASE)),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED — disallowed content detected in the generated answer. "
    "Retry without credential-like or personal-data-like strings in the "
    "supplied material.]"
)

# Every state field that can carry released text. On a violation each one is
# overwritten, so no path out of the graph — including the framework's own
# fallback to state["result"] — can reach the un-gated answer.
_OUTPUT_BEARING_FIELDS = (
    "result",
    "formatted_output",
    "compliance_qa_answer",
    "grounded_answer",
    "citations",
    "compliance_flags",
)


def security_gate_output(content: Any) -> Optional[str]:
    """Scan released content and name the first violation, or None if clean.

    Walks nested mappings and sequences, scanning every leaf. Returns the name
    of the matched pattern — never the matched text, which would put the leak
    into the log that reports it.

    Used by:
      - PostProcessNode.execute() below, on the whole released surface;
      - WebIntelligenceComplianceQAAgent.get_output(), which re-scans the
        structured payload before surfacing it. One implementation, called
        twice — a second copy would be a second set of patterns to keep in step.
    """
    if content is None:
        return None
    if isinstance(content, dict):
        for value in content.values():
            hit = security_gate_output(value)
            if hit:
                return hit
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            hit = security_gate_output(item)
            if hit:
                return hit
        return None
    text = str(content)
    for name, pattern in _DISALLOWED_PATTERNS:
        if pattern.search(text):
            return name
    for name, pattern in PERSONAL_DATA_PATTERNS:
        if pattern.search(text):
            return name
    return None


class PostProcessNode(FunctionNode):
    """Output gate: refuse to release credential-like or personal-data-like content."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = state.get("result", "")

        if not result or not str(result).strip():
            # Nothing was generated — forward as-is, there is nothing to gate.
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        # The structured payload is released alongside the rendered answer, so
        # it is gated with it. Blocking here rather than dropping the fields
        # later means the caller is told the answer was withheld instead of
        # receiving a success envelope quietly missing its citations.
        released = {
            "result": str(result),
            "citations": from_json(state.get("citations"), []),
            "compliance_flags": from_json(state.get("compliance_flags"), []),
        }
        violation = security_gate_output(released)
        if violation:
            logger.error("PostProcessNode: output blocked — violation type: %s", violation)
            emit_trace_event(
                "post_process_blocked",
                {"violation": violation},
                state,
            )
            blocked: Dict[str, Any] = {field: None for field in _OUTPUT_BEARING_FIELDS}
            blocked.update(
                {
                    "formatted_output": _SANITISED_STUB,
                    "result": _SANITISED_STUB,
                    "status": AgentStatus.ERROR.value,
                    "error_log": [f"PostProcessNode: output blocked — disallowed content detected ({violation})"],
                }
            )
            return blocked

        # Domain audit: the answer cleared the output gate.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(result))},
            state,
        )

        return {
            "formatted_output": result,
            "status": AgentStatus.SUCCESS.value,
        }
