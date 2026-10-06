"""AgentCore Platform v1.0"""

# PreProcessNode — the request boundary of the competitive-intelligence
# compliance Q&A agent, and the one place where caller data is validated.
#
# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# Trust: this node occupies the backbone's pre_process slot and declares
# VERIFIED_EXTERNAL, so an unauthenticated or unverified caller is denied
# before the retrieval workflow runs. The framework returns an error status
# from the denied node and every downstream node is then skipped, so a refusal
# here contains the whole request rather than merely annotating it.
#
# Validation: everything a caller can send — the question, the optional JSON
# envelope around it, and the structured invocation parameters including the
# caller's own intelligence passages — is screened and bounds-checked by
# src/services/caller_contract.py. The template owns that guarantee itself
# rather than relying on the framework's own input gate, which covers the
# question field only and is not present on every deployment path. A refusal
# names the field and never repeats the value.
#
# The validated contract is handed to the inner workflow through the caller
# bridge (src/graph/context_bridge.py); nothing downstream re-parses raw
# request data.

from typing import Any, ClassVar, Dict

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import to_json
from src.services.caller_contract import (
    CallerDataError,
    build_caller_contract,
    strip_direct_identifiers,
)


class PreProcessNode(FunctionNode):
    """Trust gate, disallowed-instruction screen and caller-contract validation."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        try:
            contract = build_caller_contract(user_input, input_context)
        except CallerDataError as rejected:
            # The message names the field; the rejected value is never carried
            # into the log, the audit event or the response.
            emit_trace_event(
                "pre_process_rejected",
                {"reason": "caller_contract"},
                state,
            )
            return {
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PreProcessNode: request rejected — {rejected}"],
            }

        validated_input = strip_direct_identifiers(contract["query"])
        contract["query"] = validated_input

        emit_trace_event(
            "pre_process_complete",
            {
                "input_chars": len(validated_input),
                "caller_documents": len(contract["documents"]),
                "has_category_filter": bool(contract["category"]),
                "has_top_k_override": contract["top_k"] is not None,
                "has_threshold_override": contract["score_threshold"] is not None,
            },
            state,
        )

        enriched_context: Dict[str, Any] = {
            "source": "WebIntelligenceComplianceQAAgent",
            "channel": contract["channel"] or "unknown",
        }
        return {
            "validated_input": validated_input,
            # JSON string, not a bare mapping: structured State fields are
            # serialized so a checkpointed run round-trips without silent
            # corruption (see src/schemas/state.py).
            "caller_contract": to_json(contract),
            "enriched_context": enriched_context,
            "status": AgentStatus.SUCCESS.value,
        }
