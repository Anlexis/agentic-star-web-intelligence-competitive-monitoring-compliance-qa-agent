"""AgentCore Platform v1.0"""

# InputValidateNode
# Domain node 1: normalise the question and publish the caller's filters for
# the rest of the pipeline.
#
# It does NOT parse raw request data. Everything a caller sent was screened and
# bounds-checked once, at the request boundary (PreProcessNode via
# src/services/caller_contract.py), and reaches this node already validated:
# the question as the inner graph's input string, the filters on the seeded
# `caller_contract` field. A second parser here would be a second contract, and
# the two would drift.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).
# execute(self, state) -> dict only — no config parameter, so there is nothing
# to seed here.

import re
from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json
from src.services.caller_contract import MAX_QUERY_CHARS

_WHITESPACE_RE = re.compile(r"\s+")


class InputValidateNode(FunctionNode):
    """Normalise the question and publish the validated caller filters.

    Input state keys:
        validated_input | user_input: the validated question
        caller_contract: JSON dict of the validated caller contract

    Output state keys (partial dict):
        search_query:  normalised free-text search query
        query_filters: JSON dict {"category": str|None, "top_k": int|None,
                       "score_threshold": float|None}
        intake_notes:  (when anomalies were seen) JSON list[str]
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        raw = state.get("validated_input") or state.get("user_input", "")
        contract = from_json(state.get("caller_contract"), {}) or {}
        if not isinstance(contract, dict):
            contract = {}
        notes: List[str] = []

        query = _WHITESPACE_RE.sub(" ", raw if isinstance(raw, str) else "").strip()
        if not query:
            notes.append("InputValidateNode: empty request — no query to search.")
        if len(query) > MAX_QUERY_CHARS:
            # Defence in depth. The boundary already caps the question; this
            # keeps the node correct when the inner graph is driven directly.
            query = query[:MAX_QUERY_CHARS]
            notes.append(f"InputValidateNode: query truncated to {MAX_QUERY_CHARS} chars.")

        category: Optional[str] = contract.get("category") or None
        top_k: Optional[int] = contract.get("top_k")
        score_threshold: Optional[float] = contract.get("score_threshold")
        filters: Dict[str, Any] = {
            "category": category,
            "top_k": top_k,
            "score_threshold": score_threshold,
        }

        emit_trace_event(
            "input_validate_complete",
            {
                "query_chars": len(query),
                "has_category_filter": category is not None,
                "has_top_k_override": top_k is not None,
                "has_threshold_override": score_threshold is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "search_query": query,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
