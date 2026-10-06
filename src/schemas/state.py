"""AgentCore Platform v1.0"""

# State is a flat TypedDict, never a Pydantic model: checkpoints are
# msgpack-serialized and model objects corrupt silently on the round trip.
# Extend AgentState with agent-specific fields only. Never add credentials or
# secrets.
#
# For the same reason, structured fields (dict / list[dict]) are stored as JSON
# STRINGS rather than bare containers. Producers serialize with to_json() on
# write; consumers deserialize with from_json() on read.
#
# WebIntelligenceComplianceQAAgent — a two-layer nested graph: outer backbone
# plus an inner domain workflow. The fields below cover both layers.
#
# Personal-data note: direct-identifier shapes in the request are stripped at
# the request boundary (PreProcessNode) before any field is written to state,
# whichever channel they arrived on. Cited excerpts are redacted again by
# GenerateAnswerNode before they reach the answer body, and the output gate
# (PostProcessNode) is the fail-closed backstop on everything released. Only
# the normalised query, passage summaries and the final answer are persisted —
# never a raw personal identifier.

import json
from typing import Any, NotRequired, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list state field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for CMN-C2-299.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are NotRequired so the TypedDict is valid at graph
    initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / ComplianceQAGraphNode.merge_output
    # ------------------------------------------------------------------

    # The validated, identifier-stripped question produced by PreProcessNode.
    # Raw request text is not persisted beyond that node.
    validated_input: NotRequired[str]

    # JSON STRING (to_json) of the validated caller contract, written by
    # PreProcessNode and carried into the inner graph by the caller bridge.
    # Deserialised shape: {"query": str, "channel": str, "category": str,
    # "top_k": int | None, "score_threshold": float | None,
    # "documents": list[dict]}. Every value passed its bounds check before
    # this field was written; consumers read it back with from_json() and do
    # not re-validate.
    caller_contract: NotRequired[Optional[str]]

    # Final competitive-intelligence compliance-lens answer, mapped from the
    # inner graph's formatted_answer output via merge_output.
    compliance_qa_answer: NotRequired[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode outputs
    # Normalised free-text search query (whitespace-collapsed, length-capped).
    search_query: NotRequired[str]

    # JSON STRING (to_json) of the validated caller filters. Deserialised dict
    # shape: {"category": str | None, "top_k": int | None,
    # "score_threshold": float | None}.
    # Consumers (RetrieveNode, RerankFilterNode) read it back via from_json().
    query_filters: NotRequired[Optional[str]]

    # Runtime `retrieval` block forwarded by ComplianceQAGraphNode.
    # _parent_config() -> DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"top_k": int, "score_threshold": float,
    # "kb_path": str}. Consumers (RetrieveNode, RerankFilterNode) read it
    # back via from_json().
    retrieval_config: NotRequired[Optional[str]]

    # RetrieveNode output
    # JSON STRING (to_json) of scored corpus candidates. Deserialised shape:
    # list[dict], each entry {"id": str, "title": str, "category": str,
    # "source_url": str, "retrieved_at": str, "origin": str, "score": float,
    # "excerpt": str}.
    # Consumers (RerankFilterNode) read it back via from_json().
    retrieved_documents: NotRequired[Optional[str]]

    # RerankFilterNode output
    # JSON STRING (to_json) of reranked + threshold-filtered passages, capped
    # at top_k. Same entry shape as retrieved_documents.
    # Consumers (GenerateAnswerNode) read it back via from_json().
    ranked_documents: NotRequired[Optional[str]]

    # GenerateAnswerNode outputs
    # Rule-assembled grounded answer body with numbered citation markers.
    # Personal-data shapes found in cited excerpts are redacted inline before
    # this field is written (APPI 2026 compliance lens).
    grounded_answer: NotRequired[str]

    # JSON STRING (to_json) of citations carrying provenance. Deserialised
    # shape: list[dict], each entry {"ref": int, "id": str, "title": str,
    # "source_url": str, "retrieved_at": str, "origin": str}.
    # Consumers (OutputFormatNode) read it back via from_json().
    citations: NotRequired[Optional[str]]

    # JSON STRING (to_json) of the compliance-lens flags. Deserialised shape:
    # list[dict], each entry {"regime": str, "triggered": bool,
    # "basis": list[str], "note": str} - one entry per regime
    # (keihyoho / appi / eu_ai_act_gpai), always present.
    # Consumers (OutputFormatNode) read it back via from_json().
    compliance_flags: NotRequired[Optional[str]]

    # OutputFormatNode output
    # Final formatted answer (body + sources + compliance notes + standing
    # disclaimer). Written by OutputFormatNode; surfaced to the outer graph
    # via get_output() -> merge_output().
    formatted_answer: NotRequired[str]

    # Validation / parse notes accumulated during intake (no PII).
    # JSON STRING (to_json) of list[str].
    intake_notes: NotRequired[Optional[str]]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
