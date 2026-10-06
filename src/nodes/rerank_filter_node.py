"""AgentCore Platform v1.0"""

# RerankFilterNode
# Domain node 3: rerank the retrieval candidates and enforce the relevance
# floor. Deterministic: a small category-match boost on top of the retrieval
# score, drop everything below `score_threshold`, cap the survivors at `top_k`.
#
# Config: reads `top_k` / `score_threshold` from the state-seeded
# `retrieval_config` field, falling back to module defaults that mirror
# config/config.yaml. execute(self, state) -> dict only — no config parameter.
#
# Caller overrides may only NARROW the result set — a smaller top_k, a higher
# relevance floor. Both arrive already validated (finite, in range) from the
# request boundary. Narrow-only keeps a caller from widening the answer past
# the tuning the deployment chose, while still letting them ask for a tighter
# one.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import math
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_state import AgentState
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.schemas.state import from_json, to_json

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
}

# Boost applied when a candidate's category matches the caller's filter.
_CATEGORY_BOOST = 0.1


def _resolve_retrieval_config(state: AgentState) -> Dict[str, Any]:
    """Effective retrieval config: state-seeded retrieval_config over defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


class RerankFilterNode(FunctionNode):
    """Rerank candidates, apply the score threshold, cap at top_k.

    Input state keys:
        retrieved_documents: JSON list of scored candidates (from RetrieveNode)
        query_filters:       JSON dict with the optional category filter and the
                             validated top_k / score_threshold overrides
        retrieval_config:    state-seeded retrieval tuning (JSON)

    Output state keys (partial dict):
        ranked_documents: JSON list of surviving passages (score desc, <= top_k)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: AgentState) -> Dict[str, Any]:
        candidates: List[Dict[str, Any]] = from_json(state.get("retrieved_documents"), []) or []
        filters = from_json(state.get("query_filters"), {}) or {}
        retrieval_cfg = _resolve_retrieval_config(state)

        try:
            top_k = int(retrieval_cfg.get("top_k", _DEFAULT_RETRIEVAL["top_k"]))
        except (TypeError, ValueError):
            top_k = int(_DEFAULT_RETRIEVAL["top_k"])
        top_k = max(1, min(20, top_k))
        # A stricter caller override wins. The value arrived validated from the
        # request boundary; the guards here keep the node correct when the inner
        # graph is driven directly. bool is excluded explicitly because it is an
        # int in Python, so True would silently read as top_k = 1.
        caller_top_k = filters.get("top_k")
        if isinstance(caller_top_k, int) and not isinstance(caller_top_k, bool) and 1 <= caller_top_k < top_k:
            top_k = caller_top_k

        try:
            score_threshold = float(retrieval_cfg.get("score_threshold", _DEFAULT_RETRIEVAL["score_threshold"]))
        except (TypeError, ValueError):
            score_threshold = float(_DEFAULT_RETRIEVAL["score_threshold"])
        score_threshold = max(0.0, min(1.0, score_threshold))
        # A stricter caller floor wins; a looser one is ignored. Non-finite values
        # never survive the request boundary, and are re-excluded here because a
        # NaN reaching a comparison would make every one of them False and turn
        # the relevance floor into "keep nothing" with no error anywhere.
        caller_threshold = filters.get("score_threshold")
        if (
            isinstance(caller_threshold, (int, float))
            and not isinstance(caller_threshold, bool)
            and math.isfinite(caller_threshold)
        ):
            score_threshold = max(score_threshold, float(caller_threshold))

        category = filters.get("category")

        reranked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            entry = dict(candidate)  # local copy - inputs stay immutable
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            if category and str(entry.get("category", "")).lower() == str(category).lower():
                score = min(1.0, score + _CATEGORY_BOOST)
            entry["score"] = round(score, 4)
            reranked.append(entry)

        # Deterministic ordering: score desc, then id asc for stable ties.
        reranked.sort(key=lambda c: (-c.get("score", 0.0), str(c.get("id", ""))))

        kept = [c for c in reranked if c.get("score", 0.0) >= score_threshold][:top_k]
        dropped = len(reranked) - len(kept)

        # Domain audit: rerank and relevance floor applied.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "kept": len(kept),
                "dropped": dropped,
                "score_threshold": score_threshold,
                "top_k": top_k,
            },
            state,
        )

        return {"ranked_documents": to_json(kept)}
