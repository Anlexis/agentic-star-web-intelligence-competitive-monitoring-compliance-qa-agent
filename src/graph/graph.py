"""AgentCore Platform v1.0"""

# CMN-C2-299 — outer graph (two-layer nested Cat 2 architecture)
#
# WebIntelligenceComplianceQAAgent — competitive-intelligence compliance Q&A.
#
# Architecture:
#
#   Outer backbone (fixed — do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, bounded by max_retry)
#                                             -> pre_process
#
#   The `main` slot is a GraphNode subclass (ComplianceQAGraphNode) that
#   delegates the whole domain workflow to DomainWorkflowGraph (inner graph:
#   input_validate -> retrieve -> rerank_filter -> generate_answer ->
#   output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph; the outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- validated caller contract across the boundary
#
# Class-name contract:
#   graph.py class:           WebIntelligenceComplianceQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.WebIntelligenceComplianceQAAgent"
#   src/api/server.py import: from src.graph.graph import WebIntelligenceComplianceQAAgent
#
# Rules enforced:
#   - WebIntelligenceComplianceQAAgent inherits AgentBaseGraph (direct framework inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - ComplianceQAGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the LIVE runtime config (never {})
#   - merge_output() returns only changed keys
#   - get_output() EXTENDS super().get_output()
#   - add_edges() NOT overridden on the outer graph
#   - No platform SDK imports

from pathlib import Path
from typing import Any, ClassVar, Dict

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from framework.utils.config_loader import load_agent_config
from src.graph.context_bridge import set_caller_contract
from src.nodes.post_process_node import PostProcessNode, security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

# Repo root: src/graph/graph.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Fallbacks mirror config/config.yaml so the retrieval tuning is never empty
# even where the config file is unreadable in an exotic deployment layout.
_FALLBACK_RETRIEVAL: Dict[str, Any] = {
    "top_k": 4,
    "score_threshold": 0.25,
    "kb_path": "config/kb/web_intelligence_kb.json",
}


def runtime_config() -> Dict[str, Any]:
    """Load config/config.yaml — the live runtime parameters.

    The registry loads this file and passes it to the graph constructor; the
    standalone HTTP entry point does the same, so `max_retry` and the retrieval
    tuning are live in both deployments rather than declared and ignored.

    Reading the static manifest (config/agent.yaml) here instead would return
    nothing: the manifest carries identity and compile-time requirements only,
    and a reader pointed at it degrades silently to defaults.
    """
    loaded = load_agent_config(_REPO_ROOT)
    return dict(loaded) if isinstance(loaded, dict) else {}


class ComplianceQAGraphNode(GraphNode):
    """The `main` slot: wraps the inner competitive-intelligence workflow.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   - hand the validated question to the inner graph and
                          stash the validated caller contract on the bridge
      merge_output()    - map sub_result fields into the outer state delta
      error_strategy    - "propagate": re-raise inner errors (fail fast)
    """

    # "propagate": re-raise inner graph exceptions as SubgraphError (fail fast).
    # "handle": call on_subgraph_error() instead — for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: human-review interrupts are handled inside the inner graph only.
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> Dict[str, Any]:
        """Forward the live `retrieval` tuning to the inner graph.

        Returns the tuning block under config["configurable"] — never an empty
        dict. The inner graph republishes it into inner state
        (DomainWorkflowGraph._extra_initial_state()) so the retrieval and
        rerank nodes read live top_k / score_threshold values: node execute()
        methods take no config parameter, so state seeding is the only route
        config can travel.
        """
        retrieval = runtime_config().get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        return {"configurable": {"retrieval": retrieval}}

    def get_subgraph(self) -> "Any":
        """Instantiate and return the inner domain workflow graph.

        Imported inside the method to avoid circular-import risk at module load
        time. The inner graph receives the runtime-derived config through its
        constructor; its domain nodes still take no constructor arguments and
        read config per call from seeded state.
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def extract_input(self, state: AgentState) -> str:
        """Return the question string, and bridge the validated caller contract.

        The framework hands only a string to the inner graph, so the structured
        part of the request travels on the bridge instead — set here, one step
        before the inner invoke, and read by the inner graph's initial-state
        hook. Only the contract the pre_process node already validated crosses.
        """
        set_caller_contract(from_json(state.get("caller_contract"), {}) or {})
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: Dict[str, Any]) -> Dict[str, Any]:
        """Map the inner result back into the outer state delta (changed keys only).

        Key coupling, designed together with DomainWorkflowGraph.get_output():

          Inner get_output() emits  -> "formatted_answer", "citations",
                                       "compliance_flags", "status", ...
          This merge_output() reads -> the same four keys

        `result` is set as well as `compliance_qa_answer`: the post_process slot
        and the output gate both read state["result"], so without that mapping
        the gated output would always be empty.

        citations / compliance_flags travel as JSON strings for get_output() to
        surface; they are never released here — the output gate is the single
        fail-closed checkpoint.
        """
        return {
            "compliance_qa_answer": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "compliance_flags": sub_result.get("compliance_flags"),
            "status": sub_result.get("status"),
        }


class WebIntelligenceComplianceQAAgent(AgentBaseGraph):
    """Outer graph for CMN-C2-299.

    Inherits AgentBaseGraph directly. Domain logic is fully encapsulated in
    ComplianceQAGraphNode (main slot), which delegates to DomainWorkflowGraph.

    Backbone (fixed):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY topology override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (trust gate + caller-contract validation)
      - main:         ComplianceQAGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate)

    get_output() is also overridden to extend the base envelope; see below.
    add_edges() is NOT overridden — backbone wiring belongs to the framework.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with the agent registry."""
        return "WebIntelligenceComplianceQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first — it injects the
        framework's default initialize node (schema version, session id, trust
        level) and finalize node (response metadata, total elapsed time).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = ComplianceQAGraphNode()
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden — backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Extend the base envelope with the structured compliance-lens fields.

        super().get_output() returns {output, status, trace_id, correlation_id,
        node_history}; that envelope is EXTENDED, never replaced, so callers and
        the audit trail keep relying on those fields. The structured addition:

          1. proceeds only when the final status is SUCCESS — the gated status
             written by the post_process slot. Any other status returns the base
             envelope unchanged, so a blocked run releases nothing extra;
          2. is built from vetted scalar-bearing fields only — citations
             ({ref, id, title, source_url, retrieved_at, origin}) and
             compliance_flags ({regime, triggered, basis, note}) — never a raw
             request or response object;
          3. re-scans that payload with the SAME recursive gate the post_process
             node uses (one implementation, not a second one that could drift).
             It walks nested structures, not only top-level strings. Any
             violation withholds the structured keys entirely — never a partial
             release.
        """
        env: Dict[str, Any] = dict(super().get_output(state))
        if env.get("status") != AgentStatus.SUCCESS.value:
            return env

        citations = from_json(state.get("citations"), [])
        compliance_flags = from_json(state.get("compliance_flags"), [])
        structured: Dict[str, Any] = {
            "citations": citations if isinstance(citations, list) else [],
            "compliance_flags": compliance_flags if isinstance(compliance_flags, list) else [],
        }
        if security_gate_output(structured):
            # Fail closed: withhold the structured fields, keep the base envelope.
            return env
        env.update(structured)
        return env


# Back-compat alias — config/agent.yaml names the class by dotted path and
# src/api/server.py imports it directly. Keep both names pointing at the agent.
Graph = WebIntelligenceComplianceQAAgent
