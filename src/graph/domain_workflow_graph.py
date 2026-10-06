"""AgentCore Platform v1.0"""

# DomainWorkflowGraph (inner graph)
#
# The inner graph of the two-layer nested architecture.
# It encapsulates the full competitive-intelligence compliance-QA domain
# workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by ComplianceQAGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with ComplianceQAGraphNode.merge_output()
#   - No platform SDK imports
#   - Not placed under src/subagents/

from typing import Any, Dict

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.graph.context_bridge import get_caller_contract
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for CMN-C2-299.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by ComplianceQAGraphNode.get_subgraph() in graph.py, which
    passes the manifest-derived config (`_parent_config()`) into the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise the query
          -> retrieve        (RetrieveNode)       - keyword-score the seeded corpus
          -> rerank_filter   (RerankFilterNode)   - boost / threshold / top_k cut
          -> generate_answer (GenerateAnswerNode) - grounded answer + citations + compliance flags
          -> output_format   (OutputFormatNode)   - final format + compliance notes + disclaimer
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "cmn_c2_299_web_intelligence_compliance_qa_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (top_k / score_threshold / kb_path) is
        read per call by the domain nodes with safe defaults, so absence is
        non-fatal and this hook stays permissive rather than raising.
        """

    # -- Config forwarding into state (manifest -> inner nodes) -----------------

    def _extra_initial_state(self) -> Dict[str, Any]:
        """Seed the inner state with everything that cannot travel as a string.

        The framework passes only the question string into a nested graph, so
        two things are seeded here instead:

        `retrieval_config` — the live tuning block forwarded by
        ComplianceQAGraphNode._parent_config(). Domain nodes take no config
        parameter, so state seeding is the only route runtime config can reach
        RetrieveNode / RerankFilterNode.

        `caller_contract` — the validated caller contract stashed on the bridge
        by ComplianceQAGraphNode.extract_input() one step earlier. It carries
        the caller's filters and their own intelligence passages, already
        bounds-checked by the request boundary.

        Both are stored as JSON strings rather than mappings, matching the
        serialization rule the shared state schema documents.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "caller_contract": to_json(get_caller_contract()),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - SDK v1.0.0rc1
        FunctionNode subclasses take no __init__; config flows in via State
        seeding (_extra_initial_state() above), never a per-call execute()
        parameter. Every key registered here is referenced in add_edges().
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear competitive-intelligence domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear - no conditional
        branching between domain nodes. route() is implemented as required by
        the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: State) -> str:
        """Conditional routing — required by the BaseGraph contract.

        Annotated with this graph's OWN State, not the framework base state.
        A path callable's annotation is read as its input schema and every field
        outside that schema is projected away before the callable sees it, so a
        base-state annotation would hide the domain fields a future branch would
        route on — and the unit suite would still pass, because a directly
        called function receives whatever it is handed.

        The topology is linear today: add_conditional_edges() is not used, so
        this is never called at runtime. It returns END on error so an
        unexpected call cannot re-enter a processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return END
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> Dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by ComplianceQAGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations",
                                        "compliance_flags", "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("compliance_flags"),
                                        sub_result.get("status")

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "compliance_flags": state.get("compliance_flags"),
            "status": state.get("status"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
