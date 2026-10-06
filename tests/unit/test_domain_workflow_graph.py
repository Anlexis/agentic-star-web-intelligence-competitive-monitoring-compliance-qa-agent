# CMN-C2-299 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer boundary
# is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# Mirrors docs/03_test_spec.md S3 (INT-01..INT-04).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import ComplianceQAGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_PRICING_QUERY = "What claims has the competitor made about pricing for the enterprise " "subscription tier?"


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "cmn_c2_299_web_intelligence_compliance_qa_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_block(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"top_k": 2}}})
        extra = inner._extra_initial_state()
        assert set(extra.keys()) == {"retrieval_config", "caller_contract"}
        # Structured state fields travel as JSON strings, never as mappings.
        assert isinstance(extra["retrieval_config"], str)
        assert isinstance(extra["caller_contract"], str)
        assert from_json(extra["retrieval_config"]) == {"top_k": 2}

    def test_extra_initial_state_with_no_config_is_empty_block(self):
        assert from_json(DomainWorkflowGraph()._extra_initial_state()["retrieval_config"]) == {}

    def test_int_02_the_validated_caller_contract_crosses_the_boundary(self):
        """The framework hands only a string to a nested graph, so the
        structured part of the request travels on the bridge instead."""
        from src.graph.context_bridge import set_caller_contract

        set_caller_contract({"category": "pricing_claims", "top_k": 2, "documents": []})
        try:
            seeded = from_json(DomainWorkflowGraph()._extra_initial_state()["caller_contract"])
        finally:
            set_caller_contract({})
        assert seeded["category"] == "pricing_claims"
        assert seeded["top_k"] == 2


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "compliance_flags": "[]",
                "status": AgentStatus.SUCCESS.value,
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["compliance_flags"] == "[]"
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: manifest-derived
        # config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=ComplianceQAGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_PRICING_QUERY)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Competitive Intelligence Compliance Q&A Result")
        assert "[1]" in answer
        assert "does not constitute legal advice" in answer
        citations = from_json(result["citations"])
        assert citations and citations[0]["id"] == "kb-001"
        flags = from_json(result["compliance_flags"])
        assert {f["regime"] for f in flags} == {"keihyoho", "appi", "eu_ai_act_gpai"}

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_PRICING_QUERY)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_no_coverage_query_still_terminates_success(self):
        result = self._invoke("quantum telepathy sandwich recipes")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result["formatted_answer"]
