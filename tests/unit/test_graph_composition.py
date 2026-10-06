# CMN-C2-299 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (WebIntelligenceComplianceQAAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md S3 (INT-05..INT-13).
# Deterministic — no LLM, no network. framework.* / src.* imports only.


from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    ComplianceQAGraphNode,
    Graph,
    WebIntelligenceComplianceQAAgent,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_PRICING_QUERY = "What claims has the competitor made about pricing for the enterprise " "subscription tier?"


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(WebIntelligenceComplianceQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is WebIntelligenceComplianceQAAgent

    def test_state_schema_is_state(self):
        assert WebIntelligenceComplianceQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = WebIntelligenceComplianceQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], ComplianceQAGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in WebIntelligenceComplianceQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = ComplianceQAGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = ComplianceQAGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = ComplianceQAGraphNode()
        citations = to_json([{"ref": 1, "id": "kb-001", "title": "t", "source_url": "s", "retrieved_at": "d"}])
        flags = to_json([{"regime": "keihyoho", "triggered": False, "basis": [], "note": "n"}])
        delta = node.merge_output(
            {},
            {
                "formatted_answer": "ANSWER",
                "citations": citations,
                "compliance_flags": flags,
                "status": AgentStatus.SUCCESS.value,
            },
        )
        # The inner formatted_answer surfaces as BOTH compliance_qa_answer and
        # result (the output gate reads state["result"]).
        assert delta == {
            "compliance_qa_answer": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "compliance_flags": flags,
            "status": AgentStatus.SUCCESS.value,
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert ComplianceQAGraphNode.error_strategy == "propagate"
        assert ComplianceQAGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_the_config_file(self, monkeypatch):
        # Even where the runtime config file is unreadable, the forwarded tuning
        # carries the fallback block — never {}.
        monkeypatch.setattr(src.graph.graph, "runtime_config", dict)
        cfg = ComplianceQAGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/web_intelligence_kb.json"


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_PRICING_QUERY)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_PRICING_QUERY).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Competitive Intelligence Compliance Q&A Result")
        assert "[1]" in output
        assert "does not constitute legal advice" in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_PRICING_QUERY).get("node_history", [])
        for cls_name in ("PreProcessNode", "ComplianceQAGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_no_coverage_query_still_terminates_success(self):
        result = _run("quantum telepathy sandwich recipes")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert "does not contain sufficient coverage" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """The trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot and routes past post_process to finalize — no domain answer
        is ever produced."""
        result = _run(_PRICING_QUERY, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStructuredProductOutput:
    """get_output() EXTENDS the base envelope with citations and
    compliance_flags — released on a success status only, and withheld entirely
    when the re-scan finds a violation.
    Exercised directly against get_output() (isolated from PostProcessNode's
    own pipeline gate, which is covered separately in
    test_post_process_node.py)."""

    _BASE_STATE = {
        "formatted_output": "clean answer",
        "result": "clean answer",
        "trace_id": "t-1",
        "correlation_id": "c-1",
        "node_history": [
            "InitializeNode",
            "PreProcessNode",
            "ComplianceQAGraphNode",
            "PostProcessNode",
            "FinalizeNode",
        ],
    }

    def test_int_13_success_surfaces_citations_and_compliance_flags(self):
        citations = [
            {
                "ref": 1,
                "id": "kb-001",
                "title": "t",
                "source_url": "https://intel.example.com/a",
                "retrieved_at": "2026-06-02",
            }
        ]
        flags = [{"regime": "keihyoho", "triggered": False, "basis": [], "note": "n"}]
        state = dict(
            self._BASE_STATE,
            status=AgentStatus.SUCCESS.value,
            citations=to_json(citations),
            compliance_flags=to_json(flags),
        )
        env = WebIntelligenceComplianceQAAgent().get_output(state)
        assert env["citations"] == citations
        assert env["compliance_flags"] == flags
        # Base envelope fields are still present — extended, not replaced.
        assert env["status"] == AgentStatus.SUCCESS.value

    def test_int_13_non_success_returns_base_envelope_only(self):
        state = dict(
            self._BASE_STATE, status=AgentStatus.ERROR.value, citations=to_json([]), compliance_flags=to_json([])
        )
        env = WebIntelligenceComplianceQAAgent().get_output(state)
        assert "citations" not in env
        assert "compliance_flags" not in env

    def test_int_13_violation_in_structured_payload_withholds_it_fail_closed(self):
        # A credential-shaped string smuggled into a citation title must
        # withhold the WHOLE structured payload, never a partial leak.
        citations = [
            {"ref": 1, "id": "kb-001", "title": "sk-ABCDEF0123456789abcdef", "source_url": "s", "retrieved_at": "d"}
        ]
        flags = [{"regime": "keihyoho", "triggered": False, "basis": [], "note": "n"}]
        state = dict(
            self._BASE_STATE,
            status=AgentStatus.SUCCESS.value,
            citations=to_json(citations),
            compliance_flags=to_json(flags),
        )
        env = WebIntelligenceComplianceQAAgent().get_output(state)
        assert "citations" not in env
        assert "compliance_flags" not in env
        # The base envelope (output/status/etc.) is still returned.
        assert env["status"] == AgentStatus.SUCCESS.value


class TestStateRoundTrip:
    """Serialization helpers: producers to_json(), consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "kb-001", "score": 0.69, "title": "pricing claims"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"category": "pricing_claims", "top_k": 3}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
