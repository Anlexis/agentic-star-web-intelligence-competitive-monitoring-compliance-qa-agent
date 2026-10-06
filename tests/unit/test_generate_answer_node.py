# CMN-C2-299 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# The grounded answer and citations are DOMAIN fields (not input-mask targets), so
# Title-Case KB titles inside them are safe to assert on.
#
# Compliance-lens coverage (CMN-specific — not in the golden's own suite):
# keihyoho (superlative-claim lens) / appi (personal-data lens, WITH inline
# redaction) / eu_ai_act_gpai (AI-capability-claim lens). All three regimes
# are ALWAYS present in compliance_flags (triggered=False when no basis was
# found) — see TestComplianceLensAlwaysPresent.
#
# APPI layering note: this node's inline redaction runs on the EXCERPT before
# it is written into grounded_answer — the DIFFERENT, later output-gate scan
# lives on PostProcessNode / graph.py get_output() (test_post_process_node.py
# / test_graph_composition.py). Do not conflate the two.
#
# Mirrors docs/03_test_spec.md S2.5 (GEN-01..GEN-09).
# Deterministic — rule-assembled from ranked_documents only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.generate_answer_node
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json


def _ranked(*entries):
    return to_json(list(entries))


def _doc(doc_id, title, excerpt, source_url="https://intel.example.com/x", retrieved_at="2026-06-01"):
    return {
        "id": doc_id,
        "title": title,
        "category": "pricing_claims",
        "source_url": source_url,
        "retrieved_at": retrieved_at,
        "score": 0.9,
        "excerpt": excerpt,
    }


def _make_state(ranked_documents, query="competitor pricing claims", **extra) -> dict:
    state = {
        "ranked_documents": ranked_documents,
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGroundedAnswer:
    def test_gen_01_answer_carries_numbered_citation_markers(self):
        ranked = _ranked(
            _doc("kb-001", "Competitor pricing tier claims", "three tiers are published for the plan."),
            _doc("kb-006", "Discount guarantee claims bulletin", "the bulletin describes seasonal terms."),
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "[1] Competitor pricing tier claims:" in answer
        assert "[2] Discount guarantee claims bulletin:" in answer

    def test_gen_02_lead_sentence_quotes_the_query(self):
        ranked = _ranked(_doc("kb-001", "Competitor pricing tier claims", "excerpt."))
        result = GenerateAnswerNode()(_make_state(ranked, query="competitor pricing claims"))
        assert 'the question: "competitor pricing claims"' in result["grounded_answer"]

    def test_gen_03_citations_mirror_ranked_order_with_provenance(self):
        ranked = _ranked(
            _doc(
                "kb-001",
                "Competitor pricing tier claims",
                "a.",
                source_url="https://intel.example.com/a",
                retrieved_at="2026-06-02",
            ),
            _doc("kb-006", "Discount guarantee claims bulletin", "b."),
        )
        citations = from_json(GenerateAnswerNode()(_make_state(ranked))["citations"])
        assert [c["ref"] for c in citations] == [1, 2]
        assert [c["id"] for c in citations] == ["kb-001", "kb-006"]
        assert citations[0]["source_url"] == "https://intel.example.com/a"
        assert citations[0]["retrieved_at"] == "2026-06-02"

    def test_citations_is_json_string(self):
        # List-shaped state fields travel as JSON strings.
        ranked = _ranked(_doc("kb-001", "Competitor pricing tier claims", "a."))
        result = GenerateAnswerNode()(_make_state(ranked))
        assert isinstance(result["citations"], str)

    def test_gen_04_answer_is_grounded_in_ranked_passages_only(self):
        ranked = _ranked(_doc("kb-001", "Competitor pricing tier claims", "the plan lists a flat annual rate."))
        answer = GenerateAnswerNode()(_make_state(ranked))["grounded_answer"]
        # Every content line traces to the single ranked passage.
        assert "the plan lists a flat annual rate." in answer
        assert "[2]" not in answer


class TestNoCoverage:
    def test_gen_05_empty_ranked_set_yields_no_coverage_answer(self):
        result = GenerateAnswerNode()(_make_state(_ranked()))
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_missing_ranked_field_is_treated_as_no_coverage(self):
        state = _make_state(None)
        del state["ranked_documents"]
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]


class TestComplianceLensAlwaysPresent:
    """All three regimes are always present, triggered=False when clean —
    the lens applies to every answer, not only when something looks wrong."""

    def test_gen_06_three_regimes_always_present_on_clean_passage(self):
        ranked = _ranked(
            _doc("kb-004", "Market share estimate disclosure", "a directional estimate with a named source.")
        )
        flags = from_json(GenerateAnswerNode()(_make_state(ranked))["compliance_flags"])
        assert {f["regime"] for f in flags} == {"keihyoho", "appi", "eu_ai_act_gpai"}
        assert all(f["triggered"] is False for f in flags)
        assert all(f["basis"] == [] for f in flags)

    def test_gen_06_three_regimes_always_present_on_no_coverage(self):
        flags = from_json(GenerateAnswerNode()(_make_state(_ranked()))["compliance_flags"])
        assert {f["regime"] for f in flags} == {"keihyoho", "appi", "eu_ai_act_gpai"}
        assert all(f["triggered"] is False for f in flags)


class TestKeihyohoLens:
    def test_gen_07_superlative_claim_triggers_keihyoho(self):
        ranked = _ranked(
            _doc("kb-001", "Competitor pricing tier claims", "the vendor guaranteed savings for every customer.")
        )
        flags = from_json(GenerateAnswerNode()(_make_state(ranked))["compliance_flags"])
        keihyoho = next(f for f in flags if f["regime"] == "keihyoho")
        assert keihyoho["triggered"] is True
        assert "guaranteed" in keihyoho["basis"]
        assert "景品表示法" in keihyoho["note"]

    def test_no_superlative_terms_keihyoho_not_triggered(self):
        ranked = _ranked(
            _doc("kb-004", "Market share estimate disclosure", "a directional estimate with a named source.")
        )
        flags = from_json(GenerateAnswerNode()(_make_state(ranked))["compliance_flags"])
        keihyoho = next(f for f in flags if f["regime"] == "keihyoho")
        assert keihyoho["triggered"] is False
        assert keihyoho["basis"] == []
        assert "No unsubstantiated" in keihyoho["note"]


class TestEuAiActGpaiLens:
    def test_gen_08_ai_capability_claim_triggers_eu_ai_act_gpai(self):
        ranked = _ranked(
            _doc("kb-003", "AI-powered recommendation engine", "the sheet promotes a machine learning model.")
        )
        flags = from_json(GenerateAnswerNode()(_make_state(ranked))["compliance_flags"])
        ai_flag = next(f for f in flags if f["regime"] == "eu_ai_act_gpai")
        assert ai_flag["triggered"] is True
        assert "machine learning" in ai_flag["basis"]
        assert "EU AI Act" in ai_flag["note"]

    def test_no_ai_terms_eu_ai_act_gpai_not_triggered(self):
        ranked = _ranked(
            _doc("kb-004", "Market share estimate disclosure", "a directional estimate with a named source.")
        )
        flags = from_json(GenerateAnswerNode()(_make_state(ranked))["compliance_flags"])
        ai_flag = next(f for f in flags if f["regime"] == "eu_ai_act_gpai")
        assert ai_flag["triggered"] is False


class TestAppiLensAndInlineRedaction:
    def test_gen_09_personal_data_triggers_appi_and_is_redacted_inline(self):
        ranked = _ranked(
            _doc(
                "kb-005",
                "Press contact handling in campaign material",
                "press inquiries can be directed to sales@example.com for a quote.",
            )
        )
        result = GenerateAnswerNode()(_make_state(ranked))
        answer = result["grounded_answer"]
        assert "sales@example.com" not in answer
        assert "[REDACTED]" in answer
        flags = from_json(result["compliance_flags"])
        appi = next(f for f in flags if f["regime"] == "appi")
        assert appi["triggered"] is True
        assert "email" in appi["basis"]
        assert "APPI" in appi["note"]

    def test_no_personal_data_appi_not_triggered(self):
        ranked = _ranked(
            _doc("kb-004", "Market share estimate disclosure", "a directional estimate with a named source.")
        )
        flags = from_json(GenerateAnswerNode()(_make_state(ranked))["compliance_flags"])
        appi = next(f for f in flags if f["regime"] == "appi")
        assert appi["triggered"] is False
        assert "No personal" in appi["note"]


class TestGenerateAnswerAudit:
    def test_gen_10_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.generate_answer_node, "emit_trace_event", spy)
        ranked = _ranked(_doc("kb-001", "Competitor pricing tier claims", "the vendor guaranteed savings."))
        GenerateAnswerNode()(_make_state(ranked))
        events = [call.args[0] for call in spy.call_args_list]
        assert "generate_answer_complete" in events
        payload = spy.call_args_list[events.index("generate_answer_complete")].args[1]
        assert payload["citation_count"] == 1
        assert payload["compliance_flags_triggered"] >= 1
