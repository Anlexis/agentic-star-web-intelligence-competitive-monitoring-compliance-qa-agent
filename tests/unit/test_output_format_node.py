# CMN-C2-299 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# formatted_answer is a DOMAIN field (not an input-mask target); the standing
# advisory disclaimer + Compliance Notes section are part of THIS node's
# output contract (docs/02_design.md: "disclaimer is part of this node, NOT
# post_process").
#
# Mirrors docs/03_test_spec.md S2.6 (FMT-01..FMT-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json

_DISCLAIMER_FRAGMENT = "does not constitute legal advice"


def _citation(
    ref=1, title="Competitor pricing tier claims", source_url="https://intel.example.com/a", retrieved_at="2026-06-02"
):
    return {"ref": ref, "id": "kb-001", "title": title, "source_url": source_url, "retrieved_at": retrieved_at}


def _flag(regime, triggered, note="a note"):
    return {"regime": regime, "triggered": triggered, "basis": [], "note": note}


def _make_state(grounded_answer, citations, compliance_flags=None, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "compliance_flags": compliance_flags if compliance_flags is not None else to_json([]),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswer:
    def test_fmt_01_composes_header_body_sources_disclaimer(self):
        citations = to_json([_citation()])
        result = OutputFormatNode()(_make_state("[1] the grounded answer body.", citations))
        answer = result["formatted_answer"]
        assert answer.startswith("# Competitive Intelligence Compliance Q&A Result")
        assert "[1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] Competitor pricing tier claims (https://intel.example.com/a — retrieved: 2026-06-02)" in answer
        assert _DISCLAIMER_FRAGMENT in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: state carries the plain string, never the enum.
        # `is str`, not isinstance: the status enum subclasses str, so
        # isinstance() would accept it and drop the regression this pins.
        assert type(result["status"]) is str  # noqa: E721

    def test_fmt_02_source_suffix_url_only_when_no_retrieved_at(self):
        citations = to_json([_citation(retrieved_at="")])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] Competitor pricing tier claims (https://intel.example.com/a)" in answer

    def test_fmt_02_source_suffix_omitted_when_both_blank(self):
        citations = to_json([_citation(source_url="", retrieved_at="")])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] Competitor pricing tier claims\n" in answer + "\n"
        assert "()" not in answer

    def test_fmt_03_disclaimer_present_on_every_answer(self):
        # The disclaimer must ride WITH the substance, never separately.
        for grounded in ("a body.", ""):
            answer = OutputFormatNode()(_make_state(grounded, to_json([])))["formatted_answer"]
            assert _DISCLAIMER_FRAGMENT in answer


class TestDegradedInputs:
    def test_fmt_04_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert "- none (no corpus passage cleared the relevance threshold)" in answer

    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No answer is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestComplianceNotesSection:
    def test_fmt_06_no_flags_renders_no_flags_line(self):
        answer = OutputFormatNode()(_make_state("body.", to_json([])))["formatted_answer"]
        assert "## Compliance Notes" in answer
        assert "- compliance-lens produced no flags for this answer" in answer

    def test_fmt_07_flagged_marker_and_regime_label(self):
        flags = to_json([_flag("keihyoho", True, note="substantiation review prompt")])
        answer = OutputFormatNode()(_make_state("body.", to_json([]), compliance_flags=flags))["formatted_answer"]
        assert "- [FLAGGED] 景品表示法 (Premiums & Representations Act): substantiation review prompt" in answer

    def test_fmt_08_clear_marker_and_regime_label(self):
        flags = to_json([_flag("appi", False, note="no pattern found")])
        answer = OutputFormatNode()(_make_state("body.", to_json([]), compliance_flags=flags))["formatted_answer"]
        assert "- [CLEAR] APPI 2026 (personal-data handling): no pattern found" in answer

    def test_fmt_09_all_three_regime_labels_render(self):
        flags = to_json(
            [
                _flag("keihyoho", False, note="n1"),
                _flag("appi", False, note="n2"),
                _flag("eu_ai_act_gpai", True, note="n3"),
            ]
        )
        answer = OutputFormatNode()(_make_state("body.", to_json([]), compliance_flags=flags))["formatted_answer"]
        assert "景品表示法 (Premiums & Representations Act)" in answer
        assert "APPI 2026 (personal-data handling)" in answer
        assert "- [FLAGGED] EU AI Act GPAI (transparency): n3" in answer
