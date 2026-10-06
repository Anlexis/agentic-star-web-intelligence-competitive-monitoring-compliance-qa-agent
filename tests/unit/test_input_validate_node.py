# Unit Tests: InputValidateNode — inner domain node 1
#
# The node normalises the question and republishes the validated caller
# filters. It does NOT parse raw request data: that happens once, at the
# request boundary. These tests pin that division — a filter reaches the
# pipeline only because the boundary validated it and the bridge carried it.
#
# Mirrors docs/03_test_spec.md S2.2 (VAL-01..VAL-07).
# Deterministic — no model call, no network.

from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json, to_json


def _make_state(payload, contract=None, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_contract": to_json(contract) if contract is not None else None,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestQueryNormalisation:
    def test_val_01_the_question_becomes_the_search_query(self):
        result = InputValidateNode()(_make_state("competitor pricing tier claims"))
        assert result["search_query"] == "competitor pricing tier claims"
        assert from_json(result["query_filters"]) == {
            "category": None,
            "top_k": None,
            "score_threshold": None,
        }

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  competitor   pricing\n tier claims "))
        assert result["search_query"] == "competitor pricing tier claims"

    def test_val_03_query_filters_travel_as_a_json_string(self):
        result = InputValidateNode()(_make_state("competitor pricing claims"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestValidatedFiltersArePublished:
    def test_val_04_filters_come_from_the_bridged_contract(self):
        contract = {
            "query": "ai disclosure obligations",
            "category": "ai_disclosure",
            "top_k": 2,
            "score_threshold": 0.4,
            "documents": [],
        }
        result = InputValidateNode()(_make_state("ai disclosure obligations", contract))
        assert from_json(result["query_filters"]) == {
            "category": "ai_disclosure",
            "top_k": 2,
            "score_threshold": 0.4,
        }

    def test_val_05_no_contract_means_no_filters(self):
        result = InputValidateNode()(_make_state("pricing claims"))
        filters = from_json(result["query_filters"])
        assert filters["category"] is None and filters["top_k"] is None

    def test_a_corrupt_contract_field_degrades_to_no_filters(self):
        result = InputValidateNode()(_make_state("pricing claims", caller_contract="not json"))
        assert from_json(result["query_filters"])["category"] is None


class TestDefensiveGuards:
    def test_val_06_an_oversize_question_is_capped(self):
        """The boundary already caps it; this keeps the node correct when the
        inner graph is driven directly."""
        result = InputValidateNode()(_make_state("competitor " * 300))
        assert len(result["search_query"]) == 2000
        assert any("truncated" in n for n in from_json(result.get("intake_notes"), []))

    def test_val_07_an_empty_request_yields_a_note_not_an_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["search_query"] == ""
        assert any("empty request" in n for n in from_json(result.get("intake_notes"), []))
