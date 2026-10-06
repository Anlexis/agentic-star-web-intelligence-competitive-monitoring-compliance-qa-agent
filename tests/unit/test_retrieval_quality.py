# Unit Tests: retrieval quality over the seeded corpus
#
# Golden-query suite: drives the REAL inner retrieval chain
# (InputValidateNode -> RetrieveNode -> RerankFilterNode) through the framework
# call path against config/kb/web_intelligence_kb.json and pins the expected
# top hit per domain query. The scorer is deterministic (keyword field weights,
# stable tie-break); the expected identifiers and scores below were computed by
# replicating the scoring algorithm against the real corpus rather than derived
# by hand, and every golden query's top hit clears the next candidate by a wide
# margin, so none of them is a near tie.
#
# Mirrors docs/03_test_spec.md S2.9 (QUAL-01..QUAL-07).
# Deterministic — no model call, no network.

import json
import pathlib

import pytest

from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_KB_IDS = {
    entry["id"]
    for entry in json.loads((_ROOT / "config" / "kb" / "web_intelligence_kb.json").read_text(encoding="utf-8"))
}

_DEFAULT_SCORE_THRESHOLD = 0.25  # mirrors the config/config.yaml retrieval block


def _search(payload: str, contract: dict | None = None) -> list[dict]:
    """Run the real inner retrieval chain and return the surviving passages."""
    state = {
        "validated_input": payload,
        "caller_contract": to_json(contract or {}),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "quality-session",
        "execution_time": {},
    }
    state.update(InputValidateNode()(state))
    state.update(RetrieveNode()(state))
    state.update(RerankFilterNode()(state))
    return from_json(state["ranked_documents"], [])


# (query, expected top-1 KB entry id) — verified against the deterministic
# scorer over the real seeded corpus (config/kb/web_intelligence_kb.json).
_GOLDEN_QUERIES = [
    (
        "What claims has the competitor made about pricing for the enterprise " "subscription tier?",
        "kb-001",
    ),
    ("Does the campaign messaging include unsubstantiated superlative claims?", "kb-002"),
    ("What AI disclosure obligations apply to the recommendation engine?", "kb-003"),
    ("What market share estimate does the investor relations material disclose?", "kb-004"),
    ("What personal data handling practice applies to press contact information?", "kb-005"),
    ("Is the discount guarantee claim in the promotions bulletin substantiated?", "kb-006"),
    ("What does the free trial onboarding messaging promise customers?", "kb-007"),
    ("Does the generative AI chatbot disclose that responses are AI generated?", "kb-008"),
    ("How is the regional market entry and localization strategy described?", "kb-009"),
    ("What does the competitor privacy notice say about data handling?", "kb-010"),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(("query", "expected_id"), _GOLDEN_QUERIES)
    def test_qual_01_top_hit_per_golden_query(self, query, expected_id):
        kept = _search(query)
        assert kept, f"no passage cleared the relevance floor for: {query!r}"
        assert kept[0]["id"] == expected_id

    def test_qual_02_all_survivors_clear_the_relevance_floor(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["score"] >= _DEFAULT_SCORE_THRESHOLD

    def test_qual_03_survivor_ids_exist_in_the_seeded_kb(self):
        for query, _expected in _GOLDEN_QUERIES:
            for doc in _search(query):
                assert doc["id"] in _KB_IDS


class TestPrecision:
    def test_qual_04_ai_chatbot_query_keeps_only_the_rollout_entry(self):
        # Off-topic passages score below the floor and are cut — precision, not
        # just recall.
        kept = _search(_GOLDEN_QUERIES[7][0])
        assert [d["id"] for d in kept] == ["kb-008"]

    def test_qual_05_category_filter_restricts_to_that_category(self):
        kept = _search("pricing claims", {"category": "pricing_claims"})
        assert kept, "pricing_claims category carries seeded entries"
        assert {d["category"] for d in kept} == {"pricing_claims"}
        assert kept[0]["id"] == "kb-001"


class TestNoCoverage:
    def test_qual_06_out_of_domain_query_yields_no_survivors(self):
        assert _search("quantum telepathy sandwich recipes") == []

    def test_qual_07_no_coverage_produces_the_escalation_answer(self):
        state = {
            "ranked_documents": "[]",
            "search_query": "quantum telepathy sandwich recipes",
            "caller_trust_level": TrustLevel.ANONYMOUS.value,
            "node_history": [],
            "error_log": [],
            "session_id": "quality-session",
            "execution_time": {},
        }
        result = GenerateAnswerNode()(state)
        assert "does not contain sufficient coverage" in result["grounded_answer"]
        assert from_json(result["citations"]) == []
