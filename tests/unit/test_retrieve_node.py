# CMN-C2-299 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# CMN-C2-299 RetrieveNode.execute(self, state) -> dict takes NO config
# parameter — config values are
# exercised by SEEDING the retrieval_config State field, always through
# node(state); there is no direct execute(state, config=...) carve-out for
# this template.
#
# Golden-query expectations below were computed by replicating the exact
# scoring algorithm (title/tag/content token weights) against the real
# config/kb/web_intelligence_kb.json — not hand-derived.
#
# Mirrors docs/03_test_spec.md S2.3 (RET-01..RET-08).
# Deterministic — keyword scoring over the seeded config/kb/web_intelligence_kb.json;
# no LLM, no network. framework.* / src.* imports only.

from framework.schemas.trust_level import TrustLevel

from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

_PRICING_QUERY = "What claims has the competitor made about pricing for the enterprise " "subscription tier?"


def _make_state(query=_PRICING_QUERY, **extra) -> dict:
    state = {
        "search_query": query,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestRetrieveHappyPath:
    def test_ret_01_top_hit_is_pricing_entry(self):
        result = RetrieveNode()(_make_state())
        docs = from_json(result["retrieved_documents"])
        assert docs, "expected candidates for the pricing-claims query"
        assert docs[0]["id"] == "kb-001"
        assert docs[0]["score"] == 1.0

    def test_ret_02_scores_sorted_descending(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        scores = [d["score"] for d in docs]
        assert scores == sorted(scores, reverse=True)
        assert all(s > 0.0 for s in scores)

    def test_ret_03_entry_shape_and_excerpt_cap(self):
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        for doc in docs:
            assert set(doc.keys()) == {
                "id",
                "title",
                "category",
                "source_url",
                "retrieved_at",
                "origin",
                "score",
                "excerpt",
            }
            assert len(doc["excerpt"]) <= 400

    def test_ret_03_provenance_is_carried(self):
        """Every candidate must carry source_url + retrieved_at (docs/02_design.md:
        'Every candidate carries provenance end to end')."""
        docs = from_json(RetrieveNode()(_make_state())["retrieved_documents"])
        top = docs[0]
        assert top["source_url"] == "https://intel.example.com/reports/enterprise-tier-pricing-2026-06"
        assert top["retrieved_at"] == "2026-06-02"

    def test_retrieved_documents_is_json_string(self):
        # List-shaped state fields travel as JSON strings.
        result = RetrieveNode()(_make_state())
        assert isinstance(result["retrieved_documents"], str)


class TestRetrieveFilters:
    def test_ret_04_category_filter_restricts_pool(self):
        state = _make_state(
            query="pricing claims",
            query_filters=to_json({"category": "pricing_claims", "top_k": None}),
        )
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs, "pricing_claims category has seeded entries"
        assert {d["category"] for d in docs} == {"pricing_claims"}
        assert docs[0]["id"] == "kb-001"

    def test_ret_05_empty_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query=""))["retrieved_documents"])
        assert docs == []

    def test_no_coverage_query_yields_no_candidates(self):
        docs = from_json(RetrieveNode()(_make_state(query="quantum telepathy sandwich recipes"))["retrieved_documents"])
        assert docs == []


class TestRetrieveConfigViaState:
    """Config reaches this ANONYMOUS node ONLY through state seeding
    (retrieval_config) — no node on this template takes a config parameter."""

    def test_ret_06_state_retrieval_config_kb_path_override(self):
        state = _make_state(retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}))
        result = RetrieveNode()(state)
        assert from_json(result["retrieved_documents"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_ret_07_score_threshold_is_not_this_nodes_concern(self):
        # RetrieveNode does not read score_threshold — RerankFilterNode owns
        # the relevance floor. A strict threshold in retrieval_config must NOT
        # shrink RetrieveNode's own candidate pool.
        state = _make_state(retrieval_config=to_json({"score_threshold": 0.99}))
        docs = from_json(RetrieveNode()(state)["retrieved_documents"])
        assert docs and docs[0]["id"] == "kb-001"


class TestRetrieveNotesAccumulation:
    def test_ret_08_notes_append_never_clobber(self):
        state = _make_state(
            intake_notes=to_json(["earlier note from input validation"]),
            retrieval_config=to_json({"kb_path": "config/kb/bogus.json"}),
        )
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note from input validation"
        assert len(notes) == 2
