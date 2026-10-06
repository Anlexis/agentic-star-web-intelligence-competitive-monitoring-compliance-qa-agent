# Unit Tests: manifest and runtime-config consistency
#
# Two files, two jobs, and the difference matters:
#
#   config/agent.yaml  — the static manifest the registry reads to DISCOVER the
#                        agent: identity, entry point, compile-time requirements.
#   config/config.yaml — the runtime parameters the registry LOADS and passes to
#                        the graph constructor.
#
# A reader pointed at the wrong one does not fail; it gets nothing and degrades
# to its defaults, so every declared value goes quietly dead while the suite
# stays green. These tests pin each declaration to the code that consumes it,
# and the end-to-end test proves a declared value actually arrives.
#
# Mirrors docs/03_test_spec.md S2.8 (CFG-01..CFG-09).
# Deterministic — no model call, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel

from src.graph.graph import (
    ComplianceQAGraphNode,
    WebIntelligenceComplianceQAAgent,
    runtime_config,
)
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_CONFIG = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestIdentity:
    def test_cfg_01_the_manifest_is_flat(self):
        """Every key sits at the root: the registry reads them there, and a
        nested block is simply not seen."""
        assert _MANIFEST["id"] == "CMN-C2-299"
        assert _MANIFEST["name"] == "WebIntelligenceComplianceQAAgent"
        assert "agent" not in _MANIFEST

    def test_cfg_02_the_declared_class_is_the_graph_class(self):
        assert _MANIFEST["class"] == "src.graph.graph.WebIntelligenceComplianceQAAgent"
        assert _MANIFEST["name"] == WebIntelligenceComplianceQAAgent().name

    def test_cfg_03_category_and_industry(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "CMN"
        assert _MANIFEST["base_type"] == "RAGAgent"

    def test_cfg_04_no_unprovisioned_requirement_is_declared(self):
        """A declared secret or extra that is not actually provisioned makes the
        agent fail at compile time, so both lists stay empty until one is used."""
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_cfg_05_required_trust_level_matches_the_boundary_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared


class TestRuntimeConfig:
    def test_cfg_06_max_retry_is_within_the_framework_ceiling(self):
        max_retry = _CONFIG["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10

    def test_cfg_07_runtime_config_loads_the_live_file(self):
        loaded = runtime_config()
        assert loaded["max_retry"] == _CONFIG["max_retry"]
        assert loaded["retrieval"] == _CONFIG["retrieval"]

    def test_cfg_08_the_agent_is_constructed_with_the_live_config(self):
        """Declared retry budget reaches the graph rather than the framework
        default that happens to look the same."""
        agent = WebIntelligenceComplianceQAAgent(config=runtime_config())
        assert agent.config["max_retry"] == _CONFIG["max_retry"]

    def test_human_review_is_not_enabled(self):
        assert (_CONFIG.get("hitl") or {}).get("enabled", False) is False


class TestRetrievalTuning:
    def test_cfg_09_the_tuning_block_matches_the_node_defaults(self):
        """The node defaults mirror the file. A drift between them would change
        behaviour silently in any deployment where the file is unreadable."""
        from src.nodes.rerank_filter_node import _DEFAULT_RETRIEVAL as rerank_defaults
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        retrieval = _CONFIG["retrieval"]
        assert retrieval["top_k"] == retrieve_defaults["top_k"] == rerank_defaults["top_k"]
        assert (
            retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == rerank_defaults["score_threshold"]
        )
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_the_tuning_block_is_forwarded_from_the_live_file(self):
        forwarded = ComplianceQAGraphNode()._parent_config()["configurable"]["retrieval"]
        assert forwarded == _CONFIG["retrieval"]
        assert forwarded, "the forwarded tuning block must never be empty"

    def test_the_manifest_no_longer_carries_runtime_tuning(self):
        """Guard against a reader being pointed back at the manifest: the tuning
        lives in one place, and the manifest is not it."""
        assert "retrieval" not in _MANIFEST


class TestSeededCorpus:
    def _entries(self):
        return json.loads((_ROOT / _CONFIG["retrieval"]["kb_path"]).read_text(encoding="utf-8"))

    def test_the_corpus_is_a_well_formed_entry_list_with_provenance(self):
        entries = self._entries()
        assert isinstance(entries, list)
        assert len(entries) >= 5, "the seeded corpus must carry a usable baseline"
        for entry in entries:
            assert set(entry.keys()) == {
                "id",
                "title",
                "category",
                "source_url",
                "retrieved_at",
                "tags",
                "content",
            }
            assert entry["id"] and entry["title"] and entry["content"]
            assert entry["source_url"] and entry["retrieved_at"]

    def test_corpus_identifiers_are_unique(self):
        ids = [e["id"] for e in self._entries()]
        assert len(ids) == len(set(ids))
