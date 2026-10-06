# Unit Tests: PostProcessNode — the output boundary
#
# Invocation canon: node(state) via the framework call path, never a bare
# execute(). PostProcessNode requires VERIFIED_EXTERNAL, so the behavioural
# tests build state at that level; the ANONYMOUS rejection lives in
# test_trust_gate.py.
#
# Layering: the module-level scan runs INSIDE execute() and blocks by returning
# a dict, not by raising. Raising would not be containment — the framework's
# output envelope falls back to state["result"] whatever the status, so a gate
# that raised would still ship the un-gated answer inside the error envelope.
# The tests below therefore assert the CONTAINMENT, not just the status.
#
# This is the OUTPUT side. The input side (screens, bounds, strips) lives in
# test_caller_contract.py and test_pre_process_node.py.
#
# Mirrors docs/03_test_spec.md S2.7 (POST-01..POST-10).
# Deterministic — no model call, no network.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import PostProcessNode, security_gate_output
from src.schemas.state import to_json

_CLEAN_REPORT = (
    "# Competitive Intelligence Compliance Q&A Result\n\n"
    "[1] the enterprise subscription tier is offered at three price points.\n"
)

# Credential-shaped strings are assembled at runtime so no such literal ever
# sits in the repository for a scanner to find.
_FAKE_JWT = "eyJ" + "a" * 12 + "." + "b" * 12 + "." + "c" * 12
_FAKE_API_KEY = "sk-" + "ABCDEF0123456789abcdef"
_FAKE_CONNECTION_STRING = "postgresql://" + "user" + ":" + "pw" + "@db.internal:5432/intel"


def _make_state(result_text, **extra) -> dict:
    state = {
        "result": result_text,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestCleanOutput:
    def test_post_01_clean_output_passes_through(self):
        result = PostProcessNode()(_make_state(_CLEAN_REPORT))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: state carries the plain string, never the enum.
        # `is str`, not isinstance: the status enum subclasses str, so
        # isinstance() would accept it and drop the regression this pins.
        assert type(result["status"]) is str  # noqa: E721
        assert result["formatted_output"] == _CLEAN_REPORT

    def test_post_02_empty_result_is_non_fatal(self):
        result = PostProcessNode()(_make_state(""))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == ""


class TestCredentialGate:
    def _assert_contained(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result["error_log"])
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))
        assert "[OUTPUT BLOCKED" in result["formatted_output"]

    def test_post_03_api_key_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\n<!-- debug api_key={_FAKE_API_KEY} -->\n"))
        self._assert_contained(result, _FAKE_API_KEY)

    def test_post_04_credential_assignment_is_blocked(self):
        result = PostProcessNode()(_make_state("# Report\n\ninternal note: password=super_secret_value_123\n"))
        self._assert_contained(result, "super_secret_value_123")

    def test_post_05_jwt_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsession token {_FAKE_JWT}\n"))
        self._assert_contained(result, _FAKE_JWT)

    def test_post_06_bearer_token_is_blocked(self):
        secret = "Bearer abcdefghijklmnopqrstuvwxyz0123456789"
        result = PostProcessNode()(_make_state(f"# Report\n\nauthorization: {secret}\n"))
        self._assert_contained(result, secret)

    def test_connection_string_is_blocked(self):
        result = PostProcessNode()(_make_state(f"# Report\n\nsource {_FAKE_CONNECTION_STRING}\n"))
        self._assert_contained(result, _FAKE_CONNECTION_STRING)


class TestPersonalDataGate:
    """The template's own output requirement: personal-data shapes are refused,
    not only the credential shapes the framework scans for."""

    def _assert_contained(self, result, secret):
        assert result["status"] == AgentStatus.ERROR.value
        assert secret not in str(result.get("formatted_output", ""))
        assert secret not in str(result.get("result", ""))

    def test_post_07_email_is_blocked(self):
        secret = "press.desk@example.com"
        self._assert_contained(PostProcessNode()(_make_state(f"# Report\n\ncontact: {secret}\n")), secret)

    def test_post_07_phone_number_is_blocked(self):
        secret = "03-1234-5678"
        self._assert_contained(PostProcessNode()(_make_state(f"# Report\n\ncall {secret}\n")), secret)

    def test_post_07_spaced_phone_number_is_blocked(self):
        secret = "090 1234 5678"
        self._assert_contained(PostProcessNode()(_make_state(f"# Report\n\ncall {secret}\n")), secret)

    def test_post_07_international_phone_number_is_blocked(self):
        secret = "+81-3-1234-5678"
        self._assert_contained(PostProcessNode()(_make_state(f"# Report\n\ncall {secret}\n")), secret)

    def test_post_07_long_id_number_is_blocked(self):
        secret = "1234 5678 9012"
        self._assert_contained(PostProcessNode()(_make_state(f"# Report\n\naccount ref {secret}\n")), secret)

    def test_structural_numbers_are_not_blocked(self):
        """The other direction: an answer full of ordinary figures is released."""
        report = (
            "# Report\n\nThree tiers were published in 2026 over a 90d window; "
            "the relevance floor was 0.25 and 12345 visitors were reported.\n"
        )
        result = PostProcessNode()(_make_state(report))
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["formatted_output"] == report


class TestStructuredPayloadIsGatedToo:
    """The structured fields are released alongside the answer, so they are
    gated with it — a violation there blocks the response instead of quietly
    dropping the citations from a success envelope."""

    def test_a_violation_in_the_citations_blocks_the_response(self):
        state = _make_state(
            _CLEAN_REPORT,
            citations=to_json(
                [{"ref": 1, "title": f"leak {_FAKE_API_KEY}", "source_url": "https://intel.example.com/a"}]
            ),
        )
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert _FAKE_API_KEY not in str(result)

    def test_clean_structured_fields_do_not_block(self):
        state = _make_state(
            _CLEAN_REPORT,
            citations=to_json([{"ref": 1, "title": "clean", "source_url": "https://intel.example.com/a"}]),
            compliance_flags=to_json([{"regime": "appi", "triggered": False, "basis": [], "note": "clean"}]),
        )
        assert PostProcessNode()(state)["status"] == AgentStatus.SUCCESS.value


class TestContainment:
    """A violating gate must CLEAR the output-bearing fields. Returning an error
    while leaving them set would still release the answer: the framework's
    output envelope reads state["result"] whatever the status is."""

    _OUTPUT_FIELDS = ("compliance_qa_answer", "grounded_answer", "citations", "compliance_flags")

    def test_every_output_bearing_field_is_cleared(self):
        leak = f"# Report\n\ncontact press.desk@example.com about {_FAKE_API_KEY}\n"
        state = _make_state(
            leak,
            compliance_qa_answer=leak,
            grounded_answer=leak,
            citations=to_json([{"ref": 1, "title": "t", "source_url": "https://intel.example.com/a"}]),
            compliance_flags=to_json([{"regime": "appi", "triggered": True, "basis": ["email"], "note": "n"}]),
        )
        result = PostProcessNode()(state)
        assert result["status"] == AgentStatus.ERROR.value
        for field in self._OUTPUT_FIELDS:
            assert result[field] is None, f"{field} still carries released content"
        assert "press.desk@example.com" not in str(result)
        assert _FAKE_API_KEY not in str(result)

    def test_the_block_carries_no_traceback_or_source_path(self):
        result = PostProcessNode()(_make_state(f"# Report\n\n{_FAKE_API_KEY}\n"))
        rendered = str(result)
        assert "Traceback" not in rendered
        assert "/src/" not in rendered
        assert ".py" not in rendered


class TestRecursiveScan:
    """The scan walks nested structures. The clean control matters as much as
    the leak probe: without it, a passing nested test cannot tell a working
    gate from a probe that never reached the value."""

    def test_a_nested_credential_is_found(self):
        nested = {
            "citations": [
                {"ref": 1, "title": "clean title", "source_url": "https://intel.example.com/a"},
                {"ref": 2, "title": _FAKE_API_KEY, "source_url": "https://intel.example.com/b"},
            ],
            "compliance_flags": [{"regime": "keihyoho", "triggered": False, "basis": [], "note": "clean"}],
        }
        assert security_gate_output(nested) == "api_key"

    def test_a_credential_two_levels_down_is_found(self):
        assert security_gate_output({"payload": {"args": {"text": f"token {_FAKE_JWT}"}}}) == "jwt"

    def test_a_clean_nested_payload_returns_none(self):
        nested = {
            "citations": [{"ref": 1, "title": "clean title", "source_url": "https://intel.example.com/a"}],
            "compliance_flags": [{"regime": "appi", "triggered": False, "basis": [], "note": "clean"}],
        }
        assert security_gate_output(nested) is None

    def test_none_is_clean(self):
        assert security_gate_output(None) is None
