# Test Specification — CMN-C2-299

**Template ID:** CMN-C2-299
**Template Name:** WebIntelligenceComplianceQAAgent
**Category:** Cat 2 (nested retrieval workflow)

This document is the contract the shipped test suite implements. Every file named below exists in
the repository, and every count is the number of tests that file collects.

> **Scope.** The pipeline is deterministic and network-free: retrieval over
> `config/kb/web_intelligence_kb.json` plus whatever passages the caller sent, and rule-assembled
> answers and compliance flags. No test here exercises or implies a live external integration. See
> `docs/02_design.md`, "Deterministic assembly".

## 1. Scope and invocation conventions

- Per-node unit tests for the five inner domain nodes and the two outer boundary nodes.
- The caller-request contract: screens, bounds, parsers, and the identifier strip.
- Manifest and runtime-config consistency, and corpus integrity.
- Retrieval quality — golden queries against the shipped corpus.
- Inner-graph and outer-graph composition, including the structured `get_output()` extension.
- Boundary proofs: import isolation, state safety, invoke order, human-review propagation, server
  boot, and the end-to-end HTTP entry point.

**Invocation canon.** Every per-node test invokes the node via `node(state)` — the framework call
path, which runs the trust gate, then the input gate, then `execute()`, then the output gate —
never a bare `node.execute(state)`, which bypasses the gate under test. The state builder sets
`caller_trust_level` to `VERIFIED_EXTERNAL` for the two boundary slots (the manifest's declared
caller level) and `ANONYMOUS` for the five inner domain nodes.

**No config parameter.** Every node is `execute(self, state) -> dict` exactly. The retrieval tuning
is exercised by seeding the `retrieval_config` state field, always through `node(state)`.

**Input-mask expectations.** The framework's input gate masks `user_input`, `validated_input` and
`llm_response` before `execute()` runs. Positive-path payloads are therefore lowercase phrasing
free of personal-data shapes; the intentional-identifier tests assert the raw value is gone rather
than pinning which layer removed it. Domain fields are not input-mask targets.

**Refusals are asserted directly on the node.** The framework's own input gate sits in front of the
node on some deployment paths and not others, and an end-to-end assertion cannot tell which layer
refused. Assertions are behavioural — error status, nothing carried forward, the value never
repeated — never a gate's wording.

**Both directions, always.** Every screen is probed for what it must catch AND for what it must not
block. The false-positive probe uses the repository's own shipped corpus, not invented sentences: a
screen that refuses genuine passages fails on every legitimate request, which is worse than one
crafted attack it misses.

**Audit muting.** The framework's `shared` package is never stubbed in `sys.modules` — the framework
imports it at load time. The domain emitter is muted by an autouse fixture
(`tests/unit/conftest.py`) patching `src.nodes.<module>.emit_trace_event`; audit-assertion tests
re-patch the same attribute with a spy and assert on the event payload, never the whole-call repr.

## 2. Unit tests

### 2.1 `tests/unit/test_caller_contract.py` — 104 tests

The single validation point, so the single place the input guarantees are pinned.

| Area | What is proven |
|---|---|
| Number parser | booleans, non-numeric text, `"NaN"`, `"Infinity"`, `"-Infinity"`, raw `float("nan")` / `float("inf")`, fractional values for an integer field, and out-of-range magnitudes all refuse the request; in-range values parse |
| Non-finite per field | the matrix is applied to `top_k` AND `score_threshold`, and to a body carrying bare `NaN` / `Infinity` decoded by `json` |
| Refusal hygiene | the message names the field and never repeats the value |
| Instruction screen | the control-token class (`<|im_start|>`, `[INST]`, `<<SYS>>`) and instruction-shaped phrases are caught |
| Screen, other direction | eleven realistic business sentences and every shipped corpus entry pass untouched |
| Escaped payloads | a `\u`-escaped token is invisible to a raw scan and caught after parsing |
| Keys and nesting | a hostile mapping KEY is caught and reported by position, not echoed; a value nested three levels down is caught; a clean structure of the same depth returns None; pathological nesting is refused rather than recursed |
| Personal-data shapes | e-mail, national and international dialling forms, spaced digit runs and long identifier runs are detected; years, day counts, decimals and embedded acronyms are untouched |
| One definition | what the inbound strip removes is exactly what the outbound detector refuses |
| Contract assembly | labels outside the inert alphabet, an empty or oversize question, and malformed JSON all refuse; unknown structured fields are ignored rather than refused, because the hosting platform puts its own material there; `input_context` wins over the JSON envelope |
| Caller passages | entry cap, duplicate identifiers, identifier alphabet, free-text length caps, empty content, non-`http(s)` URLs and malformed dates all refuse; whitespace collapse prevents Markdown-block injection; personal data is stripped on the way in; an injection inside a passage refuses |

### 2.2 `tests/unit/test_pre_process_node.py` — 37 tests

| Area | What is proven |
|---|---|
| Accepted | a valid question is accepted; the validated contract travels as a JSON string; the channel reaches `enriched_context`; absent caller data is not an error |
| Refused | empty, whitespace-only and missing questions; five attack forms in the question; an injection inside a caller passage; unusable `top_k` and `score_threshold` values |
| Refusal hygiene | nothing is carried forward on the refusal path; the rejected value is never repeated; the message names the field |
| Other direction | three ordinary questions containing the same words as the attack forms are accepted |
| Identifier strip | e-mail, dialling and name shapes never reach the validated question, and the strip also covers the structured channel |
| Audit | the accepted path emits `pre_process_complete` with the passage count; the refusal path emits `pre_process_rejected` carrying no rejected value |

### 2.3 `tests/unit/test_input_validate_node.py` — 8 tests

Question normalisation, whitespace collapse, filters republished from the bridged contract, a
corrupt contract field degrading to no filters, the defensive length cap, and the empty-request
note.

### 2.4 `tests/unit/test_retrieve_node.py` — 11 tests

Candidate shape including the `origin` provenance field, excerpt cap, deterministic ordering,
category filtering, the seeded tuning taking effect, and corpus-read failures degrading with a
note rather than raising.

### 2.5 `tests/unit/test_rerank_filter_node.py` — 10 tests

The category boost, the relevance floor, the `top_k` cap, a caller override narrowing but never
widening, and deterministic ordering.

### 2.6 `tests/unit/test_generate_answer_node.py` — 16 tests

Grounded assembly from the ranked passages only, numbered citation markers, provenance carried into
each citation, inline redaction of personal-data shapes before the answer body is written, all
three compliance regimes always present, and the no-coverage answer.

### 2.7 `tests/unit/test_output_format_node.py` — 10 tests

Body, Sources list with provenance, Compliance Notes, the standing disclaimer, and the plain-string
status.

### 2.8 `tests/unit/test_post_process_node.py` — 21 tests

| Area | What is proven |
|---|---|
| Clean | a clean answer passes through; an empty result is non-fatal |
| Credentials | API key, credential assignment, JWT, Bearer token and connection string are all blocked |
| Personal data | e-mail, national / spaced / international dialling forms and long identifier runs are blocked |
| Other direction | an answer full of ordinary figures — years, day counts, decimals, visitor counts — is released byte-identical |
| Structured payload | a violation in the citations blocks the response rather than quietly dropping the field; clean structured fields do not block |
| Containment | every output-bearing field is cleared on a violation, and the blocked response carries no traceback and no source path |
| Recursive scan | a credential nested two levels down is found, AND a clean payload of the same shape returns None — without that control, a passing nested test cannot tell a working gate from a probe that never reached the value |

### 2.9 `tests/unit/test_trust_gate.py` — 7 tests

Denial and admission on both boundary slots, with no execute-only key present on a denial, and the
declared trust level of every node.

### 2.10 `tests/unit/test_config_manifest.py` — 14 tests

The manifest is flat and carries identity only; the declared class is the graph class; no
unprovisioned requirement is declared; the declared trust level matches the boundary nodes; the
runtime config file loads and reaches the constructed graph; the tuning block matches the node
defaults and is forwarded from the live file; the manifest no longer carries runtime tuning; and
the corpus is well formed with unique identifiers.

### 2.11 `tests/unit/test_retrieval_quality.py` — 16 tests

Golden queries against the shipped corpus. Expected identifiers and scores were computed by
replicating the scoring algorithm against the real corpus rather than derived by hand, and every
golden query's top hit clears the next candidate by a wide margin, so none is a near tie.

### 2.12 `tests/unit/test_domain_workflow_graph.py` — 12 tests

Inner topology and node registration, the initial-state hook seeding both the retrieval tuning and
the bridged caller contract as JSON strings, and the output shape the outer merge consumes.

### 2.13 `tests/unit/test_graph_composition.py` — 22 tests

Outer composition, the `main` slot delegation, config forwarding with a fallback when the runtime
file is unreadable, the merge contract, the structured `get_output()` extension released only on a
success status and a clean scan, and a graph-level trust denial.

### 2.14 `tests/unit/test_framework_compliance_tc06_tc07.py` — 2 tests

Overriding either default security gate raises at class-definition time.

## 3. Boundary proofs

| File | Tests | What it proves |
|---|---|---|
| `tests/proof_of_boundary/test_pb_invoke_endpoint.py` | 41 | the end-to-end HTTP entry point — see below |
| `tests/proof_of_boundary/test_pb_invoke_order.py` | 6 | a real invoke executes the fixed backbone in order |
| `tests/proof_of_boundary/test_server_boot.py` | 4 | the standalone entry point imports, compiles and reports health |
| `tests/proof_of_boundary/test_import_isolation.py` | 1 | no platform SDK import anywhere in the tree |
| `tests/proof_of_boundary/test_state_safety.py` | 1 | the state definition carries no credential-like field and no prohibited type |
| `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py` | 1 | skip stub — this template does not opt into cross-boundary human-review propagation |

### 3.1 The end-to-end entry point

`test_pb_invoke_endpoint.py` drives the real ASGI application with Bearer authentication:

| Area | What is proven |
|---|---|
| Real work | an authenticated question returns a cited answer; a different question cites different sources; an out-of-domain question says so rather than inventing |
| Caller data reaches the inner graph | a caller passage is retrieved and cited with `origin: caller_supplied`; without passages the answer falls back to the corpus; a caller passage shadows a corpus entry sharing its identifier; a caller relevance floor and a caller `top_k` visibly narrow the answer; a declared runtime value bounds it |
| Authentication | a missing and a wrong token both return 401 with the same generic body |
| Refusals | the non-finite matrix on both numeric fields; a label outside the inert alphabet, never echoed; a malformed passage; four attack forms in the question; an injection inside a passage; an escaped payload; a hostile field name |
| Adapter guards | oversized structured parameters return 413; a credential-shaped passage returns 400 naming the field without echoing the value; ordinary passage text on the same field still succeeds |
| Containment | no credential-shaped string appears anywhere in a response; a violating answer releases nothing — not the answer, not the citations, not a traceback, not a source path; personal data in a caller passage never reaches the answer |

## 4. Running the suite

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

The suite is deterministic and needs no network. It runs without a platform connection; running the
agent itself does not.
