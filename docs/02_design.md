# Template Design Specification — CMN-C2-299

**Template ID:** CMN-C2-299
**Template Name:** WebIntelligenceComplianceQAAgent
**Category:** Cat 2 (multi-step domain workflow — retrieval pattern)
**Industry:** CMN

## Position in the architecture

| Aspect | Value |
|---|---|
| Agent class | `WebIntelligenceComplianceQAAgent` (alias `Graph`) |
| L1 Base (framework base class) | `AgentBaseGraph` — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | two-layer nested: the fixed outer backbone, with a `GraphNode` in the `main` slot wrapping an inner workflow graph |

Three-layer separation:

- **State** — a flat `TypedDict`. No models: checkpoints are msgpack-serialized and model objects
  corrupt silently on the round trip. Structured fields travel as JSON strings via `to_json()` /
  `from_json()`.
- **Node** — `FunctionNode` subclasses overriding `execute(self, state) -> dict` only. No node takes
  a `config` parameter; runtime config reaches nodes through seeded state.
- **Graph** — composition via `register_nodes()`. The outer `add_edges()` is never overridden.

## Purpose

Competitive-intelligence Q&A. Natural-language questions from competitive-monitoring and
legal/compliance staff are answered by retrieval over an intelligence corpus of competitors'
pricing, campaigns and product claims. Every answer carries source provenance — origin URL,
capture date, and whether the passage came from the deployment's own corpus or from the caller —
and passes through a rule-based compliance lens covering 景品表示法 (the Act against Unjustifiable
Premiums and Misleading Representations), APPI personal-data handling, and EU AI Act GPAI
transparency obligations, before an output gate that refuses to release personal data or
credential shapes.

The pipeline is deterministic: keyword retrieval, rule-assembled answers, rule-based compliance
flags. No model call and no live web ingest — see "Deterministic assembly" below.

## Architecture

### Outer backbone

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, bounded by max_retry)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | framework default | session id, trust level, schema version | — (framework) |
| pre_process | `PreProcessNode` | the request boundary: trust gate, disallowed-instruction screen, caller-contract validation, identifier strip → `validated_input` + `caller_contract` | `VERIFIED_EXTERNAL` |
| main | `ComplianceQAGraphNode` (`GraphNode`) | bridges the validated contract into the inner workflow and maps its result back | — (delegation) |
| post_process | `PostProcessNode` | the output boundary: refuses credential and personal-data shapes in the answer AND the structured payload, clearing every output-bearing field on a violation | `VERIFIED_EXTERNAL` |
| finalize | framework default | response metadata, elapsed time | — (framework) |

A refusal from `pre_process` contains the whole request: the framework skips every downstream node
once the state carries an error status, so `main` never runs and no answer is assembled.

### Inner graph (`DomainWorkflowGraph`, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner nodes declare `required_trust_level = TrustLevel.ANONYMOUS`. The external boundary
lives on the outer backbone; a stricter inner level would deny a genuine `VERIFIED_EXTERNAL`
request, because the inner graph inherits the caller's trust level unchanged.

| Node | Responsibility | Input state | Output state |
|------|----------------|-------------|--------------|
| `InputValidateNode` | normalise the question; republish the validated caller filters. Parses no raw request data — that happened once, at the boundary | `validated_input`, `caller_contract` | `search_query`, `query_filters`, `intake_notes` |
| `RetrieveNode` | merge the two corpus sources and score them: tokenise the question, weight title / tag / content overlap, apply the category filter, carry provenance | `search_query`, `query_filters`, `retrieval_config`, `caller_contract` | `retrieved_documents`, `intake_notes` |
| `RerankFilterNode` | rerank (category-match boost), drop everything below `score_threshold`, cap at `top_k`. Caller overrides may only narrow | `retrieved_documents`, `query_filters`, `retrieval_config` | `ranked_documents` |
| `GenerateAnswerNode` | assemble the grounded answer from the ranked passages only, redacting personal-data shapes in every cited excerpt inline, with numbered citation markers carrying provenance; produce the compliance-lens flags | `ranked_documents`, `search_query` | `grounded_answer`, `citations`, `compliance_flags` |
| `OutputFormatNode` | compose the final answer: body, Sources list with provenance, Compliance Notes, and the standing disclaimer | `grounded_answer`, `citations`, `compliance_flags` | `formatted_answer`, `status` |

`route()` on the inner graph is annotated with this graph's own `State`. The topology is linear
today, so it is never called at runtime, but a path callable's annotation is read as its input
schema and every field outside it is projected away — a base-state annotation would silently hide
the domain fields a future branch would route on, and unit tests would not notice, because a
directly called function receives whatever it is handed.

### Data flow

```
user_input + input_context
  → PreProcessNode              → validated_input + caller_contract (validated, JSON string)
  → ComplianceQAGraphNode.extract_input
        stashes the contract on the caller bridge, hands the question to the inner graph
        → input_validate        → search_query / query_filters
        → retrieve              → retrieved_documents (corpus + caller passages, with provenance)
        → rerank_filter         → ranked_documents
        → generate_answer       → grounded_answer / citations / compliance_flags
        → output_format         → formatted_answer (+ Compliance Notes + disclaimer)
     get_output()               → {formatted_answer, citations, compliance_flags, status, ...}
  → ComplianceQAGraphNode.merge_output → result, compliance_qa_answer, citations, compliance_flags
  → PostProcessNode             → formatted_output (gated) or a blocked, cleared response
```

### The caller contract

`/invoke` accepts the question as `input` and the structured parameters as `input_context`:

| Field | Type | Bounds |
|---|---|---|
| `channel` | label | `[a-z0-9_]{1,32}` |
| `category` | label | `[a-z0-9_]{1,32}`; restricts retrieval to that category |
| `top_k` | integer | finite, 1–20; may only narrow the configured cap |
| `score_threshold` | number | finite, 0.0–1.0; may only raise the configured floor |
| `documents` | list | ≤ 20 entries, each `{id, title, category, source_url, retrieved_at, tags, content}` |

A caller passage needs an inert `id` (`[a-z0-9_-]{1,64}`) and non-empty `content` (≤ 5000
characters); `source_url` must be an `http(s)` URL and `retrieved_at` a `YYYY-MM-DD` date. Free
text is whitespace-collapsed before storage — the answer is Markdown, and a block element only
starts at the beginning of a line, so text that cannot contain a newline cannot forge a heading, a
list item or a fake disclaimer.

A caller passage whose identifier matches a corpus entry shadows it: the same identifier means the
same source document, and the caller's copy is the later observation of it.

Every field is validated in one place, `src/services/caller_contract.py`. A value outside its
bounds refuses the request with a message naming the field and never repeating the value. Absent
data is not an error — the agent answers from the seeded corpus.

### The caller bridge

The framework invokes a nested graph as `subgraph.invoke(user_input, ...)`. Only the question
string crosses; neither the outer state nor the structured parameters are forwarded. The two
sanctioned subclass hooks bridge the gap: `ComplianceQAGraphNode.extract_input()` stashes the
validated contract on a `ContextVar` immediately before the inner invoke, and
`DomainWorkflowGraph._extra_initial_state()` seeds it into the inner state from there. A
`ContextVar` keeps the hand-off correct per thread and per task, so concurrent invocations in one
process cannot see each other's request.

Smuggling the data inside the question string is not viable: the framework masks that field at
every node boundary, and real passage text trips the masking heuristics, so the pipeline would
score and cite corrupted text.

### Runtime configuration

`config/agent.yaml` is the static manifest: identity, entry point, compile-time requirements. It
carries no runtime tuning. `config/config.yaml` carries the runtime parameters, and the registry
loads it and passes it to the graph constructor; `src/api/server.py` mirrors that exactly, so the
same values are live in both deployments.

```yaml
max_retry: 3          # the backbone's retry budget
timeout_s: 30         # honoured by framework service clients
retrieval:
  top_k: 4
  score_threshold: 0.25
  kb_path: "config/kb/web_intelligence_kb.json"
```

`ComplianceQAGraphNode._parent_config()` reads the `retrieval` block from that file and forwards it
under `config["configurable"]`; the inner graph republishes it into inner state as the JSON field
`retrieval_config`. That is HOW config reaches the inner nodes — no node's `execute()` takes a
config parameter, so state seeding is the only route. Module defaults mirror the file so the
forwarded block is never empty, and the end-to-end tests assert a declared value actually bounds
the answer rather than a default that happens to look the same.

### State definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `NotRequired[str]` | the validated, identifier-stripped question | outer |
| `caller_contract` | `NotRequired[Optional[str]]` (JSON) | the validated caller contract | both |
| `compliance_qa_answer` | `NotRequired[str]` | final answer, mapped from the inner `formatted_answer` | outer |
| `search_query` | `NotRequired[str]` | normalised search query | inner |
| `query_filters` | `NotRequired[Optional[str]]` (JSON) | validated `category` / `top_k` / `score_threshold` | inner |
| `retrieval_config` | `NotRequired[Optional[str]]` (JSON) | forwarded retrieval tuning | inner |
| `retrieved_documents` | `NotRequired[Optional[str]]` (JSON) | scored candidates with provenance | inner |
| `ranked_documents` | `NotRequired[Optional[str]]` (JSON) | reranked, threshold-filtered passages | inner |
| `grounded_answer` | `NotRequired[str]` | rule-assembled answer body, redacted | inner |
| `citations` | `NotRequired[Optional[str]]` (JSON) | `[{ref, id, title, source_url, retrieved_at, origin}]` | inner |
| `compliance_flags` | `NotRequired[Optional[str]]` (JSON) | `[{regime, triggered, basis, note}]` | inner |
| `formatted_answer` | `NotRequired[str]` | answer + sources + compliance notes + disclaimer | inner |
| `intake_notes` | `NotRequired[Optional[str]]` (JSON) | normalisation notes, no personal data | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

State constraints:

- flat `TypedDict` only — primitives and JSON-serialisable types;
- structured fields stored as JSON strings via `to_json()` / `from_json()`, by every producer and
  every consumer;
- domain fields are `NotRequired[...]`, so the schema is valid before any node has written;
- `formatted_output` is not re-declared — it stays a framework-owned backbone field;
- no credentials and no raw personal identifiers;
- no models, dataclasses or arbitrary Python objects.

## Security boundaries

**Trust.** Every node declares `required_trust_level`. The boundary slots require
`VERIFIED_EXTERNAL`; the standalone server elevates an authenticated Bearer caller to that level
when `INVOKE_AUTH_TOKEN` is set, and never demotes trust an upstream middleware established.

**The template owns its input guarantees.** The framework's own input gate covers the question
field and is not present on every deployment path, so the refusals live in the node that owns the
caller contract and are proven by calling that node directly. Screened as a class:

- chat-template control tokens — `<|…|>`, `[INST]`, `<<SYS>>` — which carry no meaning in business
  prose and are how a payload forges a turn boundary;
- instruction-shaped phrases, each requiring a verb AND its object, so a passage that merely
  mentions rules or an assistant does not match. Competitive intelligence quotes marketing copy
  verbatim, and a screen that fires on it refuses genuine work, which is the more damaging failure.

The screen runs depth-first over the parsed payload INCLUDING mapping keys — a payload delivered as
JSON can write a pattern into a key, and `\u` escapes make a scan of the raw request text
unreliable. It runs on each string as received and after the identifier strip: control tokens have
to be seen before any rewrite could consume them, and a directive split by an identifier-shaped run
only reads as a directive once the strip has collapsed it.

**Numbers.** Every caller-controlled number goes through one finite-and-bounded parser, which
rejects booleans, non-numeric text, NaN, the infinities, and anything out of range. Non-finite
values are the dangerous case: they survive `float()`, `json` parses bare `NaN` and `Infinity` from
a request body, and every comparison against them is False — so an unchecked relevance floor would
silently keep nothing or keep everything with no error anywhere in the run.

**Personal data, both directions, one definition.** The shapes — e-mail, long identifier runs,
national and international dialling forms — are defined once in
`src/services/caller_contract.py`. The request boundary strips them from caller text on the way in,
whichever channel it arrived on; `GenerateAnswerNode` redacts them again in every cited excerpt
before it reaches the answer body; and the output gate refuses to release anything still matching.
Two lists would drift, and the drift would always favour the leak.

**Credential shapes on the structured channel.** `input_context` is screened at the adapter with
the framework's own credential detector, before `invoke()`. The framework's mandatory output gate
scans every value of every node result, and the backbone's first node copies `input_context`
verbatim into its own result — so a credential-shaped string anywhere in it fails the FIRST node of
the graph, before any template code runs, and the caller gets an error naming nothing. On a hosted
conversation the same context is replayed every turn, so the session never recovers. The request
cannot succeed either way; refusing it at the adapter changes nothing about what is accepted, and
changes an opaque failure into an actionable one. The screen calls the same detector the gate calls
on the same assembled object, so the refusal set and the block set are one set by construction.

**The output gate contains rather than raises.** On a violation `PostProcessNode` returns an error
AND clears every output-bearing field. Raising is not containment: the framework's output envelope
falls back to `state["result"]` whatever the status, so a gate that raised — or that set an error
status without clearing — would still ship the un-gated answer, credentials included, inside the
error envelope. The same recursive scan gates the structured payload
(`citations`, `compliance_flags`), which `get_output()` releases only on a success status and a
clean scan; any violation withholds them entirely rather than partially.

No node defines `_extra_security_gate_input` or `_extra_security_gate_output`: the framework
auto-wraps those hooks, and defining them changes the node's call pipeline.

**Output rounding.** This template renders no monetary aggregates — answers are cited passage text,
source metadata and compliance flags, and no figure is computed or summarised. The precision grid
some templates enforce does not apply here; the invariant this boundary enforces instead is the
personal-data and credential refusal above, applied to every representation released, nested
structures included.

**Audit.** Every node's `execute()` emits exactly one domain event on its success path. Nodes do not
emit the framework's own lifecycle events — the framework emits those. Domain event names:

- `pre_process_complete`, `pre_process_rejected`
- `input_validate_complete`
- `retrieve_complete`
- `rerank_filter_complete`
- `generate_answer_complete`
- `output_format_complete`
- `post_process_complete`, `post_process_blocked`
- `input_context_credential_refused` (emitted by the adapter, not a node)

## Compliance lens

`GenerateAnswerNode` evaluates each ranked passage against three independent rule-based lenses and
emits a `compliance_flags` list that always carries all three regimes — `triggered: false` when no
basis was found, because the lens applies to every answer and not only when something looks wrong.

1. **景品表示法.** Flags unsubstantiated superlative or absolute claims reported in a passage
   ("guaranteed", "best in", "number one", "risk-free", "100%", "completely safe") — a signal that
   the REPORTED competitor claim may need a 有利誤認 / 優良誤認 substantiation review. It is a lens
   over what the competitor is reported to claim, never a claim made by this agent.
2. **APPI.** Flags personal-data shapes detected in the source material. Any match is both flagged
   here and redacted inline, so it is never surfaced verbatim; the flag records that a redaction
   occurred and that the source's personal-data handling basis should be verified before further
   use.
3. **EU AI Act GPAI.** Flags material referencing AI or machine-learning capability claims — a
   prompt to confirm the competitor's transparency obligations are met in the primary source. It is
   a presence check: deciding whether a disclosure appears nearby would need judgement this
   deterministic pipeline does not have, so every capability reference is surfaced for review.

The lens never calls a model and never asserts a legal conclusion. Every flag's `note` frames it as
a prompt for human review, and `OutputFormatNode` appends a standing disclaimer restating that the
answer is informational, is not legal advice, and should be verified against the primary source and
the primary regulatory text before anyone acts on it. The disclaimer belongs to that node's output
contract; the post_process slot gates, it does not compose.

## Deterministic assembly

The pipeline makes no model call. Retrieval is keyword scoring, the answer is assembled from the
ranked passages (a lead sentence plus one cited point each, redacted), and the compliance flags are
keyword and pattern matches. There is no live web ingest either: the seeded corpus is a static JSON
file, and the ingest that would build or refresh it is a separate concern outside this template's
runtime loop — which is exactly why callers can send their own freshly captured passages with a
question.

`config/prompts/answer_synthesis_prompt.md` documents the synthesis seam: a model-backed
`GenerateAnswerNode` would consume the same `ranked_documents` input and emit the same
`grounded_answer` / `citations` / `compliance_flags` contract, so no other node changes.

## Composition

- Pattern: a `GraphNode` in the outer `main` slot wrapping `DomainWorkflowGraph`.
- Error strategy: `propagate` — inner errors are re-raised rather than swallowed.
- Inner domain nodes run at `ANONYMOUS`; the outer boundary slots at `VERIFIED_EXTERNAL`.

## Import isolation

- [x] The template does not import the platform SDK.
- [x] Import targets are `framework/` and `shared/` only.
- [x] No intermediate agent class names appear in any base position.

## Design decisions

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base class | `AgentBaseGraph` | `AutonomousBaseGraph` | **`AgentBaseGraph`** | a fixed multi-step retrieval workflow, no autonomous loop |
| Composition | flat single-node slots | `GraphNode` → inner graph | **`GraphNode` → inner graph** | a five-step workflow exceeds one `main` node; nesting keeps the outer backbone untouched |
| Answer synthesis | rule-based assembly | model call | **rule-based** | deterministic and testable; the seam for a model-backed version is documented |
| Corpus | live web ingest | seeded corpus + caller passages | **seeded corpus + caller passages** | self-contained and network-free, while still answering over material the caller captured today |
| Compliance lens | model-judged analysis | rule-based keyword lens | **rule-based** | deterministic, and every flag is framed as a review prompt rather than a legal conclusion |
| Caller filters | parsed per node | validated once at the boundary | **validated once** | two parsers would be two contracts, and they would drift |
| Config location | the manifest | `config/config.yaml` | **`config/config.yaml`** | the manifest carries no runtime tuning; a reader pointed at it gets nothing and degrades to defaults with nothing failing |
