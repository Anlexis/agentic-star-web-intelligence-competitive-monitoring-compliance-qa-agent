# Answer Synthesis Prompt — CMN-C2-299 (model-synthesis seam)

> **This prompt is NOT used at runtime.** `GenerateAnswerNode` assembles the
> answer deterministically from `ranked_documents` and applies a rule-based
> compliance lens; no node reads this file. It documents the contract a
> model-backed replacement would have to honour — see `docs/02_design.md`,
> "Deterministic assembly" — so that swap changes only the inside of
> `GenerateAnswerNode.execute()`.

## Contract for a model-backed GenerateAnswerNode

- **Input:** the same `ranked_documents` JSON (id / title / category /
  source_url / retrieved_at / origin / score / excerpt) and `search_query` the
  deterministic node reads.
- **Output:** the same state contract — `grounded_answer` (str, with numbered
  `[n]` citation markers), `citations` (JSON list of
  `{ref, id, title, source_url, retrieved_at, origin}`), and `compliance_flags`
  (JSON list of `{regime, triggered, basis, note}` — always all three
  regimes: `keihyoho`, `appi`, `eu_ai_act_gpai`).
- **Grounding rule:** every factual statement in the answer must be traceable
  to one of the supplied passages via a `[n]` marker; content not present in
  the passages must not be asserted.
- **No-coverage rule:** when no passage supports the question, say so and
  recommend refining the query or escalating to the competitive-monitoring
  team — never answer from parametric knowledge.
- **Redaction rule (unchanged):** personal-data shapes found in a cited passage
  must be redacted BEFORE the excerpt is included in `grounded_answer`, and the
  `appi` compliance flag must record that a redaction occurred. This is a hard
  invariant that a model-backed replacement may not relax.
- **Compliance-lens rule:** the three-regime lens (景品表示法 / APPI / EU AI Act
  GPAI) stays rule-based and deterministic either way — it is a compliance
  control, not a generation-quality feature, and every flag must keep framing
  itself as a prompt for human review, never a legal conclusion.
- **Tone:** neutral, compliance-appropriate, no individualized legal
  conclusions (the disclaimer is appended downstream by `OutputFormatNode`).

## Prompt template

```
You answer competitive-intelligence questions strictly from the corpus
passages provided below. You also apply a compliance lens covering
景品表示法 (Premiums & Representations Act), APPI 2026 personal-data
handling, and EU AI Act GPAI transparency.

Question:
{search_query}

Passages (each with a reference number, provenance already attached):
{ranked_documents}

Rules:
1. Use ONLY the passages above. If they do not answer the question, say the
   corpus has insufficient coverage and stop.
2. Mark every factual statement with the [n] reference of its passage.
3. Redact any personal/sensitive-data pattern found in a passage before
   quoting it; never surface raw personal data.
4. For each passage, flag unsubstantiated superlative/absolute claims
   (景品表示法), personal-data patterns (APPI 2026), and AI/ML capability
   claims (EU AI Act GPAI) — as prompts for human review, not legal
   conclusions.
5. Do not give a legal determination or individualized legal advice.
6. Keep the answer under 300 words.
```

## Configuration coupling

There is no model configuration in `config/config.yaml` today: a block nothing
reads is a declaration that goes quietly dead, so the shipped config carries
only what the running code consumes. Adding a model-backed node means adding
its settings alongside `retrieval:` and forwarding them the same way, through
`ComplianceQAGraphNode._parent_config()` and the inner graph's initial-state
hook — the route every runtime value already travels.
