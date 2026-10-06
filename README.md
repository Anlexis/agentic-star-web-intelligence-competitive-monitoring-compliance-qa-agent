# Web Intelligence & Competitive Monitoring Compliance Q&A Agent

AI agent for answering compliance questions about web intelligence and competitive monitoring, built with Agentic Star.

> **Category**: Cat 2 (domain-specific retrieval pipeline)
> **Industry**: Common
> **Template ID**: CMN-C2-299

## Overview

Answers natural-language questions about a competitor's public claims — pricing, campaigns,
product features — over a curated intelligence corpus, and puts every answer through a
regulatory-compliance lens before returning it.

Each answer is grounded: it cites the passages it was built from, with their origin URL and the
date they were captured, and it never states anything that is not in a cited passage. Alongside
the answer the agent returns a structured flag per regime it screens for — unsubstantiated
superlative claims (景品表示法, the Japanese Act against Unjustifiable Premiums and Misleading
Representations), personal-data handling (APPI), and AI capability claims (EU AI Act GPAI
transparency) — each one a prompt for human review rather than a legal conclusion.

Callers can search the seeded corpus, or send their own freshly captured passages with the
question and have them searched, cited and compliance-screened the same way. Every caller field is
bounds-checked, personal-data shapes are stripped on the way in and refused on the way out, and a
violating answer is withheld rather than trimmed.

Typical users are competitive-monitoring and legal/compliance teams who need a cited, reviewable
answer rather than a summary they then have to verify.

This is an agent template built with the **AGENTIC STAR** development platform and the
**AgentCore Framework**. It is intended to be taken as a starting point: fork it, adapt it to
your own data and policies, and run it inside your own AGENTIC STAR deployment.

## Requirements

**This template does not run standalone.** It requires:

| Requirement | Notes |
|---|---|
| **AGENTIC STAR platform** | The agent connects to the platform at start-up. Without it, start-up fails immediately (see *Behaviour without the platform* below). Deployment guides and API documentation: [AGENTIC STAR Developers](https://developers.fd.agenticstar.tm.softbank.jp/) |
| **AgentCore Framework** (`agenticstar-agentcore`) | Installed from PyPI as a dependency. |
| Python | >= 3.11 |

```bash
pip install -e .
```

### Behaviour without the platform

The framework is designed to run **only** on AGENTIC STAR. There is no fallback or degraded
mode. If the platform is unreachable or the SDK version does not match, the agent fails at graph
compile / start-up preflight rather than starting in a partially working state. This is
intentional — a half-running agent is worse than one that refuses to start.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests run without a platform connection. Running the agent itself does not.

## Project Structure

```
src/nodes/       the five domain nodes: validate, retrieve, rerank, answer, format
src/graph/       the outer graph, the inner workflow, and the caller bridge between them
src/services/    the caller-request contract: screens, bounds, and the identifier strip
src/schemas/     the shared state definition
src/api/         the standalone HTTP entry point
tests/           unit tests and end-to-end boundary tests
config/          the manifest, the runtime parameters, the seeded corpus and the prompt
docs/            design and test documentation
```

`docs/02_design.md` describes the architecture and the security boundaries;
`docs/03_test_spec.md` maps every shipped test to what it proves.

## Customising

1. Replace `config/kb/web_intelligence_kb.json` with your own corpus. Entries need an `id`, a
   `title`, a `category`, tags, the `content`, and the provenance pair `source_url` /
   `retrieved_at` — the answer cites all three.
2. Tune `config/config.yaml`: `retrieval.top_k` caps how many passages an answer may cite and
   `retrieval.score_threshold` is the relevance floor below which a passage is dropped.
3. Adjust the compliance lens in `src/nodes/generate_answer_node.py` for the regimes you screen
   for, and the personal-data shapes in `src/services/caller_contract.py` for your jurisdiction —
   they drive both the inbound strip and the outbound gate.
4. Re-run the test suite. The tests are written against behaviour, not wording, so they should
   keep passing across those changes.

## License

MIT — see [LICENSE](LICENSE).

## Status of this repository

This template is published **as is**, by its individual author, under the MIT license. It carries
**no warranty and no support commitment**, and no organisation stands behind its behaviour or
fitness for any purpose. Issues and pull requests may or may not receive a response; that is at
the sole discretion of the repository owner.
