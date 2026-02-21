# Risk-Aware Introspective RAG (RAI-RAG)

**RAI-RAG** is a research framework for **safety-aligned Retrieval-Augmented Generation (RAG)** that treats **retrieval as a safety-critical decision**, rather than a neutral preprocessing step.  
It extends **introspective reasoning, safety-dominant search, and meta-learning** to control *whether to retrieve*, *what evidence to use*, and *how retrieved knowledge influences generation*.

> **Core insight**  
> A language model can be aligned at generation time and still produce *grounded but unsafe outputs* if retrieval supplies unsafe evidence.  
> **RAI-RAG aligns evidence access, not only output.**


##  Key Features

- **Structured Introspective Reasoning**
  - Explicit reasoning steps with strict `<Reasoning_step>` → `<Output>` separation
  - Enforced refusal and safety constraints

- **Risk-Aware Retrieval Gating**
  - Dynamically decides `{Retrieve | Restrict | No-Retrieve}`
  - Retrieval becomes a *reasoned action*, not a mandatory step

- **Evidence-Level Safety Alignment**
  - Safety scoring and pruning of retrieved documents
  - Prevents *evidence-driven jailbreaks* and *grounded harm*

- **Safety-Informed MCTS (SI-MCTS)**
  - Search over reasoning + retrieval trajectories
  - Safety-dominant pruning and backpropagation

- **Meta-Learned Safety Adaptation**
  - Dual-loop learning across jailbreak task families
  - Robust generalization under distribution shift and adaptive attacks

- **Modular & Reproducible Design**
  - Hugging Face compatible
  - Supports FAISS / BM25 / hybrid retrieval
  - Clean separation between reasoning, retrieval, safety, and learning

## High-Level Architecture
<p align="center">
  <img src="assets/RAI-RAG Architecture.png" width="95%">
</p>

RAI-RAG introduces a **safety-first RAG pipeline** where retrieval becomes an auditable, optimizable decision governed by structured introspection and safety policies. RAI-RAG is a **safety-first Retrieval-Augmented Generation (RAG)** system that makes *evidence access* an explicit, auditable, and optimizable part of alignment. Unlike standard RAG pipelines that perform retrieval unconditionally, RAI-RAG treats retrieval as a **safety-critical action** that must be justified by **structured introspective reasoning** before any external knowledge is fetched or used. At a high level, RAI-RAG decomposes end-to-end generation into **five stages** with clear interfaces, enabling modular research (swap retrievers/judges/search) while maintaining a consistent safety contract across the entire pipeline.

### 1) User Prompt Ingestion
**Input:** user prompt `x`  
The system first normalizes and logs the raw query (e.g., language detection, basic sanitation, and optional PII stripping depending on policy). At this stage, RAI-RAG does *not* retrieve any documents. The prompt is treated as an untrusted input that may contain role-play framing, forced prefixes, or jailbreak patterns.

**Output:** canonicalized prompt object:
- `x.text`
- `x.metadata` (timestamp, source, optional domain tag)

### 2) Structured Introspection (Risk & Intent Analysis)
RAI-RAG performs **structured introspective reasoning** to infer the user’s intent and assess safety risks *prior to retrieval*. The introspection process produces a step-wise trace (e.g., `<Reasoning_step> ... </Reasoning_step>`) that is strictly separated from the final output (`<Output> ... </Output>`). This separation ensures that:
- safety reasoning is explicit and inspectable,
- unsafe content is not “leaked” into the final response,
- downstream modules can condition behavior on validated reasoning states.

Typical introspective sub-tasks include:
- **Intent inference:** benign information seeking vs. harmful procedural request vs. disguised intent
- **Policy mapping:** which safety category/policy constraints apply
- **Ambiguity assessment:** whether intent is unclear or borderline, requiring conservative handling
- **Retrieval necessity analysis:** is external evidence required to answer safely and helpfully?

**Output:** introspection trace `y_IR` (validated and parsed), plus a structured state:
- `intent_hypothesis`
- `risk_category`
- `confidence` / `uncertainty`
- `retrieval_need` (provisional)

### 3) Risk-Aware Retrieval Decision (Retrieve / Restrict / No-Retrieve)
In standard RAG, retrieval is always executed. In RAI-RAG, retrieval is a **decision** produced by the introspection policy. Concretely, the system chooses one of three actions:

- **Retrieve:** proceed with normal retrieval when risk is low and external evidence is needed.
- **Restrict Retrieval:** retrieve under constraints when risk is moderate or ambiguous, e.g.:
  - domain allowlist (trusted sources only),
  - time windows (avoid outdated or policy-sensitive historical content),
  - reduced `top_k`,
  - query rewriting to remove procedural/harmful intent triggers.
- **No-Retrieve:** skip retrieval entirely when:
  - the request is clearly unsafe (retrieval would amplify harm), or
  - the question can be safely answered from general knowledge without external documents.

This stage is the **central safety innovation**: RAI-RAG recognizes that retrieval can be an attack surface, and therefore must be controlled.

**Output:** retrieval plan `π_rag(x, y_IR)` containing:
- action ∈ {Retrieve, Restrict, No-Retrieve}
- retrieval constraints (if any)
- retrieval query (original or rewritten)
- expected evidence type (definitions, high-level overview, non-procedural references)

### 4) Safe Evidence Selection (Retrieval + Evidence Safety Filtering)
If retrieval is enabled, RAI-RAG retrieves a candidate set of documents `D = {d_i}` using a chosen retriever (FAISS/BM25/hybrid). It then applies **evidence-level safety alignment** to ensure that retrieved documents are safe to use.

This stage includes two complementary operations:

1. **Document Retrieval**
   - run the retrieval query under the selected constraints,
   - optionally rerank using a cross-encoder or lightweight reranker.

2. **Evidence Safety Scoring & Pruning**
   Each document is evaluated with an evidence safety function:
   - `S_e(d_i | x, y_IR)` → safety score (e.g., in `[-1, 1]`)
   - documents with negative score (policy-violating, procedural harm, extremist content, etc.) are filtered out
   - if too many documents are filtered, the system may:
     - fall back to restricted retrieval,
     - trigger query rewriting,
     - or switch to a safe high-level response without external evidence.

This prevents **grounded harm** (unsafe answers supported by retrieved text) and mitigates **retrieval-driven jailbreaks** (prompts designed to fetch harmful documents).

**Output:** safe evidence bundle `D_safe` plus provenance:
- kept docs (IDs, snippets, scores)
- filtered docs (reasons for filtering)
- audit trail for reproducibility

### 5) Introspective Answer Synthesis (Grounded, Safe, and Justified)
Finally, RAI-RAG synthesizes the response conditioned on:
- the validated introspection trace `y_IR`,
- the safe evidence `D_safe` (if available),
- the applicable policy constraints.

The generator can produce one of three outcomes depending on risk and evidence availability:

- **Safe grounded response:** uses only `D_safe`, avoids prohibited procedural details, and provides verifiable claims where possible.
- **High-level safe summary:** when the topic is sensitive or evidence is borderline, respond with non-actionable, general guidance.
- **Refusal (with safe alternatives):** when the user’s intent is clearly unsafe or policy-disallowed.

Optionally, RAI-RAG can apply **test-time selection** (best-of-`N`, beam search) guided by a **process reward model** that scores partial trajectories for safety/helpfulness consistency, selecting the safest high-quality candidate.

**Output:** final response `f` with optional:
- citations or snippet references,
- structured refusal template,
- safety justification (if enabled by setting)


## Execution Flow Summary

```text
(1) Prompt x
   ↓
(2) Structured Introspection → y_IR (intent, risk, constraints)
   ↓
(3) Retrieval Gate → {Retrieve | Restrict | No-Retrieve} + query plan
   ↓
(4) Retrieve docs D → Evidence Safety Filter → D_safe
   ↓
(5) Answer Synthesis (grounded in D_safe, policy-safe)
```

## 📁 Repository Structure

```text
rai_rag/
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ base.yaml
│  ├─ model_llama.yaml
│  ├─ model_qwen.yaml
│  ├─ rag.yaml
│  ├─ safety_judges.yaml
│  └─ training_meta.yaml
├─ scripts/
│  ├─ build_corpus.py
│  ├─ build_index.py
│  ├─ generate_dir.py
│  ├─ train_reward_model.py
│  ├─ train_meta.py
│  ├─ eval_all.py
│  └─ demo_chat.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ dir/              # structured introspection dataset
│  ├─ preference/       # pairwise preferences (optional)
│  └─ benchmarks/       # jailbreak + benign eval sets
├─ prompts/
│  ├─ introspection.jinja
│  ├─ retrieval_gate.jinja
│  ├─ evidence_filter.jinja
│  └─ refusal_template.jinja
├─ rai_rag/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ logging.py
│  ├─ types.py          # Prompt, Doc, Trace, Scores, etc.
│  │
│  ├─ models/
│  │  ├─ backbone.py     # HF model wrapper (LLaMA/Qwen)
│  │  ├─ adapters.py     # θᵣ / LoRA or prefix adapters
│  │  ├─ generation.py   # best-of-N / beam hooks
│  │  └─ tokenization.py
│  │
│  ├─ introspection/
│  │  ├─ formatter.py    # enforce <Reasoning_step> ... <Output> format
│  │  ├─ validators.py   # structural checks & refusal rules
│  │  ├─ planner.py      # step template selection (risk, retrieval, evidence)
│  │  └─ trace.py        # parse & normalize introspection traces
│  │
│  ├─ rag/
│  │  ├─ corpus.py       # document store interface
│  │  ├─ index.py        # FAISS/BM25/Hybrid index build & load
│  │  ├─ retriever.py    # retrieve(query) -> docs
│  │  ├─ query_rewrite.py# restricted retrieval query generation
│  │  ├─ gate.py         # {Retrieve, Restrict, NoRetrieve}
│  │  ├─ evidence_filter.py # doc scoring & pruning
│  │  └─ citations.py    # optional citation injection
│  │
│  ├─ safety/
│  │  ├─ policies.py     # safety categories & refusal policy
│  │  ├─ judges.py       # safety/helpfulness/introspection judges
│  │  ├─ doc_scorer.py   # S_e(d | x, z_k) evidence safety scoring
│  │  └─ calibrators.py  # optional uncertainty calibration
│  │
│  ├─ reward/
│  │  ├─ composite.py    # R(H,S,I)=F(S)H + S + λI
│  │  ├─ process_rm.py   # process reward model (Bradley–Terry)
│  │  └─ preferences.py  # preference dataset builder (optional DPO)
│  │
│  ├─ search/
│  │  ├─ node.py         # MCTS node = trace + retrieval action
│  │  ├─ simcts.py       # SI-MCTS core
│  │  ├─ rollout.py
│  │  └─ cache.py        # memoized judge scores
│  │
│  ├─ meta/
│  │  ├─ task_families.py # role-play, forced-prefix, narrative camouflage
│  │  ├─ sampler.py      # τ ~ T, x ~ τ
│  │  ├─ outer_loop.py   # meta-update θᵣ
│  │  └─ curriculum.py   # optional progressive hardening
│  │
│  ├─ pipeline/
│  │  ├─ rai_rag.py      # end-to-end inference pipeline
│  │  ├─ stage1_introspect.py
│  │  ├─ stage2_retrieve.py
│  │  ├─ stage3_search.py
│  │  └─ stage4_answer.py
│  │
│  ├─ eval/
│  │  ├─ metrics.py      # safety/helpfulness/evidence metrics (UER, etc.)
│  │  ├─ run_eval.py
│  │  ├─ attack_suites.py# retrieval-driven jailbreak attacks
│  │  └─ reports.py      # tables & plots export
│  │
│  └─ utils/
│     ├─ io.py
│     ├─ seed.py
│     ├─ parallel.py
│     └─ hashing.py
└─ tests/
   ├─ test_format.py
   ├─ test_gate.py
   ├─ test_doc_filter.py
   └─ test_simcts.py
```

## Installation

### Clone repo
```bash
git clone https://github.com/WarBees-AI/Risk-Aware-RAG/.git
cd rai-rag
```
### Install locally
```
pip install -e .
```


##  Author
**[Miraj Rahman](https://github.com/Miraj-Rahman-AI)**  
AI Researcher | Autonomous Agents | RAG Systems | Trustworthy AI



##  Support
If this project supports your research or learning,
please consider giving it a ⭐ on GitHub.

## ⚠️ License & Usage Restriction
© 2026 Mirage-AI. All rights reserved.

No permission is granted to use, modify, distribute, or reproduce this software in any form.

This repository is provided for **viewing purposes only**.

