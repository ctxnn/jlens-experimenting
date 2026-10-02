# Qwen J-Lens Experiment: Setup, Execution, Evidence, and Progress

> **Experiment version:** 0.5.0<br>
> **Results snapshot:** 27 August 2026<br>
> **Documentation updated:** 2 October 2026<br>
> **Status:** the official-library validation, focused spider trace, 63-prompt
> descriptive study, and real-data visualization are complete. Causal interventions
> and independent lens fitting have not been performed.

## Technical summary

This experiment asks whether a concept that is implied by a prompt can become readable
inside Qwen before the model commits to the answer associated with that concept. The
motivating example is:

```text
Prompt:   The number of legs on the animal that spins webs is
Target:   spider
Answer:   8
Decoy:    ant
```

The experiment does not ask Qwen to print the target concept. It greedily completes the
prompt, then uses Anthropic's published Jacobian Lens (J-Lens) to score `spider`, `8`, and
`ant` at every one of the 31 supported source layers immediately before the first generated
qualifying token (which can be an option label or placeholder). A matched logit-lens
readout is computed from the same prompt, position, layers, model, tokenizer, final norm,
and vocabulary head.

The complete run contains 63 prompts rather than one demonstration: 51 hide the target
behind a description and 12 name it explicitly as calibration baselines. All prompts,
prefix mismatches, and single-token coverage gaps remain in the saved data.

The measured result is deliberately narrow:

- Qwen began `50 / 63` continuations with an expected answer.
- `46 / 63` rows were fully measurable with single-token surfaces for target, answer,
  and decoy.
- The target outranked its matched decoy at its best J-Lens layer in 48 of 50 evaluable
  rows; 13 were unavailable (`48 / 63` of the full suite).
- `7 / 63` rows satisfied the declared strict relay rule under J-Lens; the matched
  logit-lens control satisfied the same rule in `12 / 63` rows.
- The maximum difference between the adapter's final readout and a direct Qwen forward
  pass was `0.0625`, within the configured numerical tolerance.

This supports the descriptive claim that **Qwen sometimes shows a readable target-to-answer
ordering under this lens and metric**. It does not show that the decoded word is the model's
literal thought, that the target representation caused the answer, or that this pattern
generalizes to arbitrary prompts. The spider ordering also appears under logit lens;
these measurements do not establish that J-Lens uniquely reveals it or outperforms the
control.

## 1. What was built, in order

The project developed in four experimental stages. Each stage closed a different source
of ambiguity before the next stage expanded the claim.

1. **Official-library smoke test.** Run Anthropic's unmodified `jlens` interface with its
   published Qwen lens and compare its final output with Qwen's ordinary logits.
2. **Focused spider trace.** Measure the paper-style spider prompt across depth and retain
   the complete numerical trace plus an official interactive slice.
3. **Multi-prompt relay suite.** Replace the single example with 63 designed prompts,
   matched decoys, explicit baselines, fixed scoring rules, and retained failures.
4. **Real-data reader.** Convert the raw results into a compact, self-contained interactive
   HTML file without simulating, smoothing, or changing the measured ranks.

```mermaid
flowchart LR
    A[Pin official code, Qwen, tokenizer, and lens] --> B[Validate the official notebook]
    B --> C[Measure the exact spider prompt]
    C --> D[Run 63 prompts on a clean T4]
    D --> E[Archive raw per-layer results]
    E --> F[Build compact deterministic projection]
    F --> G[Inject real data into standalone HTML]
    G --> H[Run structural and browser QA]
```

The implementation is split accordingly:

| File | Responsibility |
|---|---|
| [`notebooks/qwen_official_jlens.ipynb`](notebooks/qwen_official_jlens.ipynb) | Official API and position-semantics smoke test. |
| [`experiments/qwen_spider_trace.py`](experiments/qwen_spider_trace.py) | Focused, exact-prompt numerical drill-down. |
| [`experiments/qwen_relay_suite.py`](experiments/qwen_relay_suite.py) | Full 63-prompt GPU experiment and raw archive. |
| [`experiments/build_relay_visualization.py`](experiments/build_relay_visualization.py) | Deterministic raw-to-compact transformation and HTML build. |
| [`tests/test_notebook.py`](tests/test_notebook.py) | Structural and evidence-contract regression tests. |

## 2. Pinned artifacts and observed runtime

The model and lens are treated as a matched experimental instrument. Mutable default
branches are not sufficient because a model update or a changed lens file could silently
change the result while preserving the same human-readable name.

The reference source, model, tokenizer, and lens are pinned. Runtime package versions
below are observed values, not an environment lock; transitive dependencies can change
on a fresh installation.

| Component | Exact value used |
|---|---|
| Reference J-Lens code | `anthropics/jacobian-lens@581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Model and tokenizer | `Qwen/Qwen3.5-4B` |
| Model revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Published lens repository | `neuronpedia/jacobian-lens` |
| Lens revision | `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` |
| Lens file | `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt` |
| Lens fit provenance | `qwen-n1000`; the published artifact reports `1,000` fitting prompts |
| Model dtype | `torch.bfloat16` |
| Execution device | `Tesla T4` |
| Observed PyTorch version | `2.11.0+cu128` |
| Observed Transformers version | `5.15.0` |
| Generated tokens per prompt | at most `24`, greedy decoding, no sampling |
| Readout position | `-1`, after the final supplied prefix token |
| Source layers | `31` fitted source layers, indices `0–30`; ordinary final-model output at index `31` is separate |

The script also downloads `config.json` with `hf_hub_download` at the requested model SHA,
extracts the resolved SHA from the local `snapshots/<sha>/...` path, and asserts equality.
This verifies the artifact resolver that actually served the configuration. The private
Transformers `_commit_hash` field is not used because it was `None` in a real run despite
loading by an exact revision.

In the executable suite these values are centralized as `MODEL_NAME`, `MODEL_REVISION`,
`LENS_REPO`, `LENS_FILE`, `LENS_REVISION`, and `READOUT_POSITION`, rather than being copied
into individual prompt loops.

There is one remaining provenance limitation: the published lens metadata names the model
but does not cryptographically attest the exact Hugging Face model SHA used during fitting.
The selected model revision predates the published fit and is the best reproducible pairing
available from the public metadata, but it is still a provenance inference.

## 3. The measurement question

For a prompt that implies an entity and asks for an associated fact, the experiment tracks
three declared concepts:

- **Target:** the latent entity implied by the description, such as `spider`.
- **Answer:** the expected first completion, such as `8`.
- **Decoy:** a related but wrong entity, such as `ant`.

The decoy prevents a weak interpretation in which any domain-related word counts as
evidence. A low target rank is more meaningful when the target also outranks a matched
alternative.

The two readout branches are:

```text
J-Lens(l,p)    = W_U · N(J_l · h_l,p)
LogitLens(l,p) = W_U · N(      h_l,p)
```

where `h_l,p` is the residual stream at layer `l` and position `p`, `J_l` is the published
mean-Jacobian transport for that layer, `N` is Qwen's final normalization, and `W_U` is
Qwen's output vocabulary projection. The logit-lens control removes only the Jacobian
transport; all other experimental inputs remain matched.

## 4. Why the readout position is `-1`

For causal language-model tokens `[t0, t1, ..., tn]`, logits at index `i` predict token
`t(i+1)`. Therefore:

```text
Supplied tokens:       ...    t(n-1)      tn
Logit index:                  -2          -1
Predicts:                      tn       first unseen token
```

Reading index `-2` would reconstruct the already supplied final prompt token. An API/direct
model parity test can still pass at `-2`, because both paths can agree while answering the
wrong experimental question. The experiment consequently centralizes
`READOUT_POSITION = -1`, applies it to J-Lens, logit lens, and direct-model logits, and has
regression tests that reject literal `-2` readouts.

For the multi-prompt run, Qwen first generates an unedited continuation. The code finds the
first generated token containing a Unicode word character or digit (`[\w\d]`) and
constructs a readout prefix that ends immediately before it. Word characters include
underscores. Any preceding generated tokens are retained, not just whitespace.
Position `-1` predicts that qualifying token; it is not guaranteed to be an answer token.
In five option-list continuations the token begins the `A` label; for `explicit-octopus`
it begins an underscore placeholder. The saved prefix is exposed in the viewer so these
cases can be inspected without silently changing the alignment protocol.

## 5. Prompt-suite design

The suite contains four domains that share the same two-step structure: infer an entity,
then retrieve a property of that entity.

| Family | Hidden descriptions | Explicit baselines | Total | Requested relation |
|---|---:|---:|---:|---|
| Animals | 15 | 3 | 18 | infer creature, then retrieve limb count |
| Geometry | 12 | 3 | 15 | infer shape, then retrieve side/face count |
| Chemistry | 12 | 3 | 15 | infer element, then retrieve atomic number |
| Geography | 12 | 3 | 15 | infer country, then retrieve capital |
| **Total** | **51** | **12** | **63** | |

Every case is declared in code with a stable ID and the same schema:

```python
{
    "id": "animal-spider",
    "family": "animals",
    "condition": "hidden",
    "prompt": "The number of legs on the animal that spins webs is",
    "target": "spider",
    "decoy": "ant",
    "expected": "8",
    "answer_aliases": ["8", "eight"],
}
```

The explicit cases name the target directly—for example, “The number of legs on a spider
is”—and act as calibration baselines. They are not a separate control model, and the suite
was not sampled from a natural prompt population. The results therefore describe these
63 designed cases rather than an unbiased estimate of Qwen behavior. Some nominally hidden
descriptions contain strong lexical or numeric clues (for example, “eight triangles”);
the condition label is not a guarantee of a purely implicit reasoning problem. The suite
was not publicly preregistered and has no independent held-out set.

## 6. Metric definitions

The metrics are defined before interpreting their observed values.

### 6.1 Single-token surface resolution

Vocabulary ranks apply to one token ID. For each target, answer alias, and decoy, the code
tries case and leading-space variants. It retains a variant only if Qwen's tokenizer maps
the complete surface to exactly one token. Duplicate token IDs are removed.

A row is **fully measurable** only if target, answer, and decoy each have at least one
single-token variant. This is a measurement restriction, not a judgment that a multi-token
concept is absent. Seventeen rows fail this gate and remain visible as not fully measurable.
Available target, answer, or decoy ranks remain usable. Target coverage is `51 / 63`,
answer coverage `59 / 63`, and decoy coverage `57 / 63`; missing components can overlap.

### 6.2 Vocabulary rank

For candidate token `c` with score `s_c`, rank is computed as:

```text
rank(c) = 1 + count(tokens v where score(v) > score(c))
```

Rank `1` is best. Tied scores receive the same competition rank because only strictly
greater scores are counted. When a concept has several valid surface tokens, the experiment
uses the variant with the lowest rank; log probability breaks a rank tie.

### 6.3 Expected-answer prefix match (archived `correct` flag)

Generation is deterministic: `do_sample=False` and at most 24 new tokens. A result is
**correct** when the normalized continuation begins with one of the expected aliases at a
word boundary. Later answers do not rescue a failed prefix match. This is not a factual
assessment of the complete continuation: a wrong answer, option list, placeholder, or
non-answer preamble can all fail. The historical field name `correct` and flags are
preserved; reader-facing labels explain the narrower measurement.

### 6.4 Target peak

The **target peak** is the layer where the best target surface has its lowest J-Lens rank.
If multiple layers tie, the earlier layer wins. Equivalent peaks are also stored for the
answer and for the target under the logit-lens control.

### 6.5 Answer commitment layer

The **answer commitment layer** is the first source layer where an expected-answer surface
reaches J-Lens rank `10` or better. The threshold is a declared operational definition;
it does not mean the model has irreversibly or causally committed.

### 6.6 Target specificity

A row is **target-specific** when the target's best rank is better than the matched decoy's
rank at the target-peak layer. This asks whether the peak discriminates the intended entity
from one related alternative. This requires target and decoy coverage, not answer coverage:
it is evaluable in 50 rows, passes in 48, fails in 2, and is unavailable in 13. This is why
48 target-specific rows can exceed the 46 fully measurable rows.

### 6.7 Strict relay

A row satisfies the **strict relay** rule only when all of the following are true:

1. The greedy continuation begins with an expected answer alias.
2. The target reaches rank `100` or better under J-Lens.
3. The answer reaches rank `10` or better at some source layer.
4. The target's best layer occurs strictly before the answer's first top-10 layer.

Formally:

```text
relay = correct
        AND min_layer_rank(target) <= 100
        AND first_layer(answer rank <= 10) exists
        AND argmin_layer(target rank) < first_layer(answer rank <= 10)
```

This rule turns the visual hypothesis into a reproducible filter. The thresholds are not
estimated confidence bounds, and no sensitivity sweep has yet been reported.

## 7. End-to-end execution, step by step

### Step 1 — Install the exact reference implementation

The experiment installs Anthropic's repository directly from the pinned commit:

```bash
python -m pip install -q \
  git+https://github.com/anthropics/jacobian-lens.git@581d398613e5602a5af361e1c34d3a92ea82ba8e
```

This avoids silently changing the adapter or lens logic between runs.

### Step 2 — Require a GPU and initialize the run

The script asserts `torch.cuda.is_available()`, creates the artifact directory, and enables
official J-Lens logging. The large Qwen checkpoint and all layer readouts are executed on
Colab rather than approximated locally.

### Step 3 — Load Qwen and the tokenizer from one immutable snapshot

`AutoModelForCausalLM` and `AutoTokenizer` receive the same model revision. The model is
loaded in bfloat16, placed on CUDA, and set to evaluation mode. The resolved snapshot SHA
is then checked as described above.

### Step 4 — Wrap Qwen and load the published lens

`jlens.from_hf(hf_model, tokenizer)` exposes the architecture-specific blocks, final norm,
and LM head. `JacobianLens.from_pretrained` downloads the exact lens file from the pinned
lens-repository revision. No lens fitting occurs in this repository.

### Step 5 — Resolve measurable concept tokens

For each prompt, the script resolves single-token variants for target, decoy, and every
expected-answer alias. The resolved surfaces and token IDs are stored in the raw row so
rank calculations can be audited later.

### Step 6 — Generate Qwen's continuation without editing it

The prompt is tokenized and Qwen generates greedily for at most 24 tokens. The raw
continuation is preserved, including wrong answers and later explanatory text. These are
raw completion prompts, not chat-template-formatted instruction conversations.

### Step 7 — Align the readout before the first content token

The first generated token containing a word character (including underscore) or digit is
identified. The readout prefix includes the original prompt and every generated token
before it, but not that token. The measured residual predicts that qualifying token,
which is sometimes an option label or placeholder rather than the answer.

### Step 8 — Compute J-Lens logits at all 31 layers

The script calls:

```python
lens.apply(
    model,
    readout_prefix,
    layers=layer_ids,
    positions=[-1],
)
```

This returns one vocabulary-logit vector per published source layer plus the adapter's
ordinary final-model readout.

### Step 9 — Compute the matched logit-lens control

The same call is repeated with `use_jacobian=False`. This holds the prompt, activation,
position, model, final norm, vocabulary head, and layer set fixed while replacing the
learned Jacobian transport with the identity path.

### Step 10 — Verify adapter/direct-model parity

The exact token IDs used by the lens adapter are asserted equal to the constructed prefix.
Qwen then performs a direct forward pass over those IDs. Direct logits at position `-1`
must satisfy:

```python
torch.allclose(direct, adapter_final, atol=0.25, rtol=0.01)
```

The maximum absolute error is stored per prompt. This validates the adapter's final
readout path; it does not validate the semantic interpretation of intermediate layers.

### Step 11 — Record ranks and top tokens

For every layer and both lens branches, the script stores the top 10 decoded tokens and
the full-vocabulary rank plus log probability of every resolved target, answer, and decoy
surface. Nothing is inferred from colors or chart geometry during data collection.

### Step 12 — Derive per-prompt summary fields

The code computes target peak, answer peak, logit-lens target peak, answer commitment,
target specificity, and strict relay from the saved rank rows. These fields are deterministic
functions of the raw measurements.

### Step 13 — Checkpoint after every prompt

After each item, a partial JSON archive records completed and planned prompt counts. This
limits data loss if the Colab session disconnects. Large tensors are deleted and the CUDA
cache is cleared before the next prompt.

### Step 14 — Write the final evidence archive

After all prompts complete, the script adds immutable revisions, package versions, GPU,
readout semantics, thresholds, aggregate counts, and every raw prompt row to
[`artifacts/qwen_relay_suite.json`](artifacts/qwen_relay_suite.json).

### Step 15 — Build the browser-ready projection

[`experiments/build_relay_visualization.py`](experiments/build_relay_visualization.py)
reads the raw archive and creates a compact record containing all prompt IDs and all 31
layer ranks while omitting repeated per-surface detail that the browser does not use. It
also retains alignment metadata and derives logit-lens summaries using the same thresholds. It
writes [`artifacts/qwen_relay_suite_summary.json`](artifacts/qwen_relay_suite_summary.json).

### Step 16 — Produce the self-contained visualization

The builder serializes the compact data and replaces the single
`__QWEN_RELAY_DATA__` marker in
[`visualizations/qwen-concept-relay.template.html`](visualizations/qwen-concept-relay.template.html).
The result is [`visualizations/qwen-concept-relay.html`](visualizations/qwen-concept-relay.html),
which contains the real measurements and requires no server. The same builder refreshes
the atlas's complete raw token lists from `qwen_validation.json`, using the notebook's
literal prompt declarations. Token normalization happens only when displaying them.

**Important:** the `.template.html` file is intentionally blank because its data marker has
not been replaced. It is build source, not the viewer. Open `qwen-concept-relay.html` to
see the populated experiment.

### Step 17 — Validate the archive and interface

Structural tests reconcile aggregate counts against individual rows, require 63 unique
IDs, require 31 layer records for every prompt, check immutable revisions and the non-causal
scope label, compare compact/raw prompt order, and confirm the built HTML contains no data
placeholder.

Browser QA then checks desktop and narrow layouts, prompt selection, J-Lens/logit-lens
switching, strict-relay and token-split filters, body overflow, and console/page errors.

## 8. Exact Colab reproduction

Run the suite from the repository root in a fresh authenticated Colab T4 session:

```bash
colab new -s jlens-relay-suite --gpu T4
colab exec -s jlens-relay-suite \
  -f experiments/qwen_relay_suite.py \
  --timeout 1800
colab download -s jlens-relay-suite \
  /content/artifacts/qwen_relay_suite.json \
  artifacts/qwen_relay_suite.json
colab stop -s jlens-relay-suite
```

Then rebuild the compact data and populated HTML locally:

```bash
python3 experiments/build_relay_visualization.py
python3 -m unittest discover -s tests -v
```

Open the generated file:

```text
visualizations/qwen-concept-relay.html
```

Do not open `visualizations/qwen-concept-relay.template.html` as the result. If the raw JSON
is regenerated, rerun the builder before inspecting the HTML so the embedded data matches
the latest archive.

## 9. What each output contains

| Artifact | Size / scope | Meaning |
|---|---:|---|
| `artifacts/qwen_relay_suite.json` | 10,970,524 bytes | Complete evidence: provenance, continuations, variants, top tokens, per-layer ranks, summaries, aggregates. |
| `artifacts/qwen_relay_suite_summary.json` | build-dependent | Deterministic compact projection used by the browser, including matched control and alignment metadata. |
| `visualizations/qwen-concept-relay.html` | build-dependent | Populated, self-contained interactive viewer. |
| `visualizations/qwen-concept-relay.template.html` | build-dependent | Source template with a data placeholder; not a populated viewer. |
| `artifacts/qwen_spider_trace.json` | focused drill-down | Full numerical trace for the exact spider prompt. |
| `artifacts/qwen_spider_slice.html` | focused drill-down | Official position-by-layer interactive slice for that prompt. |

The compact file is not a second experiment. It is a loss-limited projection of the raw
archive for rendering: each layer row keeps layer ID and target/answer/decoy rank under
both J-Lens and logit lens. The full surface variants and probabilities remain in the raw
JSON for audit.

In other words, the template is the source template with a data placeholder; the generated
HTML is the result that a reader should open.

## 10. Observed results

### 10.1 Overall outcomes

| Outcome | Count | Share of all 63 prompts |
|---|---:|---:|
| Matches expected-answer prefix | 50 | 79.4% |
| Fully single-token measurable | 46 | 73.0% |
| Target-specific at target peak | 48 | 76.2% |
| Strict relay | 7 | 11.1% |
| Fails expected-prefix rule | 13 | 20.6% |
| Token-split / not fully measurable | 17 | 27.0% |

Correctness, measurability, target specificity, and relay are overlapping properties;
these rows should not be added together. Specificity passes in 48 of 50 evaluable rows
(96.0%); the table's 76.2% uses all 63 prompts, including 13 unavailable rows. The 13
prefix failures are not all factual errors: several begin option lists, and
`explicit-octopus` eventually says “eight” after a placeholder and preamble. The declared
prefix metric deliberately does not score that later answer.

### 10.2 Results by family

| Family | Total | Prefix match | Fully measurable | Target-specific | Strict J-Lens relay |
|---|---:|---:|---:|---:|---:|
| Animals | 18 | 17 | 14 | 13 | 5 |
| Geometry | 15 | 13 | 6 | 6 | 0 |
| Chemistry | 15 | 9 | 11 | 15 | 0 |
| Geography | 15 | 11 | 15 | 14 | 2 |

The family table exposes important asymmetries. Geometry loses many rows at the
single-token gate because several shape names split under the tokenizer. Chemistry has
target-specific peaks but more prefix mismatches and no strict relay cases under
the declared ordering rule. The study was not powered for inferential comparisons between
families, so these are diagnostics, not claims of statistically different rates.

### 10.3 Hidden descriptions versus explicit baselines

| Condition | Total | Prefix match | Fully measurable | Target-specific | Strict J-Lens relay |
|---|---:|---:|---:|---:|---:|
| Hidden description | 51 | 40 | 37 | 38 | 7 |
| Explicit target | 12 | 10 | 9 | 10 | 0 |

The absence of strict-relay cases in the explicit condition is not automatically a
failure. Because the target word is already present in the prompt, the timing hypothesis
is different and prompt copying can dominate target rank. These baselines calibrate the
reader, but they do not isolate a causal effect of hiding the entity.

### 10.4 The seven strict-relay cases

| Prompt ID | Target → answer | Target peak | First answer top-10 layer |
|---|---|---:|---:|
| `animal-spider` | spider → 8 | layer 24, rank 14 | 29 |
| `animal-spider-web-catcher` | spider → 8 | layer 24, rank 14 | 29 |
| `animal-spider-silk` | spider → 8 | layer 24, rank 9 | 28 |
| `animal-spider-corners` | spider → 8 | layer 23, rank 2 | 27 |
| `animal-butterfly` | butterfly → 6 | layer 24, rank 67 | 27 |
| `capital-france` | France → Paris | layer 11, rank 11 | 24 |
| `capital-italy` | Italy → Rome | layer 24, rank 31 | 27 |

The repeated spider variants show that the original observation is not restricted to one
exact wording. However, four closely related phrasings are not four independent conceptual
domains, and the complete suite shows that the same ordering is far from universal.

### 10.5 Matched logit-lens comparison

The same prefix flag, target-rank threshold, answer-rank threshold, and strict ordering
rule were applied to the archived logit-lens ranks, without rerunning the model:

| Condition | Total | J-Lens relays | Logit-lens relays |
|---|---:|---:|---:|
| Hidden description | 51 | 7 | 6 |
| Explicit target | 12 | 0 | 6 |
| All prompts | 63 | 7 | 12 |

J-Lens has a lower best target rank in 41 of 51 target-measurable rows, ties in 8, and a
higher rank in 2. That is a different diagnostic from relay count, not a reason to omit
the less favorable control result. Neither statistic establishes general superiority.

## 11. Worked example: spider → 8

For `animal-spider`, Qwen's unedited continuation began:

```text
 8.

This conclusion is based on the following logical steps:
...
```

The readout prefix ended in the generated leading space, before token `8`. Under J-Lens:

- `spider` reached its best rank of `14` at layer `24`;
- the matched `ant` decoy ranked worse at that target peak, so the peak was specific;
- `8` first entered the top 10 at layer `29` and reached rank `1` there; and
- the strict ordering condition therefore passed: target peak at 24, answer commitment
  at 29.

The raw row also stores several valid surfaces—such as ` spider`, `Spider`, and ` Spider`
for the target—rather than choosing a spelling after looking at the chart. The displayed
line is the best rank among those declared single-token surfaces at each layer.

Under the matched logit lens, `spider` peaks at rank `11` at layer `24`, and the answer
first enters the top 10 at layer `29`. This branch also passes the strict relay rule.
The example illustrates a real ordering in both readouts, not a uniquely J-Lens effect.

This is the cleanest intuitive example of the visualization, but the aggregate claim is
based on the full 63-row archive, including the 56 rows that do not meet the strict relay
rule.

## 12. Validation and robustness checks completed

| Validation gate | Implementation | What it establishes | What it does not establish |
|---|---|---|---|
| Immutable code | Full J-Lens Git commit | Fixed reference-source snapshot | A fully locked runtime or correctness of the theory |
| Immutable model/tokenizer | Same full HF SHA plus resolved-snapshot assertion | No default-branch drift | Exact fitter checkpoint attestation |
| Immutable lens | Full lens-repository SHA and exact file | Repeatable published artifact | Lens generalization |
| Position contract | Shared `-1` readout plus negative regression checks | First qualifying generated token is targeted | That the token is a semantic answer |
| Adapter parity | Direct Qwen vs adapter final logits | Adapter/readout agreement | Intermediate semantic truth |
| Matched baseline | `use_jacobian=False` | Fair J-Lens/logit-lens comparison | Causal advantage |
| Complete archive | 63 unique IDs, 31 layers each | No rows silently dropped | Representative sampling |
| Aggregate reconciliation | Counts recomputed from raw rows | Summary matches evidence rows | Statistical significance |
| Raw/compact reconciliation | Every projected layer rank and embedded record checked | Viewer data traces to raw archive | That archived ranks can be reconstructed without logits |
| Browser QA | Desktop and 390 px interaction checks | Core UI and filters functioned | Every browser/device combination |

At the original visualization release, all 19 automated tests passed; the later report
revision brought the suite to 20. Historical browser checks exercised prompt selection, branch switching, and filters at desktop and
mobile widths. The strict-relay filter returned 7 rows, the token-split filter returned
17, body width did not overflow, and no console or page errors were observed. Those
observations describe the original release, not a fresh browser verification of the
2 October proofreading changes.

The current local suite contains 24 tests, including four value-level fidelity checks.

The 2 October review independently reconciled every saved row summary, prefix flag,
coverage flag, projected rank, and embedded relay record. It restored complete raw token
lists in the atlas, added the matched control summaries, and checked both viewers'
JavaScript syntax and data bindings. The raw GPU archives were not modified or rerun.
Current rendered inspection was blocked by the connected browser's `file://` policy;
desktop/mobile interaction QA of the revised viewers remains a release follow-up.

The maximum adapter/direct-model parity error was `0.0625`, observed on
`element-silver`. The assertion tolerance is `atol=0.25`, `rtol=0.01` to accommodate
bfloat16 and adapter-path numerical differences. This threshold is an engineering gate,
not a statistical uncertainty interval.

## 13. What is complete and what remains open

### Complete

- Official J-Lens runs with the matched published Qwen artifact.
- Model, tokenizer, lens, and reference J-Lens source revisions are pinned. Torch and
  Transformers versions are recorded, not locked; transitive dependency locking remains open.
- The next-unseen-token position error is corrected and regression-tested.
- The official five-family notebook smoke test is archived.
- The exact spider trace and official interactive slice are archived.
- All 63 multi-prompt rows and 31 layers per row are archived.
- Prefix mismatches and partial single-token coverage cases are retained.
- A deterministic raw-to-visualization build path exists.
- The original populated visualization passed desktop/mobile interaction QA; the revised
  viewer still needs fresh rendered QA as described above.
- The work is committed on the `work` branch and included in PR #2.

### Not yet established

- No activation patching, coordinate swap, ablation, or other causal intervention has
  tested whether the target representation changes the answer.
- No independent Jacobian fit has reproduced Anthropic's fitting pipeline.
- No multi-token sequence-level metric handles concepts that fail the single-token gate.
- No held-out or externally sampled prompt distribution supports population estimates.
- No confidence intervals, bootstrap analysis, or threshold sensitivity analysis has
  been reported.
- No comparison across random seeds is meaningful yet because generation is greedy and
  deterministic; perturbation or sampling studies would require a new protocol.
- The official lens metadata still does not attest the exact model revision used to fit
  the artifact.

These are scientific boundaries, not missing UI polish. A blog can accurately present the
current work as a real, reproducible, multi-prompt descriptive readout study. It should not
present it as proof that the model thinks in English words or as a causal explanation of
why Qwen answered correctly.

The machine-readable archive expresses the same boundary as
`readout-only; no causal interventions`.

## 14. Recommended next experiments

1. **Threshold sensitivity.** Recompute relay counts over a grid of target-rank and
   answer-rank thresholds without rerunning the model.
2. **Sequence-level concept scoring.** Add a declared method for multi-token targets so
   tokenizer coverage does not dominate geometry and other families.
3. **Held-out prompt protocol.** Generate and freeze prompts independently of observed
   J-Lens traces, then report the complete held-out set.
4. **Copying diagnostic.** Separate targets already present in the prompt from genuinely
   inferred targets and quantify direct token-copy effects.
5. **Causal intervention.** Patch or replace the target-relevant representation and test
   whether answer probabilities move in the predicted direction.
6. **Independent fit.** Reproduce the Jacobian estimation pipeline, verify batching and
   finite-difference behavior, and compare fitted lenses across corpus sizes.
7. **Environment manifest.** Save package locks and CUDA/runtime metadata beside future
   archives for stronger long-term reproducibility.

## 15. Reader checklist for the blog release

Before citing a visual or number, verify:

- the open file is `visualizations/qwen-concept-relay.html`, not the template;
- the selected prompt ID matches the claim in the text;
- the readout branch is visibly labeled J-Lens or logit lens;
- rank direction is explained—smaller is better;
- prefix mismatches are not automatically described as factual errors, and partial
  coverage is not described as absence of a concept;
- a strict relay is described as satisfying declared thresholds, not as a discovered
  causal chain; and
- every general statement includes the `7 / 63` J-Lens and `12 / 63` logit-lens result,
  not only the spider example.

The conceptual background and equations are expanded in
[`understanding.md`](understanding.md). Versioned changes and earlier correctness fixes are
recorded in [`changes_log.md`](changes_log.md).
