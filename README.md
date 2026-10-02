# Official J-Lens experiment on Qwen

This repository deliberately starts by validating Anthropic's official
[`jlens`](https://github.com/anthropics/jacobian-lens) library rather than
reimplementing the algorithm from scratch.

For the new real-data graphics and paper-inspired Qwen intervention pilot, open
[`visualizations/qwen-iteration2.html`](visualizations/qwen-iteration2.html) and read
[`iteration2.md`](iteration2.md). Iteration 2 keeps hidden-description prompts separate
from named-entity calibrations, shows matched controls and actual continuations, and
exports blog-ready SVGs. It does not train Qwen or refit the published lens.
The complete Iteration 2 run contains **40 real prompts and 80 SVG figures**; browse the
[figure gallery](visualizations/iteration2/index.html) for individual exports.

For the complete chronological setup, measurement definitions, execution steps, observed
results, and Iteration 1 progress, read [`experiment.md`](experiment.md). For the Iteration 1
conceptual model, code walkthrough, diagrams, artifact schema, and interpretation guide, read
[`understanding.md`](understanding.md). Version history and scientific fixes are recorded
in [`changes_log.md`](changes_log.md).

## Publication status

The measured data supports an exploratory explainer; it is not a literal thought trace,
a general causal-mediation benchmark, or a replication of the paper's Claude intervention.
The corrected intervention exports include next-token alias-mass qualifications and the
actual saved continuations. Their bars and text use different intervention scopes, labelled
in every figure. All measured examples remain available through the
[visualization index](visualizations/index.html).

Automated evidence and selector checks are not a substitute for browser QA. The connected
browser blocks these local-file pages, so final interactive desktop/mobile rendering and
accessibility acceptance remain pending. See the [publication gate](iteration2.md#publication-gate).
For bundled official inspector code and standalone-file distribution, retain the
[third-party notices and license texts](THIRD_PARTY_NOTICES.md).

## Run in Google Colab

Open [`notebooks/qwen_official_jlens.ipynb`](notebooks/qwen_official_jlens.ipynb), select
**Runtime → Change runtime type → GPU** (L4/A100 preferred), and run every cell.

For a headless run with the authenticated Google Colab CLI:

```bash
colab new -s jlens-qwen --gpu L4
colab exec -s jlens-qwen -f notebooks/qwen_official_jlens.ipynb --timeout 1800
```

If the account has no L4 entitlement, use `--gpu T4`. Start from a fresh kernel when
re-running the notebook because a previous 4B model instance can retain enough GPU memory
to make the next load fail:

```bash
colab restart-kernel -s jlens-qwen
```

In the observed CLI 0.6.0 run, the command could save an output notebook and return exit
code 0 even when an individual cell contained an exception. Audit the saved notebook
explicitly before treating any run as successful:

```bash
jq '[.cells[].outputs[]? | select(.output_type == "error")] | length' \
  notebooks/qwen_official_jlens_output.ipynb
```

A passing run prints `0`. Release the runtime when validation is complete:

```bash
colab stop -s jlens-qwen
```

Run the 63-prompt concept-relay suite in a separate clean T4 session:

```bash
colab new -s jlens-relay-suite --gpu T4
colab exec -s jlens-relay-suite -f experiments/qwen_relay_suite.py --timeout 1800
colab download -s jlens-relay-suite \
  /content/artifacts/qwen_relay_suite.json artifacts/qwen_relay_suite.json
colab stop -s jlens-relay-suite
python3 experiments/build_relay_visualization.py
```

The builder produces a compact dataset, a self-contained relay visualization, and the
atlas's complete token lists from the archived five-prompt validation.
It does not rerun the model or alter the measured ranks.

The notebook pins the reference library to commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e` and pins both model and tokenizer files to
Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. It loads the published
1,000-prompt lens for `Qwen/Qwen3.5-4B` from `neuronpedia/jacobian-lens` at immutable
repository commit `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` and performs inference only. The
human-readable `qwen-n1000` ref is retained only as provenance metadata, never as the
download revision. Immutable model and lens revisions prevent either mutable ref from
silently invalidating the pairing.

## Outputs

- A comparison of J-Lens, logit lens, and ordinary model top tokens.
- Anthropic's interactive position × layer slice using the official `compute_slice` and
  `build_page` functions.
- A shareable `artifacts/qwen_ascii_face_slice.html` visualization.
- An `artifacts/qwen_validation.json` report containing the readout position and its
  next-token semantics, final-output parity errors, and J-Lens/logit-lens top-10 overlap.
- A purpose-built [`visualizations/jlens-concept-atlas.html`](visualizations/jlens-concept-atlas.html)
  with three linked explanations: the Jacobian transport, semantic emergence through
  depth, and the causal readout horizon.
- A real-data [`visualizations/qwen-concept-relay.html`](visualizations/qwen-concept-relay.html)
  spanning 63 prompts, with per-layer target/answer/decoy traces, J-Lens and logit-lens
  controls, answer-prefix matching states, partial token coverage, and an all-prompt
  J-Lens relay map.
- The complete `artifacts/qwen_relay_suite.json` archive and compact
  `artifacts/qwen_relay_suite_summary.json` used by the standalone visualization.
- A focused `artifacts/qwen_spider_trace.json` plus official
  `artifacts/qwen_spider_slice.html` drill-down for the exact paper prompt; both are
  explicitly readout-only and are superseded by the 63-prompt suite for general claims.

The slice uses prompt positions as columns and layers as rows. Clicking a cell pins its
top token and exposes full-vocabulary rank trajectories across positions and depth. This
preserves the paper's inspectable visual grammar instead of replacing it with a merely
decorative chart.

## Iteration 1 verification

A clean Google Colab CLI 0.6.0 run completed on a Tesla T4 on 2026-08-23:

- all five prompt families completed;
- the requested and resolved Qwen model revisions both equaled
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- the published lens loaded from immutable revision
  `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a`;
- the maximum final-readout parity error was `0.03125`;
- the saved output notebook contained zero error cells; and
- both the JSON report and 1,823,422-byte interactive HTML artifact were generated.

The HTML title and expected interactive markup were validated statically. Manual Chrome
QA then exercised three distinct position/layer selections and confirmed that the matrix
highlight, selected token, pin list, heatmap marker, and both rank charts remained
synchronized. The JSON report, official interactive slice, and executed output notebook
are committed as the Iteration 1 result archive despite the normal ignore rules for future
generated runs.

The concept atlas is intentionally explanatory rather than a replacement for the official
slice. Its transport view keeps the J-Lens and matched logit-lens paths separate; its
emergence braid computes overlap against the ordinary model's final top-10; and its causal
horizon makes the `-1` versus `-2` indexing distinction visible. Control characters are
normalized only for display, and the atlas labels overlap as set agreement rather than
confidence or accuracy.

## Multi-prompt relay verification

A pinned Tesla T4 run on 2026-08-27 evaluated 63 completion prompts: 51 hidden-description
cases and 12 explicit-entity baselines across animals, geometry, chemistry, and geography.
No prompt was removed from the archived dataset or visualization.

- `50 / 63` greedy continuations began with the expected answer;
- `46 / 63` cases had single-token surfaces for target, decoy, and expected answer;
- the target outranked its matched decoy in 48 of 50 evaluable cases; 13 lacked target
  or decoy coverage (`48 / 63` of the full suite);
- `7 / 63` met the strict relay definition; and
- the maximum adapter/direct-model parity error was `0.0625`.

Strict relay means: the continuation matches an expected-answer prefix, the target
reaches rank `≤ 100`, and its best layer occurs before the expected answer first enters
the top 10. This is a declared
descriptive threshold, not a causal test. A prefix mismatch can reflect formatting or a
non-answer preamble, not necessarily a factual error; the complete continuations remain
in the archive. Missing single-token surfaces remain visible, without discarding the
concepts that can still be measured.

Applying the same strict rule to the archived logit-lens ranks yields `12 / 63` relays:
6 of 51 hidden-description cases and 6 of 12 explicit cases, compared with J-Lens's 7
hidden and 0 explicit cases. The exact spider example passes under both branches. This
designed suite does not establish J-Lens superiority or a uniquely J-Lens relay.

## Why this is a justified Qwen test

- **Published-artifact gate:** exact official code revision and n=1,000 Qwen lens.
- **Immutable-model gate:** the model and tokenizer use the same full Hugging Face commit
  SHA, that resolved model SHA is asserted at runtime, and it is recorded in the report.
- **Immutable-lens gate:** lens weights are downloaded by the lens repository's full commit
  SHA; the mutable `qwen-n1000` name is recorded separately only for human provenance.
- **Next-token-position gate:** causal logits at position `p` predict token `p + 1`. All
  completion readouts use position `-1`, the final supplied prompt token, so they predict
  the first unseen token. Position `-2` would merely reconstruct the known final prompt
  token and is explicitly rejected by the structural tests.
- **Parity gate:** the official API's output at `-1` must agree with a direct Qwen forward
  pass at `-1`. This verifies the adapter/readout path but is not, by itself, evidence that
  the evaluation position is meaningful.
- **Control gate:** `use_jacobian=False` gives the logit-lens baseline using the same
  model, tokenizer, activations, layers, and positions.
- **Prompt-family gate:** multihop, multilingual, typo, ordering, and ASCII-face prompts
  plus the separate 63-prompt relay suite avoid relying on one cherry-picked example.
- **Rank gate:** top-k overlap and full-vocabulary ranks supplement decoded top-1 text.
- **Interpretation gate:** readable tokens are descriptive evidence, not proof of causal
  mediation.

`Qwen/Qwen3.5-4B` is the correct initial target because Anthropic publishes a matched
n=1,000 lens for it. A smaller Qwen checkpoint would require a new architecture-specific
fit and would confound library validation with fitting quality.

## Local structural checks

Open the [visualization directory](visualizations/index.html) for all viewers and
the figure gallery. Their minimal UI preserves the archived measurements and chart scales.

```bash
python -m unittest discover -s tests -v
python -m json.tool notebooks/qwen_official_jlens.ipynb >/dev/null
node tests/test_iteration2_viewer.js
node tests/test_minimal_viewers.js
```

Numerical model execution happens in Colab. The standalone viewers are built locally from
the saved measurements; rebuilding them does not require a GPU.

To rebuild the Iteration 2 viewer and vector figures from its saved archives:

```bash
python3 experiments/build_iteration2_visualization.py
```

Open the generated `.html`, not the `.template.html` source. See `iteration2.md` for
the separate protocol, archived runtime versions, reproduction steps, and scientific
limits. Interactive browser-rendered QA for Iteration 2 is still pending; numerical,
source-fidelity, and DOM-fixture checks do not substitute for it.

## Reading the results correctly

For token IDs `[t0, t1, ..., tn]`, a causal language model's logit vector at index `i`
predicts `t(i+1)`. There is no supplied target after `tn`; therefore logits at index `-1`
represent the distribution for the first new completion token. The notebook prints the
last supplied token beside “predicts next unseen token” to make this convention visible
in every prompt result.

The parity assertion answers only “does the official adapter reproduce the model's
readout at the selected residual?” The separate position contract answers “did we select
the residual that predicts an unseen continuation?” Both checks are necessary: parity can
still pass when every path consistently uses the wrong position.
