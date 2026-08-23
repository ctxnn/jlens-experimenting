# Official J-Lens experiment on Qwen

This repository deliberately starts by validating Anthropic's official
[`jlens`](https://github.com/anthropics/jacobian-lens) library rather than
reimplementing the algorithm from scratch.

For the complete conceptual model, code walkthrough, diagrams, artifact schema, and
interpretation guide, read [`understanding.md`](understanding.md). Version history and
scientific fixes are recorded in [`changes_log.md`](changes_log.md).

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

The current CLI can save an output notebook and return exit code 0 even when an individual
cell contains an exception. Audit the saved notebook explicitly before treating the run as
successful:

```bash
jq '[.cells[].outputs[]? | select(.output_type == "error")] | length' \
  notebooks/qwen_official_jlens_output.ipynb
```

A passing run prints `0`. Release the runtime when validation is complete:

```bash
colab stop -s jlens-qwen
```

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
  avoid relying on one cherry-picked example.
- **Rank gate:** top-k overlap and full-vocabulary ranks supplement decoded top-1 text.
- **Interpretation gate:** readable tokens are descriptive evidence, not proof of causal
  mediation.

`Qwen/Qwen3.5-4B` is the correct initial target because Anthropic publishes a matched
n=1,000 lens for it. A smaller Qwen checkpoint would require a new architecture-specific
fit and would confound library validation with fitting quality.

## Local structural checks

```bash
python -m unittest discover -s tests -v
python -m json.tool notebooks/qwen_official_jlens.ipynb >/dev/null
```

Numerical model execution and rendering happen in the GPU notebook.

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
