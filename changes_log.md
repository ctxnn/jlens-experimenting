# Change Log

All notable project changes are recorded here. The project uses semantic versioning for
its experiment contract:

- **MAJOR:** incompatible change to model/lens pairing, artifact schema, or scientific
  interpretation.
- **MINOR:** new experiment capability, validation gate, artifact, or substantial
  documentation.
- **PATCH:** correction that preserves the intended experiment design.

The repository is currently pre-1.0 because its expanded prompt suite remains a designed
descriptive readout study, not a causal intervention benchmark or independent lens fit.

## [0.5.0] — 2026-08-27

### Added

- Added `experiments/qwen_relay_suite.py`, a pinned 63-prompt GPU evaluation with 51
  hidden-description cases, 12 explicit-entity baselines, four domains, matched decoys,
  greedy continuations, and per-layer J-Lens/logit-lens ranks.
- Added the focused exact-prompt `qwen_spider_trace.py` experiment, numerical trace, and
  official interactive slice as a reproducible drill-down beside the broader suite.
- Added the complete `artifacts/qwen_relay_suite.json` evidence archive and deterministic
  compact `artifacts/qwen_relay_suite_summary.json` projection.
- Added `visualizations/qwen-concept-relay.html`, a self-contained real-data reader with
  selectable 31-layer traces, J-Lens/logit-lens switching, correctness/tokenization
  states, and an all-prompt relay map. Its source template and deterministic builder are
  included for reproducibility.
- Added regression coverage for immutable revisions, complete prompt retention, raw-to-
  compact reconciliation, observed aggregate counts, and scientific caveats.

### Observed

- Completed all `63 / 63` prompts on a Tesla T4 using the pinned Qwen and lens revisions.
- Observed `50 / 63` correct greedy continuations, `46 / 63` fully single-token measurable
  rows, `48 / 63` target-specific peaks, and `7 / 63` strict relay-pattern cases.
- Recorded maximum adapter/direct-model parity error `0.0625`.
- Retained incorrect answers and 17 token-split rows instead of filtering the visual to
  favorable examples.

### Verified

- Reconciled aggregate counts independently against raw result rows; all prompt IDs were
  unique and every prompt contained 31 layer records.
- Rendered and visually inspected the standalone HTML at desktop and mobile widths.
- Exercised prompt selection, J-Lens/logit-lens switching, strict-relay filtering, and
  token-split filtering at 1440 px and 390 px with no browser console errors or body
  overflow. The relay and token-split filters returned 7 and 17 rows respectively.

### Interpretation boundary

The strict relay rule is descriptive: correct generation, target rank at most 100, and
target peak before the expected answer first enters the top 10. It is not a causal
spider-to-ant intervention or a literal transcript of model thought.

## [0.4.0] — 2026-08-24

### Added

- Archived the verified Iteration 1 JSON report, official interactive slice, and executed
  output notebook in source control.
- Added `visualizations/jlens-concept-atlas.html`, a self-contained three-view explainer:
  Jacobian transport versus the identity control, semantic emergence through depth, and
  the causal `-1` readout horizon.
- Grounded the atlas in the committed Qwen run and labeled top-10 overlap as set agreement,
  never as confidence, accuracy, or causal proof.
- Added documentation and structural coverage for the visualization contract.

### Verified

- Manual Chrome QA exercised three independent position/layer selections in the official
  slice. Matrix highlights, selected tokens, pins, heatmap markers, and both rank charts
  remained synchronized.
- The concept atlas parses without JavaScript syntax errors and is checked at desktop and
  mobile viewport sizes.

## [0.3.3] — 2026-08-23

### Fixed

- Replaced the runtime assertion on Transformers' private
  `hf_model.config._commit_hash` field with a check of the immutable Hugging Face cache
  path returned by `hf_hub_download`.
- Recorded `resolved_model_revision` separately from the requested `model_revision` in
  `artifacts/qwen_validation.json`.
- Added regression coverage that rejects `_commit_hash` and requires the resolved
  snapshot SHA check.
- Added stable, unique IDs to every notebook cell and regression coverage for them,
  removing the `MissingIDFieldWarning` emitted by the real Colab CLI run and preventing
  a future `nbformat` validation failure.
- Replaced the stale browser-only Colab handoff with the tested CLI runbook, including
  L4-to-T4 fallback, clean-kernel reruns, runtime cleanup, and an explicit audit for error
  outputs because CLI notebook execution can return exit code 0 after a cell exception.
- Added documentation regression coverage for the CLI command and error audit.
- Ignored the CLI-generated `notebooks/*_output.ipynb` files so validation evidence can
  remain local without polluting source-control status.
- Documented the Iteration 1 T4 run: five prompt families, zero output-notebook error
  cells, matching requested/resolved model revisions, maximum parity error `0.03125`, and
  both expected artifacts generated.
- Kept interactive browser rendering explicitly open after local URL security blocked it.

### Why this fix matters

The first real Colab CLI run loaded the pinned model successfully, but Qwen's resolved
configuration exposed `_commit_hash = None`. The old assertion therefore failed because
it depended on private Transformers metadata, not because the requested snapshot drifted.
The Hugging Face cache path includes `snapshots/<resolved-sha>/...`, so checking that value
verifies the pin using the artifact resolver that actually served the configuration.

## [0.3.2] — 2026-08-23

### Fixed

- Replaced mutable lens revision `qwen-n1000` with immutable repository commit
  `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` for `JacobianLens.from_pretrained`.
- Retained `qwen-n1000` separately as `LENS_SOURCE_REF` for human-readable provenance.
- Recorded both `lens_revision` (immutable SHA) and `lens_source_ref` (descriptive ref) in
  `artifacts/qwen_validation.json`.
- Printed the immutable lens revision during notebook setup.
- Added structural regression coverage that rejects using the mutable ref as
  `LENS_REVISION`.

### Why this fix matters

A Hugging Face branch can move to different lens weights without changing its name. If
`qwen-n1000` moved, the old report would still claim the same revision and model parity
would remain blind to the changed Jacobian. Downloading by full repository commit makes
the lens artifact snapshot reproducible while preserving the descriptive branch name as
non-authoritative metadata.

## [0.3.1] — 2026-08-23

### Fixed

- Added `MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"`.
- Passed that immutable revision to both `AutoModelForCausalLM.from_pretrained` and
  `AutoTokenizer.from_pretrained`.
- Added a runtime assertion that the resolved model configuration commit matches the pin.
- Recorded `model_revision` in `artifacts/qwen_validation.json`.
- Documented the provenance limitation: the lens fit metadata names the model but omits
  its revision, so the selected SHA is inferred from the model repository history and the
  fit timestamp rather than attested inside the lens artifact.

### Why this fix matters

A mutable default Hugging Face revision could change model or tokenizer files while the
published Jacobian remains fixed. Final-logit parity would still pass because both parity
branches would use the same changed model. Pinning and reporting the full model SHA makes
the run reproducible and prevents that silent model–lens mismatch.

## [0.3.0] — 2026-08-23

### Added

- `understanding.md`, a full conceptual and code-level walkthrough containing:
  - J-Lens, logit-lens, and model-readout equations;
  - end-to-end and validation-flow Mermaid graphs;
  - a causal-position sequence diagram;
  - cell-by-cell notebook behavior;
  - artifact schema and visualization interpretation;
  - proof boundaries for every validation gate;
  - failure modes, runbook, repository map, and research roadmap.
- This `changes_log.md` with semantic-version policy and a chronological record of fixes.
- README links to both long-form documents.
- Structural tests that require the documentation version, causal-position explanation,
  architecture diagram, artifact documentation, and change-log entries to remain present.

### Documentation

- Distinguished applying a published lens from fitting or reimplementing one.
- Explained why a matched model checkpoint and Jacobian artifact are inseparable.
- Documented top-10 overlap semantics and warned that difference does not imply quality.
- Documented the limits of parity, qualitative prompt coverage, and descriptive readout.

## [0.2.0] — 2026-08-23

### Fixed

- Changed J-Lens evaluation from `positions=[-2]` to a shared
  `READOUT_POSITION = -1`.
- Changed logit-lens evaluation to the same `READOUT_POSITION = -1`.
- Changed direct Hugging Face logits from index `-2` to the same shared position.
- Recorded the readout position and next-token meaning in the JSON output.
- Printed the last supplied token and explicitly labeled it as predicting the next unseen
  token.
- Added regression assertions rejecting `positions=[-2]` and `logits[0, -2]`.

### Why this fix matters

Causal logits at position `p` predict token `p + 1`. For completion prompts, `-2`
predicts the already supplied final token (usually `is` or `the`), while `-1` predicts
the first unseen continuation. Because all branches previously shared `-2`, parity still
passed even though the scientific comparison targeted teacher-forced reconstruction.
This was a validity bug, not a cosmetic indexing change.

## [0.1.0] — 2026-08-23

### Added

- Colab-ready official J-Lens × Qwen notebook.
- Immutable pin to Anthropic reference commit
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.
- Matched `Qwen/Qwen3.5-4B` model and published n=1,000 lens loading.
- J-Lens versus logit-lens top-token comparison across five prompt families.
- Official interactive position × layer HTML slice.
- JSON validation artifact with model parity and top-10 overlap.
- Structural notebook tests and generated-artifact ignore rules.

### Known limitations at release

- No numerical GPU results are committed.
- Prompt families contain one hand-written example each.
- Tests are structural and do not download model weights.
- No new Jacobian fitting or causal intervention is performed.
