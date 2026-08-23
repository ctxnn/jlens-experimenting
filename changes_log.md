# Change Log

All notable project changes are recorded here. The project uses semantic versioning for
its experiment contract:

- **MAJOR:** incompatible change to model/lens pairing, artifact schema, or scientific
  interpretation.
- **MINOR:** new experiment capability, validation gate, artifact, or substantial
  documentation.
- **PATCH:** correction that preserves the intended experiment design.

The repository is currently pre-1.0 because the official GPU experiment has not yet been
archived with committed numerical results.

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
