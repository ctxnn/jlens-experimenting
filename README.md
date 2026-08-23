# Official J-Lens experiment on Qwen

This repository deliberately starts by validating Anthropic's official
[`jlens`](https://github.com/anthropics/jacobian-lens) library rather than
reimplementing the algorithm from scratch.

## Run in Google Colab

Open [`notebooks/qwen_official_jlens.ipynb`](notebooks/qwen_official_jlens.ipynb), select
**Runtime → Change runtime type → GPU** (L4/A100 preferred), and run every cell. This
host has no CUDA device or supported Colab CLI, so the notebook is the reproducible GPU
handoff.

The notebook pins the reference library to commit
`581d398613e5602a5af361e1c34d3a92ea82ba8e`, loads the published 1,000-prompt lens for
`Qwen/Qwen3.5-4B` from `neuronpedia/jacobian-lens`, and performs inference only.

## Outputs

- A comparison of J-Lens, logit lens, and ordinary model top tokens.
- Anthropic's interactive position × layer slice using the official `compute_slice` and
  `build_page` functions.
- A shareable `artifacts/qwen_ascii_face_slice.html` visualization.
- An `artifacts/qwen_validation.json` report containing the readout position and its
  next-token semantics, final-output parity errors, and J-Lens/logit-lens top-10 overlap.

The slice uses prompt positions as columns and layers as rows. Clicking a cell pins its
top token and exposes full-vocabulary rank trajectories across positions and depth. This
preserves the paper's inspectable visual grammar instead of replacing it with a merely
decorative chart.

## Why this is a justified Qwen test

- **Published-artifact gate:** exact official code revision and n=1,000 Qwen lens.
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
