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
- An `artifacts/qwen_validation.json` report containing final-output parity errors and
  J-Lens/logit-lens top-10 overlap.

The slice uses prompt positions as columns and layers as rows. Clicking a cell pins its
top token and exposes full-vocabulary rank trajectories across positions and depth. This
preserves the paper's inspectable visual grammar instead of replacing it with a merely
decorative chart.

## Why this is a justified Qwen test

- **Published-artifact gate:** exact official code revision and n=1,000 Qwen lens.
- **Parity gate:** the official API's final output must agree with a direct Qwen forward
  pass at the same token position.
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
