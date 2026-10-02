# Iteration 2: read a concept, then test the edit

Open the generated **[Iteration 2 viewer](visualizations/qwen-iteration2.html)**,
not its `.template.html` source. It embeds real Qwen measurements and works offline.
Iteration 1's notebook and 63-prompt archive are unchanged; its viewers remain available
with the same minimal presentation used by Iteration 2.

## What we are testing

A lens trace asks whether a word naming a concept becomes readable in a residual stream.
A controlled representation edit asks whether changing a direction associated with that
word affects Qwen's output. Neither is a literal transcript of thoughts.

The motivating description-only prompt comes from the
[original paper](https://transformer-circuits.pub/2026/workspace/index.html):

```text
The number of legs on the animal that spins webs is
```

The target word is not supplied. Its explicit calibration names the animal directly.
Named-entity traces alone cannot establish hidden-concept inference. The viewer labels
the condition and offers a paired-version button.
Pairs share declared concepts/answers, not identical wording or token positions. Their
differences cannot be attributed solely to whether the target word is named.

This is **paper-inspired, not a replication of the paper's Claude intervention**. The
paper reports a clamped coordinate swap; this Qwen pilot uses a single-layer unclamped
swap. It omits calibrated clamping and the paper's full normalization-factor treatment.
There is no Qwen training, fine-tuning, or new lens fitting. The prompts are hand-designed;
all displayed numerical trajectories are archived model measurements, not fabricated
data or manually drawn results.

## Setup, one step at a time

1. Start from the existing checkout on the separate `iteration-2` branch. Write new
   artifacts and viewers rather than replacing Iteration 1.
2. Pin the same Qwen model, tokenizer, reference library, and published n=1,000 lens.
   Check resolved model/lens snapshot paths and record the actual runtime versions.
3. Freeze 20 explicit prompts in `experiments/iteration2_prompts.json`. Include
   paraphrases, reversed contrasts, and animals, capitals, chemistry, geometry, time,
   and calendar tasks. Each declares target, decoy, answer, alternative, and aliases.
4. After inspecting the explicit calibration, freeze 20 paired description-only prompts
   in `experiments/iteration2_hidden_prompts.json`, before their own execution. This
   follow-up is not preregistered, random, or held out from experiment design.
5. Append exactly one ASCII space to each prompt. Read the last token of this supplied
   prefix: its logits predict the first unseen token. No answer-dependent alignment is
   performed. The space is formatting, not a generated answer token.
6. Resolve declared single-token surfaces; deduplicate their IDs. Keep unavailable
   concepts/endpoints rather than dropping their prompts.
7. Record clean ordinary Qwen logits and a capped greedy continuation. Capture residuals
   and compute J-Lens and matched logit-lens readouts at every source layer.
8. At six chosen layers, also measure the target's J-Lens rank at every supplied token
   position. These are not attention weights or causal attribution scores.
9. At those layers, run a concept swap, no-op, unrelated swap, direct-answer swap, and a
   seeded norm-matched random edit. The text stays unchanged.
10. Record the resulting ordinary model probabilities, top tokens, edit norms, vector
    diagnostics, and the change in the declared answer margin.
11. At predeclared layer 24, record a sustained edited greedy continuation. This hook is
    reapplied to all positions on every decoding forward, with caching disabled.
12. Save after each completed case, retain pilot snapshots and execution-history
    notebooks, download the archives, and release the Colab sessions.
13. Validate archived values, hashes, and derived effects. Rebuild offline HTML and SVG
    figures deterministically; the visualization builder never calls a model.

The first allocated Colab session was lost before model execution. A replacement ran
the explicit three-case pilot and resumed the remaining cases in the same kernel after
parity/no-op checks. The description-only follow-up uses a separate session, frozen
protocol, output path, pilot cell, and resume cell. Its first session was lost before the
partial archive could be downloaded; the final attempt executes all 20 frozen cases as
one uninterrupted cell. Its numerical gates are checked on the completed archive, not
claimed as a pre-expansion pilot gate. The failed history is retained. Completion is
determined from archived records and checks, not the CLI's exit status alone.

## Frozen instrument

| Component | Value |
|---|---|
| Model/tokenizer | `Qwen/Qwen3.5-4B` |
| Model revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Official library revision | `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| Lens repository | `neuronpedia/jacobian-lens` |
| Lens revision | `16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a` |
| Lens file | `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt` |
| Readout layers | Post-block 0–30; final model block is 31 |
| Intervention layers | 8, 12, 16, 20, 24, 28 |
| Strength | 1, unclamped |
| Greedy edited generation | Layer 24; at most 12 tokens; no chat template |

The explicit pass used Tesla T4, bfloat16, PyTorch `2.11.0+cu130`, Transformers `5.17.0`,
Hugging Face Hub `1.31.0`, Python `3.13.15`, and CUDA `13.0`. Each archive has its own
runtime object. Source and checkpoints are immutable; transitive dependencies are
recorded, not fully locked.

## Measurement and edit definitions

For each concept the surface order is leading-space original spelling, plain spelling,
leading-space title case, and plain title case. Answers also use declared word aliases.
Readout rank is the best rank across measurable surfaces. Token mass is the sum across
their distinct IDs. The intervention endpoint is the **first available surface in the
fixed order**, not the best observed rank. Exact surfaces and IDs are visible in the reader.

Multi-token concepts are unavailable, not absent from Qwen. Numeric `60` can be predicted
correctly while its measurable word alias `sixty` has tiny probability. Alias masses are
not whole-answer probabilities or factual accuracy scores.

For a fixed token, use the raw vector `v = U[token] @ J[layer]`. With two vectors as the
columns of `V`, the swap is:

```text
c  = pinv(V) @ h
h' = h + V @ (swap(c) - c)
```

The code uses equivalent row-major arrays and rejects dependent vector pairs. It edits
all supplied prefix positions after one block, leaving text/token IDs unchanged. The
float32 edit preserves the orthogonal complement mathematically; copying it back to
bfloat16 introduces rounding. Hooks are removed in `finally` blocks.

| Condition | Edit |
|---|---|
| Concept swap | Target / decoy coordinates |
| No-op | Return hook output unchanged |
| Unrelated swap | Fixed `table` / `music` pair; not norm matched |
| Direct-answer swap | Original / alternative answer-token coordinates |
| Random, norm-matched | One seeded random direction, matching each token's float32 edit norm |

In this implementation, no-op and random controls are gated by concept-pair availability.
An independent no-op could have run even when a concept pair was unavailable, but did
not: those records are labelled unavailable, not zero. One random seed is a pilot control,
not an uncertainty estimate. Large/off-distribution edits remain a confound.

The margin is `alternative_alias_mass - original_alias_mass`; the effect is edited margin
minus clean margin. Multiply by 100 for percentage points. Its possible range is −200
to +200 points. A positive effect can be suppression of the original answer rather than
support for the alternative, so the graphics also show both masses separately.

The probability experiment edits one forward over the supplied prefix. Edited generation
is a sustained hook on every decoding forward. These scopes are different and labelled.

The archive retains ranks, alias probabilities/log-probabilities, top five tokens, and
diagnostics, not full residual tensors or full vocabulary logits. It cannot independently
reconstruct every rank count without rerunning the model.

## Graphics and interpretation

The viewer includes selectable 31-layer traces, a matched J-Lens/logit-lens switch,
six-layer prefix-position maps, intervention effects, exact token IDs, paired examples,
actual continuations, and a complete suite table. Every outcome remains visible.

The reader presents one view at a time: **Depth**, **Position**, or **Intervention**.
The prompt stays visible; continuations, the complete example table, and data/method
details start collapsed. The atlas, relay reader, token inspectors, and figure gallery
use the same restrained approach. Exported SVGs use quieter typography and neutral
backgrounds, with their plotted geometry and values preserved. This is presentation-only:
no new model run, changed measurements, or filtered examples.

Each case also has a `visualizations/iteration2/<id>-depth.svg` and
`<id>-intervention.svg`. The latter shows the clean pass and every control at fixed layer
24 on a common 0–100% axis. These are vector figures suitable for a blog.
The [figure gallery](visualizations/iteration2/index.html) displays every exported SVG
and lets you open/save it directly, without hunting for files in the output directory.

Ranks share a fixed logarithmic range. Intervention-effect plots adapt their symmetric
vertical range per example; use numeric percentage points for cross-example comparisons.
Direct-answer edits are opt-in in that plot because they address a different question and
can dominate its scale, but they remain in every exported control figure.

The explicit archive's named-spider `source` label refers to the motivating paper-example
family, not verbatim paper text. The description-only case is the actual paper prompt.
The viewer labels experimental conditions rather than repeating that ambiguous label.
Some designed descriptions use ordinary conventions, such as five-armed starfish and
standard stop signs; they are not formal taxonomy or geography benchmarks.

The supplied-space protocol also differs from Iteration 1's answer-dependent alignment
and can hurt completion/tokenization. Poor clean continuations are preserved. Do not
compare the two iterations as identical readout protocols.

## Observed results — 2 October 2026

Both complete Tesla T4 passes are archived: **40/40 prompts**, 31 J-Lens and 31 matched
logit-lens readouts per prompt, 1,200 planned intervention/control records, 1,068 executed
records, and 132 explicitly unavailable records. The reader exports **80 SVG figures**.

| Pass | Completed | Concept-pair coverage | Measurable L24 margins | Positive / negative L24 effects |
|---|---:|---:|---:|---:|
| Explicit-entity calibration | 20/20 | 17/20 | 15/20 | 13 / 2 |
| Hidden-description follow-up | 20/20 | 17/20 | 15/20 | 11 / 4 |

Positive effects are margin changes, **not successful counterfactual answers**. Three
concept pairs are unavailable because of tokenization; two additional cases lack a
measurable alternative-answer endpoint. All 40 prompts remain in the reader.

Both passes have maximum final-readout logit error **0.0625**. All 204 executed no-op
controls have exactly zero logit change. Each completed execution-history notebook has
zero error outputs; the separate failed-session history is incomplete and is not counted
as a successful run. The execution workflow stopped the Colab sessions after the downloads;
the saved archives are not evidence of current live session status.

For the original hidden-spider example, J-Lens ranks `spider` at **17 at layer 24**, before
`8` reaches rank 1 at layers 29–30. At layer 24, the concept swap changes:

- original-answer token mass: **76.34% → 72.09%**;
- alternative-answer token mass: **1.80% → 2.18%**;
- alternative-minus-original margin: **+4.63 percentage points**;
- greedy first answer: still **8**, not 6.

Most of that positive margin change is suppression of the original answer, not a large
increase in the alternative. The actual mean relative edit norm is **14.48%**, with a
maximum of **30.57%** across positions: this is not a tiny perturbation.

The hidden spider-quiz paraphrase changes its first answer from **8 to 4**, not the
proposed alternative 6. France and several other cases have poor or non-answer clean
continuations. These examples prevent interpreting a readable word, a positive margin,
or any changed answer as proof of the intended reasoning mechanism. No general mediation
or intervention-success rate is established by this pilot.

Source SHA-256 values:

```text
explicit: 1e1f39b557157ed1fb32505b8ffe87f2ab4d630a1c0d32bfb8e9da23643b7e0d
hidden:   5581ddab5b0a7c191527655c8d74130f598eade52f48c239417b817771fb93fc
```

The [run manifest](artifacts/iteration2/run_manifest.json) records source/protocol/runner
hashes, actual runtime versions, validation totals, and the generated HTML hash.
Local verification passes **44 Python tests** plus both Node chart/DOM-fixture checks
across all 40 cases, all 31 readout layers, both readout branches, and all six intervention
layers. Representative SVG samples include the hidden and explicit spider, a missing
concept surface, and a multi-token answer endpoint. Not every figure was visually audited.
The minimal-style pass additionally checks every relay prompt and atlas family, preserves
the official inspectors' embedded scripts and data, and verifies exact numeric geometry
across all 80 SVGs. Its restyled hidden-spider trace and intervention SVG were freshly
rendered and inspected; this is sample visual QA, not an audit of every exported figure.

## Reproduce it

From this checkout with an authenticated Colab CLI:

```bash
colab new -s jlens-iteration2 --gpu T4
colab upload -s jlens-iteration2 experiments/iteration2_prompts.json /content/iteration2_prompts.json
colab upload -s jlens-iteration2 experiments/qwen_iteration2.py /content/qwen_iteration2.py
colab exec -s jlens-iteration2 -f experiments/qwen_iteration2.py --timeout 1800
colab download -s jlens-iteration2 /content/qwen_iteration2.json artifacts/iteration2/qwen_iteration2.pilot.json
```

Inspect the pilot, then resume in the same kernel:

```bash
colab exec -s jlens-iteration2 -f experiments/resume_iteration2.py --timeout 1800
colab download -s jlens-iteration2 /content/qwen_iteration2.json artifacts/iteration2/qwen_iteration2.json
colab log -s jlens-iteration2 -o artifacts/iteration2/colab_execution.ipynb
colab stop -s jlens-iteration2
```

For descriptions, create a second session and upload the same runner plus
`experiments/iteration2_hidden_prompts.json` to `/content/iteration2_hidden_prompts.json`.
Execute `experiments/run_iteration2_hidden.py` for three cases, inspect them, then execute
`experiments/resume_iteration2_hidden.py`. Download `/content/qwen_iteration2_hidden.json`,
export the history, and stop that session too.

```bash
python3 experiments/build_iteration2_visualization.py
python3 -m unittest discover -s tests -v
node tests/test_iteration2_viewer.js
node tests/test_minimal_viewers.js
```

## Publication gate

Numerical and archive checks cover coordinate math, design hashes, complete retention,
alias deduplication, mass/margin arithmetic, final-readout parity, no-op invariance,
heatmap alignment, embedded-source fidelity, and all selector states in a DOM fixture.
Fixture checks are **not browser rendering or accessibility QA**.

Standalone SVGs were rendered with macOS Quick Look and visually inspected. Fresh
interactive-page rendering remains unchecked because the connected browser blocked
local-file inspection. That restriction was not bypassed using another browser or server.
Interactive visual QA is still required before publication.

### Release corrections — 2 October 2026

- All 40 intervention SVGs now label the bars as **next-token alias mass, not whole-answer
  probability**. Their saved clean and concept-edit continuations travel with the figure,
  including unchanged and unintended answers and unavailable edits.
- The figures distinguish one prefix forward for the bars from a sustained concept-edit
  hook during generation. A positive margin change is not labelled an answer-swap success.
- The conceptual guide is explicitly an Iteration 1 snapshot; evaluation prompts are
  described as hand-designed and numerical trajectories as measured, not fabricated.
- The standalone official token inspectors include the upstream Apache-2.0 and D3 ISC
  license texts. [Third-party notices](THIRD_PARTY_NOTICES.md) identify their source and
  local changes without assigning a new blanket license to the repository.
- These are presentation and packaging corrections only: neither Qwen nor the experiment
  runner was rerun or changed, and all raw measurements remain intact.
- All 51 Python tests and both Node selector fixtures pass. Fresh native SVG previews
  cover the original hidden spider, its quiz paraphrase, chemistry, alias-only numeric
  coverage, an unavailable concept pair, and the longest prompt. This is representative
  standalone-figure inspection, not a full rendered audit or interactive-page acceptance.

The connected browser again rejected local-file inspection during this correction pass.
Desktop/mobile rendering and accessibility acceptance therefore remain **pending**, not
silently marked as passed. Before publishing, check every page linked from the
[visualization index](visualizations/index.html), exercise its selectors with pointer and
keyboard input, and check for clipping and readable figure notes at narrow widths.

The data can support an explainer about readable representations and sensitivity to a
specific perturbation. It does not establish global-workspace theory, privileged causal
mediation, universal J-Lens superiority, human-like understanding, or the paper's reported
intervention success rates on Qwen.
