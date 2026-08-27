"""Measure the paper's unspoken-spider prompt with the published Qwen J-Lens.

This is a readout experiment, not a spider-to-ant causal swap. The public ``jlens``
package currently exposes the read side but does not ship the paper's intervention
hooks. Results therefore describe token ranks and must not be presented as proof that
the decoded concepts causally mediate Qwen's answer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        f"git+https://github.com/anthropics/jacobian-lens.git@{REFERENCE_COMMIT}",
    ]
)

import jlens  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from jlens.vis import build_page, compute_slice  # noqa: E402


MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
LENS_SOURCE_REF = "qwen-n1000"
LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
PROMPT = "The number of legs on the animal that spins webs is"
READOUT_PREFIX = PROMPT + " "
READOUT_POSITION = -1
CONCEPT_SURFACES = {
    "spider": [" spider", "spider", " Spider", "Spider"],
    "ant": [" ant", "ant", " Ant", "Ant"],
    "eight": [" 8", "8", " eight", "eight", " Eight", "Eight"],
    "six": [" 6", "6", " six", "six", " Six", "Six"],
    "legs": [" legs", "legs", " Legs", "Legs"],
}


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 10) -> list[str]:
    token_ids = logits.float().topk(k).indices.tolist()
    return [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]


def single_token_variants(tokenizer) -> dict[str, list[dict[str, int | str]]]:
    resolved: dict[str, list[dict[str, int | str]]] = {}
    for concept, surfaces in CONCEPT_SURFACES.items():
        variants: list[dict[str, int | str]] = []
        seen_ids: set[int] = set()
        for surface in surfaces:
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in seen_ids:
                variants.append({"surface": surface, "token_id": token_ids[0]})
                seen_ids.add(token_ids[0])
        if not variants:
            raise RuntimeError(f"No single-token Qwen surface for {concept!r}")
        resolved[concept] = variants
    return resolved


def concept_measurements(
    logits: torch.Tensor,
    variants: dict[str, list[dict[str, int | str]]],
) -> dict[str, dict[str, object]]:
    scores = logits.float().cpu()
    log_probs = torch.log_softmax(scores, dim=-1)
    result: dict[str, dict[str, object]] = {}
    for concept, candidates in variants.items():
        rows = []
        for candidate in candidates:
            token_id = int(candidate["token_id"])
            rows.append(
                {
                    **candidate,
                    "rank": int((scores > scores[token_id]).sum().item()) + 1,
                    "log_probability": float(log_probs[token_id].item()),
                }
            )
        rows.sort(key=lambda row: (int(row["rank"]), -float(row["log_probability"])))
        result[concept] = {"best": rows[0], "variants": rows}
    return result


def main() -> None:
    assert torch.cuda.is_available(), "A Colab GPU runtime is required"
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    jlens.configure_logging()

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION
    )
    resolved_config_path = Path(
        hf_hub_download(MODEL_NAME, "config.json", revision=MODEL_REVISION)
    )
    snapshot_index = resolved_config_path.parts.index("snapshots")
    resolved_model_revision = resolved_config_path.parts[snapshot_index + 1]
    assert resolved_model_revision == MODEL_REVISION

    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    layers = list(lens.source_layers)
    variants = single_token_variants(tokenizer)

    raw_input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        raw_logits = hf_model(input_ids=raw_input_ids).logits[0, -1].float().cpu()
        generated = hf_model.generate(
            input_ids=raw_input_ids,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    continuation_ids = generated[0, raw_input_ids.shape[1] :]
    first_generated_token_id = int(continuation_ids[0].item())
    first_generated_token = tokenizer.decode(
        [first_generated_token_id], clean_up_tokenization_spaces=False
    )
    assert first_generated_token == " ", repr(first_generated_token)

    j_logits, api_logits, input_ids = lens.apply(
        model,
        READOUT_PREFIX,
        layers=layers,
        positions=[READOUT_POSITION],
    )
    ll_logits, _, _ = lens.apply(
        model,
        READOUT_PREFIX,
        layers=layers,
        positions=[READOUT_POSITION],
        use_jacobian=False,
    )
    with torch.no_grad():
        direct = hf_model(input_ids=input_ids).logits[0, READOUT_POSITION].float().cpu()
    final = api_logits[0].float().cpu()
    expected_readout_ids = torch.cat(
        [raw_input_ids, continuation_ids[:1].unsqueeze(0)], dim=1
    )
    assert torch.equal(input_ids.cpu(), expected_readout_ids.cpu())
    parity_error = float((direct - final).abs().max().item())
    assert torch.allclose(direct, final, atol=0.25, rtol=0.01), parity_error

    layer_rows = {}
    for layer in layers:
        j_row = j_logits[layer][0].float().cpu()
        ll_row = ll_logits[layer][0].float().cpu()
        layer_rows[str(layer)] = {
            "jlens_top10": top_tokens(tokenizer, j_row),
            "logit_lens_top10": top_tokens(tokenizer, ll_row),
            "jlens": concept_measurements(j_row, variants),
            "logit_lens": concept_measurements(ll_row, variants),
        }

    report = {
        "schema_version": 2,
        "scope": "readout-only; no causal spider-to-ant swap",
        "reference_commit": REFERENCE_COMMIT,
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": resolved_model_revision,
        "lens_revision": LENS_REVISION,
        "lens_source_ref": LENS_SOURCE_REF,
        "lens_n_prompts": lens.n_prompts,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prompt": PROMPT,
        "readout_prefix": READOUT_PREFIX,
        "readout_prefix_source": (
            "exact prompt plus Qwen's first greedily generated token"
        ),
        "readout_position": READOUT_POSITION,
        "readout_semantics": (
            "logits after Qwen's first generated whitespace token predict the answer token"
        ),
        "readout_token": tokenizer.decode(
            [int(input_ids[0, READOUT_POSITION].item())],
            clean_up_tokenization_spaces=False,
        ),
        "model_parity_max_abs_error": parity_error,
        "raw_model_top10": top_tokens(tokenizer, raw_logits),
        "raw_model_concepts": concept_measurements(raw_logits, variants),
        "first_generated_token_id": first_generated_token_id,
        "first_generated_token": first_generated_token,
        "greedy_continuation": tokenizer.decode(
            continuation_ids, skip_special_tokens=False
        ),
        "model_top10": top_tokens(tokenizer, final),
        "model_concepts": concept_measurements(final, variants),
        "layers": layer_rows,
    }
    json_path = output_dir / "qwen_spider_trace.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    slice_data = compute_slice(
        model,
        lens,
        READOUT_PREFIX,
        layer_stride=1,
        top_n=10,
        mask_display=True,
    )
    page, _, _ = build_page(
        slice_data,
        READOUT_PREFIX,
        title="Qwen official J-Lens · unspoken spider trace",
        description=(
            "Real Qwen readout using the published n=1,000 lens. "
            "Descriptive ranks only; no causal swap."
        ),
        mode="embed",
    )
    html_path = output_dir / "qwen_spider_slice.html"
    html_path.write_text(page)

    print(json.dumps({
        "json": str(json_path.resolve()),
        "html": str(html_path.resolve()),
        "model_top10": report["model_top10"],
        "parity_error": parity_error,
    }, indent=2))


if __name__ == "__main__":
    main()
