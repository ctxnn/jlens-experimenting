"""Pinned, readout + intervention experiment. Import is safe on a CPU machine.

Run on the allocated Colab kernel with experiments/iteration2_prompts.json at
/content/iteration2_prompts.json. I2_CASE_LIMIT defaults to three (the pilot).
Set I2_CASE_LIMIT=20 and re-execute to resume the frozen suite in the same kernel.
No Qwen training, fine-tuning, or lens fitting happens here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

REFERENCE_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
CONDITIONS = ("concept_swap", "no_op", "unrelated_swap", "answer_swap", "random_norm_matched")


def coordinate_swap_delta(hidden, vectors, strength=1.0):
    """Exchange two coefficients in a fixed rank-two span, preserving its complement.

    Row-major hidden [..., d], vectors [d, 2]; no centering or clamping.
    This CPU float32 implementation is also the implementation used by GPU hooks.
    """
    import numpy as np
    h = np.asarray(hidden, dtype=np.float32)
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 2 or h.shape[-1] != v.shape[0]:
        raise ValueError("Expected hidden [..., d] and vectors [d, 2]")
    if not np.isfinite(h).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite hidden state or vector")
    if np.linalg.matrix_rank(v) != 2:
        raise ValueError("Coordinate swap requires two independent vectors")
    coefficients = h @ np.linalg.pinv(v).T
    return np.float32(strength) * ((coefficients[..., ::-1] - coefficients) @ v.T)


def norm_matched_delta(delta, seed):
    import numpy as np
    d = np.asarray(delta, dtype=np.float32)
    noise = np.random.default_rng(seed).standard_normal(d.shape).astype(np.float32)
    norms = np.linalg.norm(noise, axis=-1, keepdims=True)
    return noise * np.linalg.norm(d, axis=-1, keepdims=True) / np.maximum(norms, 1e-12)


def surface_forms(text):
    return list(dict.fromkeys((" " + text, text, " " + text.title(), text.title())))


def resolve_tokens(tokenizer, texts):
    tokens, seen = [], set()
    for text in texts:
        for surface in surface_forms(text):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in seen:
                tokens.append({"id": int(ids[0]), "surface": surface, "text": tokenizer.decode(ids)})
                seen.add(ids[0])
    return tokens


def measure(scores, tokens):
    import torch
    values = scores.detach().float().cpu().flatten()
    log_probs = torch.log_softmax(values, dim=-1)
    variants = [{**t, "rank": int((values > values[t["id"]]).sum()) + 1,
                 "log_probability": float(log_probs[t["id"]]),
                 "probability": float(log_probs[t["id"]].exp())} for t in tokens]
    variants.sort(key=lambda t: (t["rank"], -t["log_probability"]))
    return {"best": variants[0] if variants else None, "variants": variants,
            "mass": sum(t["probability"] for t in variants) if variants else None}


def top_tokens(scores, tokenizer, count=5):
    import torch
    values = scores.detach().float().cpu().flatten()
    log_probs = torch.log_softmax(values, dim=-1)
    vals, ids = torch.topk(values, count)
    return [{"id": int(i), "text": tokenizer.decode([int(i)]),
             "log_probability": float(log_probs[i]), "probability": float(log_probs[i].exp())}
            for i in ids]


def load_context():
    # The reference package is pinned; dependency versions are captured in the report.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/anthropics/jacobian-lens.git@" + REFERENCE_COMMIT,
                    "transformers>=5.0.0", "accelerate", "huggingface_hub"], check=True)
    import torch
    import transformers
    import huggingface_hub
    import jlens
    from huggingface_hub import hf_hub_download
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires the allocated CUDA GPU")
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda").eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
    config_path = hf_hub_download(MODEL_NAME, "config.json", revision=MODEL_REVISION)
    lens_path = hf_hub_download(LENS_REPO, LENS_FILE, revision=LENS_REVISION)
    if Path(config_path).parent.name != MODEL_REVISION:
        raise RuntimeError("Resolved model revision differs from the frozen revision")
    if "snapshots/" + LENS_REVISION + "/" not in lens_path:
        raise RuntimeError("Resolved lens revision differs from the frozen revision")
    return {"hf_model": hf_model, "tokenizer": tokenizer, "model": model, "lens": lens,
            "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                        "huggingface_hub": huggingface_hub.__version__,
                        "python": sys.version, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16",
                        "resolved_model_revision": Path(config_path).parent.name,
                        "resolved_lens_path": lens_path}}


def build_vectors(context, layer, token_ids):
    import numpy as np
    model, lens = context["model"], context["lens"]
    rows = model._lm_head.weight[list(token_ids)].detach().float().cpu().numpy()
    jacobian = lens.jacobians[layer].detach().float().cpu().numpy()
    vectors = (rows @ jacobian).T
    if np.linalg.matrix_rank(vectors) != 2:
        raise ValueError("Selected token vectors are not independent")
    return vectors


def patch_hook(vectors, condition, seed, diagnostics):
    import numpy as np
    import torch

    def hook(module, inputs, output):
        if condition == "no_op":
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        h = hidden.detach().float().cpu().numpy()
        delta = coordinate_swap_delta(h, vectors)
        if condition == "random_norm_matched":
            delta = norm_matched_delta(delta, seed)
        edited = hidden + torch.from_numpy(delta).to(device=hidden.device, dtype=hidden.dtype)
        actual = edited.detach().float().cpu().numpy() - h
        diagnostics.append({"mean_relative_edit_norm": float(np.mean(
            np.linalg.norm(actual, axis=-1) / np.maximum(np.linalg.norm(h, axis=-1), 1e-12))),
            "max_relative_edit_norm": float(np.max(
            np.linalg.norm(actual, axis=-1) / np.maximum(np.linalg.norm(h, axis=-1), 1e-12)))})
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    return hook


def ordinary_scores(context, ids, layer=None, vectors=None, condition=None, seed=0):
    import torch
    diagnostics = []
    handle = context["model"].layers[layer].register_forward_hook(
        patch_hook(vectors, condition, seed, diagnostics)) if layer is not None else None
    try:
        with torch.inference_mode():
            scores = context["hf_model"](input_ids=ids, use_cache=False).logits[0, -1].detach().float().cpu()
        return scores, diagnostics
    finally:
        if handle is not None:
            handle.remove()


def continuation(context, ids, count, layer=None, vectors=None, condition=None, seed=0):
    import torch
    result = ids.clone()
    diagnostics = []
    handle = context["model"].layers[layer].register_forward_hook(
        patch_hook(vectors, condition, seed, diagnostics)) if layer is not None else None
    try:
        with torch.inference_mode():
            for _ in range(count):
                scores = context["hf_model"](input_ids=result, use_cache=False).logits[:, -1]
                next_id = scores.argmax(dim=-1, keepdim=True)
                result = torch.cat([result, next_id], dim=1)
                if int(next_id.item()) in (context["tokenizer"].eos_token_id,
                                          getattr(context["hf_model"].generation_config, "eos_token_id", None)):
                    break
    finally:
        if handle is not None:
            handle.remove()
    generated = result[0, ids.shape[1]:].tolist()
    return {"text": context["tokenizer"].decode(generated), "token_ids": generated,
            "maximum_tokens": count, "intervention_layer": layer,
            "scope": "all positions on every decoding forward; no cache" if layer is not None else "none"}


def score_record(scores, groups, tokenizer):
    measurements = {name: measure(scores, tokens) for name, tokens in groups.items()}
    a, b = measurements["answer"]["mass"], measurements["alternative"]["mass"]
    return {"measurements": measurements, "top_tokens": top_tokens(scores, tokenizer),
            "alternative_minus_original_mass": b - a if a is not None and b is not None else None}


def run_case(context, case, protocol):
    import numpy as np
    import torch
    from jlens.hooks import ActivationRecorder
    started = time.monotonic()
    model, lens, tokenizer = context["model"], context["lens"], context["tokenizer"]
    prefix = case["prompt"] + protocol["readout_suffix"]
    ids = model.encode(prefix)
    groups = {"target": resolve_tokens(tokenizer, [case["target"]]),
              "decoy": resolve_tokens(tokenizer, [case["decoy"]]),
              "answer": resolve_tokens(tokenizer, case["answer_aliases"]),
              "alternative": resolve_tokens(tokenizer, case["alternative_aliases"])}
    unrelated = [resolve_tokens(tokenizer, [x]) for x in ("table", "music")]
    clean_scores, _ = ordinary_scores(context, ids)
    result = {**case, "readout_prefix": prefix, "prefix_token_ids": ids[0].tolist(),
              "prefix_tokens": [tokenizer.decode([int(i)]) for i in ids[0]],
              "readout_position": int(ids.shape[1] - 1), "token_groups": groups,
              "clean": score_record(clean_scores, groups, tokenizer), "readouts": [], "interventions": [],
              "clean_generation": continuation(context, ids, protocol["generation_tokens"])}
    layers = sorted(lens.source_layers)
    with torch.inference_mode(), ActivationRecorder(model.layers, at=layers + [model.n_layers - 1]) as recorder:
        model.forward(ids)
    final_scores = model.unembed(recorder.activations[model.n_layers - 1][0, -1]).float().cpu()
    result["final_readout_max_abs_logit_error"] = float((clean_scores - final_scores).abs().max())
    for layer in layers:
        hidden = recorder.activations[layer][0].float()
        with torch.inference_mode():
            jl_scores = model.unembed(lens.transport(hidden[-1], layer))
            ll_scores = model.unembed(hidden[-1])
        row = {"layer": layer, "j_lens": score_record(jl_scores, groups, tokenizer),
               "logit_lens": score_record(ll_scores, groups, tokenizer)}
        if layer in protocol["intervention_layers"]:
            with torch.inference_mode():
                all_scores = model.unembed(lens.transport(hidden, layer))
            row["target_rank_by_position"] = [measure(s, groups["target"])["best"]["rank"]
                if groups["target"] else None for s in all_scores]
            del all_scores
        result["readouts"].append(row)
    del recorder, final_scores
    for layer in protocol["intervention_layers"]:
        pairs = {"concept_swap": (groups["target"], groups["decoy"]),
                 "unrelated_swap": tuple(unrelated),
                 "answer_swap": (groups["answer"], groups["alternative"])}
        for condition in CONDITIONS:
            pair = pairs.get(condition, pairs["concept_swap"])
            if not all(pair):
                result["interventions"].append({"layer": layer, "condition": condition,
                    "available": False, "reason": "No single-token surface for one or both endpoints"})
                continue
            token_ids = [p[0]["id"] for p in pair]  # Fixed order, independent of measured ranks.
            try:
                vectors = build_vectors(context, layer, token_ids)
                seed = protocol["seed"] + layer + int(hashlib.sha256(case["id"].encode()).hexdigest()[:8], 16)
                scores, diagnostics = ordinary_scores(context, ids, layer, vectors, condition, seed)
                record = {"layer": layer, "condition": condition, "available": True,
                          "selected_tokens": [p[0] for p in pair], "vector_norms": np.linalg.norm(vectors, axis=0).tolist(),
                          "vector_cosine": float(np.dot(vectors[:, 0], vectors[:, 1]) /
                              np.prod(np.linalg.norm(vectors, axis=0))),
                          "vector_condition_number": float(np.linalg.cond(vectors)),
                          "max_abs_logit_change": float((scores - clean_scores).abs().max()),
                          "diagnostics": diagnostics, **score_record(scores, groups, tokenizer)}
                baseline = result["clean"]["alternative_minus_original_mass"]
                record["margin_change"] = (record["alternative_minus_original_mass"] - baseline
                    if baseline is not None and record["alternative_minus_original_mass"] is not None else None)
                if condition == "concept_swap" and layer == protocol["generation_layer"]:
                    record["generation"] = continuation(context, ids, protocol["generation_tokens"],
                                                        layer, vectors, condition, seed)
                result["interventions"].append(record)
            except ValueError as error:
                result["interventions"].append({"layer": layer, "condition": condition,
                                               "available": False, "reason": str(error)})
        print(f"  {case['id']} layer {layer} complete", flush=True)
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main():
    global I2_CONTEXT
    protocol_path = Path(os.environ.get("I2_PROTOCOL", "/content/iteration2_prompts.json"))
    output_path = Path(os.environ.get("I2_OUTPUT", "/content/qwen_iteration2.json"))
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    if "I2_CONTEXT" not in globals():
        I2_CONTEXT = load_context()
    context = I2_CONTEXT
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    report = json.loads(output_path.read_text()) if output_path.exists() else {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME, "model_revision": MODEL_REVISION, "lens_revision": LENS_REVISION,
        "lens_file": LENS_FILE, "reference_commit": REFERENCE_COMMIT,
        "runtime": context["runtime"], "protocol_sha256": protocol_hash,
        "protocol": protocol, "source_layers": sorted(context["lens"].source_layers),
        "lens_training_prompts": context["lens"].n_prompts,
        "intervention_definition": "Unclamped coordinate swap in raw U@J token-vector span; post-block, all prefix positions; strength 1",
        "readout_definition": "Last token of prompt plus a fixed supplied space; no answer-dependent alignment",
        "cases": [], "errors": []}
    if report["protocol_sha256"] != protocol_hash:
        raise RuntimeError("Refusing to resume results from a different protocol")
    script_path = Path("/content/qwen_iteration2.py")
    if script_path.exists():
        report["runner_sha256"] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    limit = int(os.environ.get("I2_CASE_LIMIT", "3"))
    completed = {case["id"] for case in report["cases"]}
    for case in protocol["cases"][:limit]:
        if case["id"] in completed:
            continue
        print(f"START {case['id']}", flush=True)
        report["cases"].append(run_case(context, case, protocol))
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        report["completed_cases"] = len(report["cases"])
        output_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"SAVED {case['id']} ({len(report['cases'])}/{len(protocol['cases'])})", flush=True)
    print(f"RESULT {output_path} {len(report['cases'])} cases", flush=True)


if __name__ == "__main__":
    main()
