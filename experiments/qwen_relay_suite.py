"""Run a multi-prompt, readout-only concept-relay evaluation with Qwen J-Lens.

The suite tests whether a hidden target concept becomes readable before Qwen commits
to its answer. It keeps incorrect generations, decoy concepts, and logit-lens controls
in the saved dataset. This is descriptive evidence, not a causal intervention or a
literal transcript of model thoughts.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
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


MODEL_NAME = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
LENS_SOURCE_REF = "qwen-n1000"
LENS_REVISION = "16a01f309fcec900fdcec3f4cd5b64f3d00e4d5a"
READOUT_POSITION = -1
MAX_NEW_TOKENS = 24
TOP_K = 10
TARGET_READABLE_RANK = 100
ANSWER_COMMIT_RANK = 10
OUTPUT_PATH = Path("artifacts/qwen_relay_suite.json")


def case(
    case_id: str,
    family: str,
    prompt: str,
    target: str,
    decoy: str,
    expected: str,
    *aliases: str,
    condition: str = "hidden",
) -> dict[str, object]:
    return {
        "id": case_id,
        "family": family,
        "condition": condition,
        "prompt": prompt,
        "target": target,
        "decoy": decoy,
        "expected": expected,
        "answer_aliases": list(dict.fromkeys((expected, *aliases))),
    }


PROMPTS = [
    # Animals: infer the animal, then retrieve a count.
    case("animal-spider", "animals", "The number of legs on the animal that spins webs is", "spider", "ant", "8", "eight"),
    case("animal-spider-web-catcher", "animals", "The leg count of the creature that catches insects in webs is", "spider", "ant", "8", "eight"),
    case("animal-spider-silk", "animals", "The number of legs on the small creature that produces silk webs is", "spider", "ant", "8", "eight"),
    case("animal-spider-corners", "animals", "The number of legs on the web-building creature often found in room corners is", "spider", "ant", "8", "eight"),
    case("animal-ant", "animals", "The number of legs on the insect that follows colony scent trails is", "ant", "spider", "6", "six"),
    case("animal-bee", "animals", "The number of legs on the insect that makes honey is", "bee", "butterfly", "6", "six"),
    case("animal-butterfly", "animals", "The number of legs on the winged insect that begins life as a caterpillar is", "butterfly", "bee", "6", "six"),
    case("animal-octopus", "animals", "The number of arms on the sea animal that releases ink and has a rounded head is", "octopus", "squid", "8", "eight"),
    case("animal-crab", "animals", "The number of legs on the clawed crustacean that often walks sideways is", "crab", "lobster", "10", "ten"),
    case("animal-starfish", "animals", "The usual number of arms on the star-shaped sea animal is", "starfish", "octopus", "5", "five"),
    case("animal-cat", "animals", "The number of legs on the common pet that purrs is", "cat", "dog", "4", "four"),
    case("animal-dog", "animals", "The number of legs on the common pet that barks is", "dog", "cat", "4", "four"),
    case("animal-snake", "animals", "The number of legs on the reptile that slithers without limbs is", "snake", "lizard", "0", "zero"),
    case("animal-owl", "animals", "The number of wings on the nocturnal bird that hoots is", "owl", "eagle", "2", "two"),
    case("animal-firefly", "animals", "The number of legs on the insect that glows at night is", "firefly", "bee", "6", "six"),
    # Geometry: infer the shape, then retrieve its defining count.
    case("shape-triangle", "geometry", "The number of sides on the simplest polygon is", "triangle", "square", "3", "three"),
    case("shape-square", "geometry", "The number of sides on the regular quadrilateral is", "square", "triangle", "4", "four"),
    case("shape-pentagon", "geometry", "The number of sides on the polygon whose name begins with penta is", "pentagon", "hexagon", "5", "five"),
    case("shape-hexagon", "geometry", "The number of sides on the polygon found in a honeycomb cell is", "hexagon", "pentagon", "6", "six"),
    case("shape-heptagon", "geometry", "The number of sides on a heptagonal polygon is", "heptagon", "octagon", "7", "seven"),
    case("shape-octagon", "geometry", "The number of sides on the shape of a standard stop sign is", "octagon", "hexagon", "8", "eight"),
    case("shape-nonagon", "geometry", "The number of sides on a nonagonal polygon is", "nonagon", "octagon", "9", "nine"),
    case("shape-decagon", "geometry", "The number of sides on a decagonal polygon is", "decagon", "octagon", "10", "ten"),
    case("shape-circle", "geometry", "The number of straight sides on a perfectly round plane shape is", "circle", "oval", "0", "zero"),
    case("shape-cube", "geometry", "The number of square faces on a standard die-shaped solid is", "cube", "sphere", "6", "six"),
    case("shape-tetrahedron", "geometry", "The number of triangular faces on the simplest Platonic solid is", "tetrahedron", "cube", "4", "four"),
    case("shape-octahedron", "geometry", "The number of faces on the Platonic solid made from eight triangles is", "octahedron", "cube", "8", "eight"),
    # Chemistry: infer the element, then retrieve its atomic number.
    case("element-carbon", "chemistry", "The atomic number of the element that forms diamond is", "carbon", "silicon", "6", "six"),
    case("element-helium", "chemistry", "The atomic number of the noble gas used in party balloons is", "helium", "neon", "2", "two"),
    case("element-oxygen", "chemistry", "The atomic number of the gas humans need to breathe is", "oxygen", "nitrogen", "8", "eight"),
    case("element-sodium", "chemistry", "The atomic number of the metallic element in ordinary table salt is", "sodium", "chlorine", "11", "eleven"),
    case("element-iron", "chemistry", "The atomic number of the element whose symbol is Fe is", "iron", "copper", "26", "twenty-six"),
    case("element-gold", "chemistry", "The atomic number of the precious metal whose symbol is Au is", "gold", "silver", "79", "seventy-nine"),
    case("element-hydrogen", "chemistry", "The atomic number of the lightest element is", "hydrogen", "helium", "1", "one"),
    case("element-nitrogen", "chemistry", "The atomic number of the gas that makes up most of Earth's atmosphere is", "nitrogen", "oxygen", "7", "seven"),
    case("element-mercury", "chemistry", "The atomic number of the metal that is liquid at room temperature is", "mercury", "silver", "80", "eighty"),
    case("element-silver", "chemistry", "The atomic number of the precious metal whose symbol is Ag is", "silver", "gold", "47", "forty-seven"),
    case("element-neon", "chemistry", "The atomic number of the noble gas associated with bright advertising signs is", "neon", "helium", "10", "ten"),
    case("element-silicon", "chemistry", "The atomic number of the element used as the base material in most computer chips is", "silicon", "carbon", "14", "fourteen"),
    # Geography: infer the country, then retrieve its capital.
    case("capital-france", "geography", "The capital of the country containing the Eiffel Tower is", "France", "Italy", "Paris"),
    case("capital-japan", "geography", "The capital of the country known as the Land of the Rising Sun is", "Japan", "China", "Tokyo"),
    case("capital-egypt", "geography", "The capital of the country containing the pyramids of Giza is", "Egypt", "Sudan", "Cairo"),
    case("capital-italy", "geography", "The capital of the country containing the Colosseum is", "Italy", "Greece", "Rome"),
    case("capital-canada", "geography", "The capital of the country whose flag bears a maple leaf is", "Canada", "America", "Ottawa"),
    case("capital-spain", "geography", "The capital of the European country famous for the Sagrada Familia is", "Spain", "Portugal", "Madrid"),
    case("capital-germany", "geography", "The capital of the country containing the Brandenburg Gate is", "Germany", "Austria", "Berlin"),
    case("capital-greece", "geography", "The capital of the country containing the Acropolis is", "Greece", "Italy", "Athens"),
    case("capital-australia", "geography", "The capital of the country containing Sydney and Melbourne is", "Australia", "Zealand", "Canberra"),
    case("capital-brazil", "geography", "The capital of the country containing Rio de Janeiro is", "Brazil", "Argentina", "Brasilia", "Brasília"),
    case("capital-kenya", "geography", "The capital of the East African country famous for the Maasai Mara is", "Kenya", "Tanzania", "Nairobi"),
    case("capital-thailand", "geography", "The capital of the Southeast Asian country formerly called Siam is", "Thailand", "Vietnam", "Bangkok"),
    # Explicit-entity baselines calibrate how much harder hidden descriptions are.
    case("explicit-spider", "animals", "The number of legs on a spider is", "spider", "ant", "8", "eight", condition="explicit"),
    case("explicit-octopus", "animals", "The number of arms on an octopus is", "octopus", "squid", "8", "eight", condition="explicit"),
    case("explicit-dog", "animals", "The number of legs on a dog is", "dog", "cat", "4", "four", condition="explicit"),
    case("explicit-triangle", "geometry", "The number of sides on a triangle is", "triangle", "square", "3", "three", condition="explicit"),
    case("explicit-hexagon", "geometry", "The number of sides on a hexagon is", "hexagon", "pentagon", "6", "six", condition="explicit"),
    case("explicit-cube", "geometry", "The number of square faces on a cube is", "cube", "sphere", "6", "six", condition="explicit"),
    case("explicit-carbon", "chemistry", "The atomic number of carbon is", "carbon", "silicon", "6", "six", condition="explicit"),
    case("explicit-oxygen", "chemistry", "The atomic number of oxygen is", "oxygen", "nitrogen", "8", "eight", condition="explicit"),
    case("explicit-gold", "chemistry", "The atomic number of gold is", "gold", "silver", "79", "seventy-nine", condition="explicit"),
    case("explicit-france", "geography", "The capital of France is", "France", "Italy", "Paris", condition="explicit"),
    case("explicit-japan", "geography", "The capital of Japan is", "Japan", "China", "Tokyo", condition="explicit"),
    case("explicit-brazil", "geography", "The capital of Brazil is", "Brazil", "Argentina", "Brasilia", "Brasília", condition="explicit"),
]


def surface_forms(terms: list[str]) -> list[str]:
    forms: list[str] = []
    for term in terms:
        for text in (term, term.lower(), term.title()):
            forms.extend((text, f" {text}"))
    return list(dict.fromkeys(forms))


def resolve_single_token_variants(tokenizer, terms: list[str]) -> list[dict[str, object]]:
    variants: list[dict[str, object]] = []
    seen: set[int] = set()
    for surface in surface_forms(terms):
        token_ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(token_ids) == 1 and token_ids[0] not in seen:
            variants.append({"surface": surface, "token_id": token_ids[0]})
            seen.add(token_ids[0])
    return variants


def top_tokens(tokenizer, logits: torch.Tensor, k: int = TOP_K) -> list[str]:
    return [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for token_id in logits.float().topk(k).indices.tolist()
    ]


def measurements(
    logits: torch.Tensor,
    variants: dict[str, list[dict[str, object]]],
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
        result[concept] = {"best": rows[0] if rows else None, "variants": rows}
    return result


def first_content_token(tokenizer, continuation_ids: torch.Tensor) -> int:
    for index, token_id in enumerate(continuation_ids.tolist()):
        piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if re.search(r"[\w\d]", piece, flags=re.UNICODE):
            return index
    return 0


def normalized(text: str) -> str:
    return re.sub(r"^[\s\W_]+", "", text, flags=re.UNICODE).casefold()


def answer_matches(continuation: str, aliases: list[str]) -> bool:
    observed = normalized(continuation)
    for alias in aliases:
        expected = normalized(alias)
        if re.match(rf"^{re.escape(expected)}(?:\b|$)", observed):
            return True
    return False


def best_point(layers: dict[str, object], lens_key: str, concept: str) -> dict[str, int] | None:
    points = []
    for layer_text, row in layers.items():
        best = row[lens_key][concept]["best"]
        if best is not None:
            points.append({"layer": int(layer_text), "rank": int(best["rank"])})
    return min(points, key=lambda point: (point["rank"], point["layer"])) if points else None


def first_layer_at_rank(
    layers: dict[str, object], lens_key: str, concept: str, threshold: int
) -> int | None:
    for layer_text in sorted(layers, key=int):
        best = layers[layer_text][lens_key][concept]["best"]
        if best is not None and int(best["rank"]) <= threshold:
            return int(layer_text)
    return None


def summarize_case(row: dict[str, object]) -> dict[str, object]:
    layers = row["layers"]
    target_peak = best_point(layers, "jlens", "target")
    answer_peak = best_point(layers, "jlens", "answer")
    logit_target_peak = best_point(layers, "logit_lens", "target")
    answer_commit_layer = first_layer_at_rank(
        layers, "jlens", "answer", ANSWER_COMMIT_RANK
    )
    target_specific = None
    if target_peak is not None:
        peak_row = layers[str(target_peak["layer"])]
        decoy_best = peak_row["jlens"]["decoy"]["best"]
        if decoy_best is not None:
            target_specific = target_peak["rank"] < int(decoy_best["rank"])
    relay = bool(
        row["correct"]
        and target_peak is not None
        and target_peak["rank"] <= TARGET_READABLE_RANK
        and answer_commit_layer is not None
        and target_peak["layer"] < answer_commit_layer
    )
    return {
        "target_peak": target_peak,
        "answer_peak": answer_peak,
        "logit_target_peak": logit_target_peak,
        "answer_commit_layer": answer_commit_layer,
        "target_specific": target_specific,
        "relay": relay,
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter()
    by_family: dict[str, Counter] = defaultdict(Counter)
    for row in results:
        family = str(row["family"])
        counts["total"] += 1
        by_family[family]["total"] += 1
        for key in ("correct", "measurable"):
            if row[key]:
                counts[key] += 1
                by_family[family][key] += 1
        for key in ("relay", "target_specific"):
            if row["summary"][key] is True:
                counts[key] += 1
                by_family[family][key] += 1
    return {
        **counts,
        "by_family": {family: dict(values) for family, values in by_family.items()},
        "definitions": {
            "correct": "greedy continuation starts with an expected answer alias",
            "measurable": "target, decoy, and answer each have a single-token surface",
            "relay": (
                f"correct; target reaches rank <= {TARGET_READABLE_RANK}; "
                f"target peak precedes answer's first rank <= {ANSWER_COMMIT_RANK}"
            ),
            "target_specific": "target outranks the matched decoy at target peak",
        },
    }


def main() -> None:
    assert torch.cuda.is_available(), "A Colab GPU runtime is required"
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
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
    layer_ids = list(lens.source_layers)
    results: list[dict[str, object]] = []

    for prompt_index, item in enumerate(PROMPTS, start=1):
        prompt = str(item["prompt"])
        aliases = list(item["answer_aliases"])
        variants = {
            "target": resolve_single_token_variants(tokenizer, [str(item["target"])]),
            "decoy": resolve_single_token_variants(tokenizer, [str(item["decoy"])]),
            "answer": resolve_single_token_variants(tokenizer, aliases),
        }
        raw_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            generated = hf_model.generate(
                input_ids=raw_input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation_ids = generated[0, raw_input_ids.shape[1] :]
        content_index = first_content_token(tokenizer, continuation_ids)
        prefix_ids = generated[0, : raw_input_ids.shape[1] + content_index]
        readout_prefix = tokenizer.decode(
            prefix_ids, clean_up_tokenization_spaces=False
        )
        continuation = tokenizer.decode(
            continuation_ids, skip_special_tokens=False
        )

        j_logits, api_logits, input_ids = lens.apply(
            model,
            readout_prefix,
            layers=layer_ids,
            positions=[READOUT_POSITION],
        )
        ll_logits, _, _ = lens.apply(
            model,
            readout_prefix,
            layers=layer_ids,
            positions=[READOUT_POSITION],
            use_jacobian=False,
        )
        assert torch.equal(input_ids.cpu(), prefix_ids.unsqueeze(0).cpu())
        with torch.no_grad():
            direct = hf_model(input_ids=input_ids).logits[0, READOUT_POSITION].float().cpu()
        final = api_logits[0].float().cpu()
        parity_error = float((direct - final).abs().max().item())
        assert torch.allclose(direct, final, atol=0.25, rtol=0.01), parity_error

        layer_rows = {}
        for layer in layer_ids:
            j_row = j_logits[layer][0].float().cpu()
            ll_row = ll_logits[layer][0].float().cpu()
            layer_rows[str(layer)] = {
                "jlens_top10": top_tokens(tokenizer, j_row),
                "logit_lens_top10": top_tokens(tokenizer, ll_row),
                "jlens": measurements(j_row, variants),
                "logit_lens": measurements(ll_row, variants),
            }

        row = {
            **item,
            "correct": answer_matches(continuation, aliases),
            "measurable": all(bool(values) for values in variants.values()),
            "continuation": continuation,
            "content_token_index": content_index,
            "readout_prefix": readout_prefix,
            "readout_token": tokenizer.decode(
                [int(input_ids[0, READOUT_POSITION].item())],
                clean_up_tokenization_spaces=False,
            ),
            "model_top10": top_tokens(tokenizer, final),
            "model_concepts": measurements(final, variants),
            "model_parity_max_abs_error": parity_error,
            "resolved_variants": variants,
            "layers": layer_rows,
        }
        row["summary"] = summarize_case(row)
        results.append(row)

        partial = {
            "schema_version": 1,
            "scope": "readout-only; no causal interventions",
            "completed": len(results),
            "planned": len(PROMPTS),
            "results": results,
        }
        OUTPUT_PATH.write_text(json.dumps(partial, indent=2, ensure_ascii=False))
        print(
            f"[{prompt_index:02d}/{len(PROMPTS)}] {item['id']}: "
            f"correct={row['correct']} relay={row['summary']['relay']} "
            f"continuation={continuation[:32]!r}"
        )
        del j_logits, ll_logits, api_logits, direct, final
        torch.cuda.empty_cache()

    report = {
        "schema_version": 1,
        "scope": "readout-only; no causal interventions",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "readout_semantics": (
            "per prompt, decode immediately before Qwen's first generated content token"
        ),
        "thresholds": {
            "target_readable_rank": TARGET_READABLE_RANK,
            "answer_commit_rank": ANSWER_COMMIT_RANK,
        },
        "aggregate": aggregate(results),
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(OUTPUT_PATH.resolve()), "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
