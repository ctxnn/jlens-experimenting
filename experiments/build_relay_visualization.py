"""Build the self-contained multi-prompt Qwen relay visualization."""

from __future__ import annotations

import ast
import json
from pathlib import Path


SOURCE = Path("artifacts/qwen_relay_suite.json")
SUMMARY = Path("artifacts/qwen_relay_suite_summary.json")
TEMPLATE = Path("visualizations/qwen-concept-relay.template.html")
OUTPUT = Path("visualizations/qwen-concept-relay.html")
DATA_MARKER = "__QWEN_RELAY_DATA__"
ATLAS = Path("visualizations/jlens-concept-atlas.html")


def branch_summary(row: dict, lens_key: str, thresholds: dict) -> dict:
    """Apply the archived relay rule identically to either readout branch."""
    points = [(int(key), value) for key, value in sorted(row["layers"].items(), key=lambda item: int(item[0]))]
    candidates = [(best_rank(value, lens_key, "target"), layer) for layer, value in points]
    candidates = [(rank, layer) for rank, layer in candidates if rank is not None]
    peak = min(candidates) if candidates else None
    commit = next((layer for layer, value in points
                   if (rank := best_rank(value, lens_key, "answer")) is not None
                   and rank <= thresholds["answer_commit_rank"]), None)
    return {
        "target_peak": {"layer": peak[1], "rank": peak[0]} if peak else None,
        "answer_commit_layer": commit,
        "relay": bool(row["correct"] and peak and peak[0] <= thresholds["target_readable_rank"]
                      and commit is not None and peak[1] < commit),
    }


def atlas_data(validation: dict, notebook: dict) -> dict:
    """Keep complete raw decoded lists; normalize only in the display layer."""
    prompts = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or not "".join(cell["source"]).startswith("PROMPTS ="):
            continue
        assignment = ast.parse("".join(cell["source"])).body[0]
        prompts = ast.literal_eval(assignment.value)
    if prompts is None:
        raise ValueError("Notebook PROMPTS declaration was not found")
    labels = {"typo": "typo recovery", "ascii_face": "ASCII face"}
    return {
        name: {
            "label": labels.get(name, name), "text": prompts[name],
            "model": row["model_top10"],
            "layers": {layer: {"j": values["jlens_top10"], "l": values["logit_lens_top10"],
                               "o": values["top10_overlap"]}
                       for layer, values in row["layers"].items()},
        }
        for name, row in validation["prompts"].items()
    }


def best_rank(row: dict[str, object], lens_key: str, concept: str) -> int | None:
    best = row[lens_key][concept]["best"]
    return int(best["rank"]) if best is not None else None


def compact(source: dict[str, object]) -> dict[str, object]:
    results = []
    for row in source["results"]:
        layers = []
        for layer_text in sorted(row["layers"], key=int):
            layer = row["layers"][layer_text]
            layers.append(
                [
                    int(layer_text),
                    best_rank(layer, "jlens", "target"),
                    best_rank(layer, "jlens", "answer"),
                    best_rank(layer, "jlens", "decoy"),
                    best_rank(layer, "logit_lens", "target"),
                    best_rank(layer, "logit_lens", "answer"),
                    best_rank(layer, "logit_lens", "decoy"),
                ]
            )
        results.append(
            {
                "id": row["id"],
                "family": row["family"],
                "condition": row["condition"],
                "prompt": row["prompt"],
                "target": row["target"],
                "decoy": row["decoy"],
                "expected": row["expected"],
                "correct": row["correct"],
                "measurable": row["measurable"],
                "missing_surfaces": [key for key, variants in row["resolved_variants"].items() if not variants],
                "readout_prefix": row["readout_prefix"],
                "readout_token": row["readout_token"],
                "content_token_index": row["content_token_index"],
                "continuation": row["continuation"][:240],
                "model_top10": row["model_top10"],
                "summary": row["summary"],
                "logit_summary": branch_summary(row, "logit_lens", source["thresholds"]),
                "layers": layers,
            }
        )
    return {
        "schema_version": 1,
        "scope": source["scope"],
        "generated_at_utc": source["generated_at_utc"],
        "model": source["model"],
        "model_revision": source["model_revision"],
        "lens_revision": source["lens_revision"],
        "reference_commit": source["reference_commit"],
        "lens_n_prompts": source["lens_n_prompts"],
        "gpu": source["gpu"],
        "thresholds": source["thresholds"],
        "aggregate": source["aggregate"],
        "comparison": {
            condition: {"total": len(rows),
                        "jlens_relay": sum(row["summary"]["relay"] for row in rows),
                        "logit_relay": sum(row["logit_summary"]["relay"] for row in rows)}
            for condition in ("all", "hidden", "explicit")
            for rows in [[row for row in results if condition == "all" or row["condition"] == condition]]
        },
        "layer_columns": [
            "layer",
            "jlens_target_rank",
            "jlens_answer_rank",
            "jlens_decoy_rank",
            "logit_target_rank",
            "logit_answer_rank",
            "logit_decoy_rank",
        ],
        "results": results,
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    data = compact(source)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text()
    assert template.count(DATA_MARKER) == 1
    SUMMARY.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    OUTPUT.write_text(template.replace(DATA_MARKER, encoded))
    validation = json.loads(Path("artifacts/qwen_validation.json").read_text())
    notebook = json.loads(Path("notebooks/qwen_official_jlens.ipynb").read_text())
    atlas = ATLAS.read_text()
    before, rest = atlas.split("    const prompts = ", 1)
    _, after = rest.split("\n\n    let activePrompt", 1)
    atlas_encoded = json.dumps(atlas_data(validation, notebook), ensure_ascii=False, indent=2).replace("<", "\\u003c")
    ATLAS.write_text(before + "    const prompts = " + atlas_encoded + ";\n\n    let activePrompt" + after)
    print(
        json.dumps(
            {
                "source": str(SOURCE),
                "summary": str(SUMMARY),
                "visualization": str(OUTPUT),
                "prompts": len(data["results"]),
                "html_bytes": OUTPUT.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
