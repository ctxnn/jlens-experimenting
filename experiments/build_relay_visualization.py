"""Build the self-contained multi-prompt Qwen relay visualization."""

from __future__ import annotations

import json
from pathlib import Path


SOURCE = Path("artifacts/qwen_relay_suite.json")
SUMMARY = Path("artifacts/qwen_relay_suite_summary.json")
TEMPLATE = Path("visualizations/qwen-concept-relay.template.html")
OUTPUT = Path("visualizations/qwen-concept-relay.html")
DATA_MARKER = "__QWEN_RELAY_DATA__"


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
                "continuation": row["continuation"][:240],
                "model_top10": row["model_top10"],
                "summary": row["summary"],
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
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    template = TEMPLATE.read_text()
    assert template.count(DATA_MARKER) == 1
    SUMMARY.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    OUTPUT.write_text(template.replace(DATA_MARKER, encoded))
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
