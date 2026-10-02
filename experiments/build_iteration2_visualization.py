"""Deterministic offline viewer + blog-ready vector figures from archived GPU data."""
from __future__ import annotations

import argparse
from hashlib import sha256
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"target": "#63ddcb", "decoy": "#8ba9ff", "answer": "#f7ce68", "alternative": "#dc99ea"}
CONDITION_COLORS = {"concept_swap": "#63ddcb", "no_op": "#92a0b5", "unrelated_swap": "#8ba9ff",
                    "answer_swap": "#f7ce68", "random_norm_matched": "#dc99ea"}


def validate(report):
    cases = report["cases"]
    if report["model"] != "Qwen/Qwen3.5-4B" or report["lens_training_prompts"] != 1000:
        raise ValueError("Unexpected model or published lens")
    planned = {c["id"] for c in report["protocol"]["cases"]}
    protocols = [ROOT / "experiments/iteration2_prompts.json", ROOT / "experiments/iteration2_hidden_prompts.json"]
    if report["protocol_sha256"] not in {sha256(p.read_bytes()).hexdigest() for p in protocols if p.exists()}:
        raise ValueError("Frozen protocol does not match the archived run")
    if report["runner_sha256"] != sha256((ROOT / "experiments/qwen_iteration2.py").read_bytes()).hexdigest():
        raise ValueError("Runner does not match the archived run")
    if len({c["id"] for c in cases}) != len(cases) or not {c["id"] for c in cases} <= planned:
        raise ValueError("Duplicate or unplanned case")
    for case in cases:
        if case["readout_prefix"] != case["prompt"] + report["protocol"]["readout_suffix"]:
            raise ValueError("Readout prefix was not fixed in advance")
        if [r["layer"] for r in case["readouts"]] != report["source_layers"]:
            raise ValueError("Incomplete layer readouts")
        if case["final_readout_max_abs_logit_error"] > .25:
            raise ValueError("Final readout parity failed")
        for row in [case["clean"]] + [r[lens] for r in case["readouts"] for lens in ("j_lens", "logit_lens")] + [r for r in case["interventions"] if r["available"]]:
            for measurement in row["measurements"].values():
                if measurement["mass"] is not None and not (0 <= measurement["mass"] <= 1.00001):
                    raise ValueError("Invalid token probability mass")
                if measurement["best"] is not None and measurement["best"]["rank"] < 1:
                    raise ValueError("Invalid rank")
                if measurement["variants"] and abs(sum(v["probability"] for v in measurement["variants"]) - measurement["mass"]) > 1e-7:
                    raise ValueError("Alias mass mismatch")
                variants = measurement["variants"]
                if len({v["id"] for v in variants}) != len(variants):
                    raise ValueError("Duplicate alias token would double-count mass")
                if variants and measurement["best"] != sorted(variants, key=lambda v: (v["rank"], -v["log_probability"]))[0]:
                    raise ValueError("Best-rank selection mismatch")
                for variant in variants:
                    if not math.isfinite(variant["log_probability"]) or abs(math.exp(variant["log_probability"]) - variant["probability"]) > 1e-7:
                        raise ValueError("Token probability/log-probability mismatch")
        if len(case["interventions"]) != 5 * len(report["protocol"]["intervention_layers"]):
            raise ValueError("Missing intervention or control records")
        for intervention in case["interventions"]:
            if intervention["available"] and intervention["condition"] == "no_op" and intervention["max_abs_logit_change"] > .001:
                raise ValueError("No-op changed the model")
            if intervention["available"]:
                a = intervention["measurements"]["answer"]["mass"]
                b = intervention["measurements"]["alternative"]["mass"]
                margin = b - a if a is not None and b is not None else None
                if margin != intervention["alternative_minus_original_mass"]:
                    raise ValueError("Margin mismatch")
                clean_margin = case["clean"]["alternative_minus_original_mass"]
                delta = margin - clean_margin if margin is not None and clean_margin is not None else None
                if delta != intervention["margin_change"]:
                    raise ValueError("Effect delta mismatch")
        for row in case["readouts"]:
            if "target_rank_by_position" in row and len(row["target_rank_by_position"]) != len(case["prefix_token_ids"]):
                raise ValueError("Prompt-position heatmap alignment mismatch")
    return {"completed": len(cases), "planned": len(planned),
            "available_concept_swaps": sum(r["available"] for c in cases for r in c["interventions"] if r["condition"] == "concept_swap"),
            "unavailable_interventions": sum(not r["available"] for c in cases for r in c["interventions"])}


def rank_path(case, concept, x=72, y=140, width=760, height=270):
    points = []
    for row in case["readouts"]:
        best = row["j_lens"]["measurements"][concept]["best"]
        if best is None:
            continue
        rank = max(1, best["rank"])
        px = x + width * row["layer"] / max(r["layer"] for r in case["readouts"])
        py = y + height * min(6, math.log10(rank)) / 6
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def trace_svg(case):
    e = html.escape
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 560" role="img"><title>{e(case["id"])}: measured J-Lens token ranks across depth</title>',
              '<rect width="960" height="560" fill="#121212"/>',
              '<g font-family="system-ui, sans-serif" fill="#e7e7e7">',
              f'<text x="36" y="42" font-size="17">{e(case["target"])} → {e(case["answer"])} · Qwen3.5-4B</text>',
              f'<text x="36" y="76" font-size="13" fill="#aaa">{e(case["prompt"])}</text>',
              '<text x="36" y="108" font-size="12" fill="#aaa">J-Lens rank · lower is more readable · fixed supplied trailing space</text>']
    for power in range(7):
        yy = 140 + 270 * power / 6
        label = str(10**power) if power < 4 else f'{10**(power-3)}k'
        chunks += [f'<path d="M72 {yy}H832" stroke="#303030"/>',
                   f'<text x="60" y="{yy+4}" text-anchor="end" font-size="11" fill="#aaa">{label}</text>']
    for layer in (0, 5, 10, 15, 20, 25, 30):
        xx = 72 + 760 * layer / 30
        chunks.append(f'<text x="{xx}" y="434" text-anchor="middle" font-size="11" fill="#aaa">{layer}</text>')
    for concept in ("target", "decoy", "answer", "alternative"):
        chunks.append(f'<polyline points="{rank_path(case, concept)}" fill="none" stroke="{COLORS[concept]}" stroke-width="3"/>')
    labels = [("target", case["target"]), ("decoy", case["decoy"]), ("answer", case["answer"]), ("alternative", case["alternative"])]
    for index, (concept, label) in enumerate(labels):
        xx = 72 + index * 205
        if not case["token_groups"][concept]:
            label += " [n/a]"
        chunks += [f'<path d="M{xx} 467h18" stroke="{COLORS[concept]}" stroke-width="3"/>',
                   f'<text x="{xx+25}" y="471" font-size="12">{e(label)} ({concept})</text>']
    chunks += ['<text x="72" y="512" font-size="12" fill="#aaa">Lens readouts are not literal thoughts or causal evidence.</text>',
               '<text x="72" y="535" font-size="12" fill="#aaa">Layers 0–30, post-block; strongest rank across declared single-token surfaces.</text>', '</g></svg>']
    return "".join(chunks)


def intervention_svg(case, layer=24):
    e = html.escape
    conditions = ["clean", "concept_swap", "no_op", "unrelated_swap", "answer_swap", "random_norm_matched"]
    title = f'{case["target"]} ↔ {case["decoy"]} · layer {layer}'
    chunks = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 650" role="img" aria-labelledby="figure-title figure-description">',
              f'<title id="figure-title">{e(title)}: original and alternative next-token alias mass</title>',
              '<desc id="figure-description">Bars show one prefix forward, not whole-answer probabilities. Saved greedy continuations below use a sustained concept edit and can remain unchanged or produce an unintended answer.</desc>',
              '<rect width="960" height="650" fill="#121212"/><g font-family="system-ui, sans-serif" fill="#e7e7e7">',
              f'<text x="36" y="42" font-size="17">{e(title)}</text>',
              f'<text x="36" y="76" font-size="13" fill="#aaa">{e(case["prompt"])}</text>',
              '<text x="36" y="108" font-size="12" fill="#aaa">Next-token alias mass, not whole-answer probability · fixed 0–100% axis</text>']
    for index, condition in enumerate(conditions):
        row = case["clean"] if condition == "clean" else next((r for r in case["interventions"] if r["layer"] == layer and r["condition"] == condition and r["available"]), None)
        yy = 150 + index * 54
        chunks.append(f'<text x="36" y="{yy+13}" font-size="12">{e(condition.replace("_", " "))}</text>')
        if row is None:
            chunks.append(f'<text x="265" y="{yy+13}" font-size="12" fill="#aaa">unavailable</text>')
            continue
        for offset, concept in enumerate(("answer", "alternative")):
            mass = row["measurements"][concept]["mass"]
            by = yy + offset * 19
            if mass is None:
                chunks.append(f'<text x="265" y="{by+11}" font-size="11" fill="#aaa">not single-token measurable</text>')
                continue
            chunks += [f'<rect x="265" y="{by}" width="500" height="13" fill="#303030"/>',
                       f'<rect x="265" y="{by}" width="{500*mass:.3f}" height="13" fill="{COLORS[concept]}"/>',
                       f'<text x="780" y="{by+11}" font-size="11">{e("<0.01%" if 0 < mass < .0001 else f"{mass*100:.2f}%")}</text>']
    original = case["answer"]
    if case["token_groups"]["answer"] and not any(t["surface"].strip().casefold() == original.casefold() for t in case["token_groups"]["answer"]):
        original += " (word aliases only)"
    concept_row = next((r for r in case["interventions"]
                        if r["layer"] == layer and r["condition"] == "concept_swap"), None)
    edited_generation = concept_row.get("generation") if concept_row and concept_row["available"] else None

    def excerpt(generation):
        text = " ".join(generation["text"].split())
        return text if len(text) <= 96 else text[:95] + "…"

    clean_text = excerpt(case["clean_generation"])
    if edited_generation:
        edited_text = '“' + excerpt(edited_generation) + '”'
        generation_note = f'Continuations: up to 12 greedy tokens; concept edit sustained at layer {layer} on every decoding forward.'
    elif concept_row and not concept_row["available"]:
        edited_text = "unavailable (no single-token concept pair)"
        generation_note = "Clean continuation: up to 12 greedy tokens. No edited continuation is available for this concept pair."
    else:
        edited_text = "not recorded at this layer"
        generation_note = "Clean continuation: up to 12 greedy tokens. Edited generation is recorded only at the fixed generation layer."
    chunks += [f'<text x="265" y="498" fill="{COLORS["answer"]}" font-size="12">Original: {e(original)}</text>',
               f'<text x="580" y="498" fill="{COLORS["alternative"]}" font-size="12">Alternative: {e(case["alternative"])}</text>',
               '<text x="36" y="528" font-size="12" fill="#aaa">Bars: one forward over the supplied prefix. A positive margin change is not an answer-swap success.</text>',
               f'<text x="36" y="555" font-size="12">Clean continuation: “{e(clean_text)}”</text>',
               f'<text x="36" y="578" font-size="12">Concept-edit continuation: {e(edited_text)}</text>',
               f'<text x="36" y="605" font-size="12" fill="#aaa">{e(generation_note)}</text>',
               '<text x="36" y="628" font-size="12" fill="#aaa">Unclamped token-coordinate swap, all prefix positions; not a paper replication.</text>', '</g></svg>']
    return "".join(chunks)


def gallery_html(cases):
    e = html.escape
    figures = []
    for index, case in enumerate(cases):
        figures.append(f'<details class="example"{" open" if index == 0 else ""}><summary>{e(case["id"])} <small>{e(case["run_kind"])}</small></summary>'
                       f'<p>{e(case["prompt"])}</p><div class="pair">'
                       f'<a href="{e(case["id"])}-depth.svg"><img loading="lazy" src="{e(case["id"])}-depth.svg" alt="Measured depth ranks for {e(case["id"])}"/></a>'
                       f'<a href="{e(case["id"])}-intervention.svg"><img loading="lazy" src="{e(case["id"])}-intervention.svg" alt="Clean and controlled layer-24 output masses for {e(case["id"])}"/></a>'
                       '</div></details>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>J-Lens · figure gallery</title><style>body{background:#121212;color:#e7e7e7;font:14px/1.55 system-ui,sans-serif;margin:0}main{max-width:1200px;margin:auto;padding:28px 32px 40px}header{display:flex;justify-content:space-between;align-items:baseline;gap:20px;margin-bottom:28px}h1{font-size:17px;font-weight:600;margin:0}a{color:inherit;text-decoration-color:#666;text-underline-offset:4px}a:focus-visible,summary:focus-visible{outline:2px solid #63ddcb;outline-offset:4px}small,p{color:#aaa}small{font-size:12px;margin-left:12px}p{font-size:12px}details{border-top:1px solid #303030;padding:13px 0}summary{cursor:pointer}summary:hover{color:#fff}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:16px 0}img{width:100%;display:block}.example p{font:13px/1.6 ui-monospace,monospace;max-width:900px}footer{margin-top:20px;font-size:11px;color:#aaa}@media(max-width:800px){.pair{grid-template-columns:1fr}main{padding:20px 18px 30px}small{display:block;margin:3px 0 0}}</style></head><body><main>'
        f'<header><h1>Figures</h1><a href="../qwen-iteration2.html">Reader</a></header><p>{len(cases)} measured prompts · {len(cases)*2} SVGs. Open an example, then select a figure to save it.</p>'
        + ''.join(figures)
        + '<details><summary>Data &amp; method</summary><p>All measured examples are retained. Traces are lens readouts, not literal thoughts; the intervention is an unclamped Qwen pilot, not a paper replication.</p><p>Intervention bars are next-token alias mass, not whole-answer probability. The saved continuations use a sustained layer-24 concept edit, unlike the single prefix forward behind the bars. A positive margin change need not produce the intended alternative answer.</p><p><a href="../../iteration2.md">Method and limitations</a></p></details><footer>Lens readouts are not literal thoughts.</footer></main></body></html>'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/iteration2/qwen_iteration2.json")
    parser.add_argument("--hidden-input", type=Path, default=ROOT / "artifacts/iteration2/qwen_iteration2_hidden.json")
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw)
    summary = validate(report)
    report["archive_sha256"] = sha256(raw).hexdigest()
    report["source_archives"] = [{"kind": "explicit-entity calibration", "path": "../artifacts/iteration2/" + args.input.name,
        "sha256": sha256(raw).hexdigest(), "protocol_sha256": report["protocol_sha256"],
        "started_at": report["started_at"], "runtime": report["runtime"], "validation": summary}]
    report["cases"] = [{**c, "run_kind": "explicit-entity calibration"} for c in report["cases"]]
    if args.hidden_input.exists():
        hidden_bytes = args.hidden_input.read_bytes()
        hidden = json.loads(hidden_bytes)
        hidden_summary = validate(hidden)
        for key in ("model_revision", "lens_revision", "source_layers", "runner_sha256"):
            if report[key] != hidden[key]:
                raise ValueError("Paired passes do not share the same instrument: " + key)
        for key in ("readout_suffix", "intervention_layers", "generation_layer", "generation_tokens", "seed"):
            if report["protocol"][key] != hidden["protocol"][key]:
                raise ValueError("Paired passes do not share the same measurement protocol: " + key)
        report["source_archives"].append({"kind": "hidden-description follow-up", "path": "../artifacts/iteration2/" + args.hidden_input.name,
            "sha256": sha256(hidden_bytes).hexdigest(), "protocol_sha256": hidden["protocol_sha256"],
            "started_at": hidden["started_at"], "runtime": hidden["runtime"], "validation": hidden_summary})
        report["cases"] = [{**c, "run_kind": "hidden-description follow-up"} for c in hidden["cases"]] + report["cases"]
        report["protocol"]["cases"] = hidden["protocol"]["cases"] + report["protocol"]["cases"]
        summary = {key: summary[key] + hidden_summary[key] for key in summary}
        explicit_by_id = {c["id"]: c for c in report["cases"] if c["run_kind"] == "explicit-entity calibration"}
        for c in hidden["cases"]:
            pair = explicit_by_id[c["paired_explicit_id"]]
            for key in ("target", "decoy", "answer", "alternative", "answer_aliases", "alternative_aliases"):
                if c[key] != pair[key]:
                    raise ValueError("Paired contrasts differ: " + c["id"])
    report["validation"] = summary
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    template = (ROOT / "visualizations/qwen-iteration2.template.html").read_text()
    output = ROOT / "visualizations/qwen-iteration2.html"
    output.write_text(template.replace("__ITERATION2_DATA__", encoded))
    figure_dir = ROOT / "visualizations/iteration2"
    figure_dir.mkdir(exist_ok=True)
    for case in report["cases"]:
        (figure_dir / f'{case["id"]}-depth.svg').write_text(trace_svg(case))
        (figure_dir / f'{case["id"]}-intervention.svg').write_text(intervention_svg(case, report["protocol"]["generation_layer"]))
    (figure_dir / "index.html").write_text(gallery_html(report["cases"]))
    manifest = {"schema_version": 1, "source_archives": report["source_archives"], "validation": summary,
        "runner_sha256": report["runner_sha256"], "model_revision": report["model_revision"],
        "lens_revision": report["lens_revision"], "svg_figures": len(report["cases"]) * 2,
        "html_sha256": sha256(output.read_bytes()).hexdigest(),
        "rendered_qa": {"svg": "Builder does not perform visual QA; see iteration2.md for sample inspection",
                        "interactive": "Browser rendering not verified"},
        "known_limits": ["Designed sequential follow-up; not held out", "Unclamped single-layer token edits",
                        "One random seed per case/layer", "Single-token alias mass, not whole-answer probability",
                        "No-op is gated by concept-pair availability", "Supplied trailing space can hurt clean completion"],
        "runs": []}
    for source in report["source_archives"]:
        cases = [c for c in report["cases"] if c["run_kind"] == source["kind"]]
        eligible = [selected for c in cases for selected in c["interventions"]
                    if selected["layer"] == 24 and selected["condition"] == "concept_swap" and
                    selected["available"] and selected["margin_change"] is not None]
        manifest["runs"].append({"kind": source["kind"], "completed": len(cases),
            "planned": source["validation"]["planned"], "layer24_measurable_effects": len(eligible),
            "layer24_positive_effects": sum(x["margin_change"] > 0 for x in eligible),
            "layer24_negative_effects": sum(x["margin_change"] < 0 for x in eligible),
            "maximum_final_readout_logit_error": max(c["final_readout_max_abs_logit_error"] for c in cases),
            "maximum_no_op_logit_change": max((x["max_abs_logit_change"] for c in cases
                for x in c["interventions"] if x["condition"] == "no_op" and x["available"]), default=None)})
    (ROOT / "artifacts/iteration2/run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps({"output": str(output), "figures": len(report["cases"]) * 2, **summary}))


if __name__ == "__main__":
    main()
