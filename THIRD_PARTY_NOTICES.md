# Third-party code and notices

These notices apply to the identified bundled third-party code, not as a new blanket
license for this repository's original code, documentation, or measurements.

## Official Jacobian-lens token inspectors

`artifacts/qwen_spider_slice.html` and `artifacts/qwen_ascii_face_slice.html` contain the
official visualization renderer from [Anthropic's jacobian-lens](https://github.com/anthropics/jacobian-lens/tree/581d398613e5602a5af361e1c34d3a92ea82ba8e),
pinned to commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

- Copyright 2026 Anthropic PBC.
- License: Apache-2.0; the full upstream text is in
  [licenses/jacobian-lens-APACHE-2.0.txt](licenses/jacobian-lens-APACHE-2.0.txt).
- Local modifications: Qwen experiment titles/metadata and measured payloads, followed
  by minimal UI styling. The upstream renderer logic is retained. Each modified HTML
  carries a modification notice and the full license text for standalone distribution.

## Embedded D3

Both official inspectors bundle D3 v7.9.0.

- Copyright 2010–2023 Mike Bostock.
- License: ISC; the full upstream copyright, permission, and disclaimer text is in
  [licenses/d3-ISC.txt](licenses/d3-ISC.txt) and embedded in each standalone inspector.
- Source: [D3 v7.9.0](https://github.com/d3/d3/tree/v7.9.0).

## Model, lens weights, and corpus

Qwen weights, the published lens weights, and any upstream fitting corpus are not bundled
in this repository. Their own terms apply. The pinned source references and runtime
provenance are recorded in [iteration2.md](iteration2.md) and the archived reports.

When distributing either inspector separately, retain its embedded notices and license
texts. For the repository bundle, retain this file and the accompanying `licenses/` files.
