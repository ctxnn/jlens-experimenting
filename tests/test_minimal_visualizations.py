"""Presentation contracts and archive fidelity, not browser-rendered QA."""
from hashlib import sha256
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

INSPECTOR_SCRIPTS = {
    "artifacts/qwen_ascii_face_slice.html": [
        "8a18886aedceabc68d49be465ccf4e3ae8101a50825af3acc9f63d2e72e301b0",
        "83de953bc6829b7c296ad182ad3647f693b9b16329abe790c811f58f6e69198b",
        "880a3ceb76fb1e0b65d800cc9549463a97f701bf18d223381411b51a4364412c"
    ],
    "artifacts/qwen_spider_slice.html": [
        "8a18886aedceabc68d49be465ccf4e3ae8101a50825af3acc9f63d2e72e301b0",
        "3734b5177ceaa0e01d8f13267038745ca3849f1703281eff0badd7375abe7a6e",
        "880a3ceb76fb1e0b65d800cc9549463a97f701bf18d223381411b51a4364412c"
    ]
}

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")

class MinimalVisualizationTest(unittest.TestCase):
    def test_inspector_data_and_renderer_are_unchanged(self):
        for filename, expected in INSPECTOR_SCRIPTS.items():
            source = Path(filename).read_text()
            actual = [sha256(body.encode()).hexdigest()
                      for body in re.findall(r"<script[^>]*>([\s\S]*?)</script>", source)]
            self.assertEqual(actual, expected, filename)
            self.assertIn("Minimal inspection UI", source)

    def test_custom_viewers_have_no_decorative_gradients_or_shadows(self):
        for filename in ("qwen-concept-relay.html", "jlens-concept-atlas.html",
                         "qwen-iteration2.html", "iteration2/index.html"):
            source = Path("visualizations", filename).read_text()
            css = re.search(r"<style>([\s\S]*?)</style>", source).group(1)
            self.assertNotIn("gradient(", css, filename)
            self.assertNotIn("box-shadow", css, filename)
            self.assertIn("#121212", css, filename)
            self.assertIn("font-size:17px", css, filename)

    def test_directory_links_resolve_and_only_gallery_first_case_starts_open(self):
        parser = Links()
        parser.feed(Path("visualizations/index.html").read_text())
        self.assertEqual(len(parser.links), 6)
        for link in parser.links:
            self.assertTrue((Path("visualizations") / link).resolve().is_file(), link)
        gallery = Path("visualizations/iteration2/index.html").read_text()
        cases = re.findall(r'<details class="example"[^>]*>', gallery)
        self.assertEqual(len(cases), 40)
        self.assertIn(" open", cases[0])
        self.assertTrue(all(" open" not in case for case in cases[1:]))
        self.assertEqual(gallery.count("<img "), 80)

    def test_all_depth_export_coordinates_match_raw_ranks(self):
        cases = []
        for filename in ("qwen_iteration2.json", "qwen_iteration2_hidden.json"):
            cases.extend(json.loads(Path("artifacts/iteration2", filename).read_text())["cases"])
        ns = {"s": "http://www.w3.org/2000/svg"}
        for case in cases:
            figure = ET.parse(Path("visualizations/iteration2", case["id"] + "-depth.svg"))
            lines = figure.findall(".//s:polyline", ns)
            self.assertEqual(len(lines), 4)
            for concept, line in zip(("target", "decoy", "answer", "alternative"), lines):
                expected = []
                for row in case["readouts"]:
                    best = row["j_lens"]["measurements"][concept]["best"]
                    if best is not None:
                        x = 72 + 760 * row["layer"] / 30
                        y = 140 + 270 * min(6, math.log10(max(1, best["rank"]))) / 6
                        expected.append(f"{x:.2f},{y:.2f}")
                self.assertEqual(line.attrib["points"], " ".join(expected), case["id"] + "/" + concept)

if __name__ == "__main__":
    unittest.main()
