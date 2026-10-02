// Source-level + DOM-fixture checks, not browser-rendered or accessibility QA.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'visualizations/qwen-iteration2.html'), 'utf8');
assert(!/class="(?:card|eyebrow|lead)"/.test(html), 'Viewer should not restore decorative cards or a hero.');
for (const id of ['continuations', 'all-examples', 'evidence']) {
  const details = html.match(new RegExp(`<details\\b[^>]*\\bid="${id}"[^>]*>`));
  assert(details, id);
  assert(!/\bopen\b/.test(details[0]), id + ' starts collapsed');
}
const data = JSON.parse(html.match(/<script type="application\/json" id="dataset">([\s\S]*?)<\/script>/)[1]);
const code = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const pure = {module: {exports: {}}};
vm.runInNewContext(code, pure);
const graphs = pure.module.exports;
assert.equal(graphs.esc('<script>&"'), '&lt;script&gt;&amp;&quot;');
assert.equal(graphs.signed(null), 'unavailable');
assert.equal(graphs.pct(null), 'unavailable');
assert.equal(graphs.pct(.000001), '<0.01%');
assert.equal(graphs.signed(.04410634712881101), '+4.41 pp');
for (const c of data.cases) {
  for (const l of data.source_layers) {
    for (const lens of ['j_lens', 'logit_lens']) {
      const svg = graphs.depthChart(c, lens, l);
      assert(!/NaN|Infinity|undefined/.test(svg), c.id);
      assert(svg.includes('<title>'));
    }
  }
  for (const l of data.protocol.intervention_layers) {
    const svg = graphs.effectChart(c, l);
    assert(!/NaN|Infinity|undefined/.test(svg), c.id);
    assert(!/NaN|Infinity|undefined/.test(graphs.effectChart(c, l, true)), c.id);
  }
  const heat = graphs.heatChart(c);
  assert(!/NaN|Infinity|undefined/.test(heat), c.id);
  assert.equal((heat.match(/<rect /g) || []).length,
               data.protocol.intervention_layers.length * c.prefix_tokens.length);
}
const elements = {};
for (const id of [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1])) {
  elements[id] = {innerHTML: '', textContent: '', value: '', hidden: true,
    handlers: {}, attributes: {},
    addEventListener(name, callback) {this.handlers[name] = callback;},
    setAttribute(name, value) {this.attributes[name] = String(value);},
    focus() {this.focused = true;}};
}
elements.dataset.textContent = JSON.stringify(data);
const fixture = {document: {getElementById: id => elements[id]}};
vm.runInNewContext(code, fixture);
function assertView(view) {
  for (const name of ['depth', 'position', 'intervention']) {
    assert.equal(elements[name + '-view'].hidden, name !== view);
    assert.equal(elements['view-' + name].attributes['aria-pressed'], String(name === view));
  }
  assert.equal(elements['readout-switch'].hidden, view !== 'depth');
}
assertView('depth');
for (let i = 0; i < data.cases.length; i++) {
  const view = ['depth', 'position', 'intervention'][i % 3];
  elements['view-' + view].handlers.click();
  assertView(view);
  elements.example.value = String(i);
  elements.example.handlers.change();
  assertView(view); // Prompt changes retain the selected view.
  assert.equal(elements.prompt.textContent, data.cases[i].readout_prefix);
  assert.equal(elements['clean-output'].textContent, data.cases[i].clean_generation.text);
  assert(elements['depth-download'].href.includes(data.cases[i].id));
  if (data.cases[i].paired_explicit_id) {
    elements['paired-example'].handlers.click();
    const pair = data.cases.find(c => c.id === data.cases[i].paired_explicit_id);
    assert.equal(elements.prompt.textContent, pair.readout_prefix);
    elements['paired-example'].handlers.click();
    assert.equal(elements.prompt.textContent, data.cases[i].readout_prefix);
    assertView(view);
  }
  elements['include-answer'].handlers.click();
  assert.equal(elements['include-answer'].attributes['aria-pressed'], 'true');
  assert(elements['condition-legend'].innerHTML.includes('Direct answer'));
  elements['include-answer'].handlers.click();
  assert.equal(elements['include-answer'].attributes['aria-pressed'], 'false');
  for (const l of data.protocol.intervention_layers) {
    elements['intervention-layer'].value = String(l);
    elements['intervention-layer'].handlers.change();
    assert(elements['intervention-name'].textContent.includes('layer ' + l));
  }
  for (const lens of ['jlens', 'logit']) {
    elements[lens].handlers.click();
    assert.equal(elements[lens].attributes['aria-pressed'], 'true');
    for (const layer of data.source_layers) {
      elements['depth-layer'].value = String(layer);
      elements['depth-layer'].handlers.input();
      assert.equal(elements['depth-label'].textContent, layer);
      for (const id of ['ranks', 'top-tokens', 'heatmap', 'effects', 'answer-bars']) {
        assert(!/NaN|Infinity|undefined/.test(elements[id].innerHTML), id + ' / ' + data.cases[i].id);
      }
    }
  }
  const generation = graphs.selectedIntervention(data.cases[i], 24, 'concept_swap')?.generation;
  assert.equal(elements['edited-output'].textContent,
    generation ? generation.text : 'Unavailable: no valid concept-vector pair');
  assertView(view); // Slider/readout changes do not make the other panels visible.
}
for (const view of ['depth', 'position', 'intervention']) {
  elements['view-' + view].handlers.click();
  assertView(view);
}
const bad = {...fixture, document: {getElementById: id => id === 'dataset'
  ? {textContent: '__ITERATION2_DATA__'} : elements[id]}};
vm.runInNewContext(code, bad);
assert.equal(elements.notice.hidden, false);
assert(elements.notice.textContent.includes('source template'));
console.log(`Iteration 2: ${data.cases.length} cases, chart paths and all selector states pass (DOM fixture; not rendered QA).`);
