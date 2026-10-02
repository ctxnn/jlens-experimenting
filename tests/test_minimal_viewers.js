// DOM-fixture regressions only. These do not establish rendered layout or accessibility.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
class Element {
  constructor(tag, attrs = {}, width = 970) {
    this.tagName = tag; this.id = attrs.id || ''; this.attributes = attrs; this.children = [];
    this.handlers = {}; this.style = {}; this.innerHTML = ''; this.textContent = '';
    this.hidden = Object.hasOwn(attrs, 'hidden'); this.value = attrs.value || '';
    this.clientWidth = width;
    this.dataset = Object.fromEntries(Object.entries(attrs).filter(([k]) => k.startsWith('data-'))
      .map(([k, v]) => [k.slice(5), v]));
  }
  setAttribute(k, v) {this.attributes[k] = String(v);}
  addEventListener(k, v) {this.handlers[k] = v;}
  append(node) {node.parent = this; this.children.push(node);}
  add(node) {this.append(node);}
  remove() {this.parent.children.splice(this.parent.children.indexOf(this), 1);}
  get lastChild() {return this.children.at(-1);}
  getBoundingClientRect() {return {width: this.clientWidth, left: 0};}
  scrollIntoView() {this.scrolled = true;}
}
function load(name, width) {
  const html = fs.readFileSync(path.join(root, 'visualizations', name), 'utf8');
  const markup = html.split('<script>')[0];
  const nodes = [...markup.matchAll(/<([a-z][a-z0-9:-]*)\b([^>]*)>/g)].map(m => {
    const attrs = Object.fromEntries([...m[2].matchAll(/([\w:-]+)(?:="([^"]*)")?/g)]
      .map(a => [a[1], a[2] ?? '']));
    return new Element(m[1], attrs, width);
  });
  const ids = Object.fromEntries(nodes.filter(n => n.attributes.id).map(n => [n.attributes.id, n]));
  assert.equal(Object.keys(ids).length, nodes.filter(n => n.attributes.id).length, 'Unique element IDs');
  const match = selector => nodes.filter(n => selector.startsWith('.')
    ? (n.attributes.class || '').split(/\s+/).includes(selector.slice(1))
    : Object.hasOwn(n.attributes, selector.slice(1, -1)));
  const context = {
    document: {getElementById: id => {assert(ids[id], id); return ids[id];},
      querySelectorAll: match, createElement: tag => new Element(tag),
      createElementNS: (_, tag) => new Element(tag), documentElement: {}},
    ResizeObserver: class {observe() {}},
    Option: class extends Element {constructor(label, value) {super('option', {value}); this.textContent = label;}},
    getComputedStyle: () => ({getPropertyValue: key => ({'--cyan': '#5de4d2', '--yellow': '#ffd45c',
      '--line': '#303030', '--muted': '#aaa', '--text': '#e7e7e7'})[key]})
  };
  vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  return {html, ids, match, context};
}
for (const width of [970, 354]) {
  const relay = load('qwen-concept-relay.html', width);
  const {ids, match, context} = relay;
  const data = vm.runInNewContext('DATA', context);
  const checkView = view => {
    for (const v of ['trace', 'map']) {
      assert.equal(ids[v + '-view'].hidden, v !== view);
      assert.equal(ids['show-' + v].attributes['aria-pressed'], String(v === view));
    }
  };
  checkView('trace');
  ids['show-map'].handlers.click(); checkView('map');
  for (const row of data.results) {
    ids['prompt-select'].value = row.id; ids['prompt-select'].handlers.change();
    assert.equal(ids['prompt-text'].textContent, row.prompt);
    assert.equal(ids['readout-prefix'].textContent, row.readout_prefix);
    checkView('map');
    for (const lens of match('[data-lens]')) {
      lens.handlers.click(); assert.equal(lens.attributes['aria-pressed'], 'true');
      const paths = ids.trace.children.filter(n => n.tagName === 'path');
      assert(paths.length <= 3);
      for (const p of paths) assert(!/NaN|Infinity/.test(p.attributes.d));
    }
  }
  for (const filter of match('[data-filter]')) {
    filter.handlers.click();
    const count = vm.runInNewContext('filteredRows().length', context);
    assert.equal(ids['relay-map'].children.filter(n => n.attributes.class === 'row-hit').length, count);
  }
  match('[data-filter]').find(n => n.dataset.filter === 'all').handlers.click();
  const rowButton = ids['relay-map'].children.find(n => n.attributes.class === 'row-hit');
  rowButton.handlers.click(); checkView('trace'); // A map row returns to its visible trace.
  assert(ids['trace-heading'].scrolled);
  ids['show-map'].handlers.click(); checkView('map');

  const atlas = load('jlens-concept-atlas.html', width);
  const prompts = vm.runInNewContext('prompts', atlas.context);
  for (const button of atlas.match('[data-view]')) {
    button.handlers.click();
    for (const view of atlas.match('.view'))
      assert.equal(view.hidden, view.attributes.id !== 'view-' + button.dataset.view);
  }
  for (const [key, prompt] of Object.entries(prompts)) {
    const selector = atlas.match('.prompt-select')[0]; selector.value = key; selector.handlers.change();
    for (const [index, layer] of [8, 16, 24, 30].entries()) {
      const slider = atlas.match('.layer-range')[0]; slider.value = String(index); slider.handlers.input();
      assert.equal(atlas.ids['prompt-text'].textContent, prompt.text);
      assert.equal(atlas.ids['branch-overlap'].textContent, Math.round(prompt.layers[layer].o * 100) + '%');
      for (const id of ['j-tokens', 'l-tokens', 'rank-list', 'braid'])
        assert(!/NaN|Infinity|undefined/.test(atlas.ids[id].innerHTML), key + ' / ' + id);
    }
  }
}
console.log('Minimal viewers: all 63 relay prompts, 5 atlas families, view buttons, filters and layer controls pass at two fixture widths (not rendered QA).');
