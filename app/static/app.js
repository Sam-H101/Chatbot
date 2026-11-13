/* The page's core: helpers, theme, Claude key, model pickers, instructions, usage, unit fixes, the
   upload queue and its jobs, what questions search (scope), sources, Chats and the chat itself.
   The documents view is docs.js; the markdown preview md.js. Classic scripts sharing one global
   scope; nothing starts before DOMContentLoaded (start, at the end), when all three have run. */
const $ = id => document.getElementById(id);
const store = { get: k => { try { return localStorage.getItem(k); } catch { return null; } },
                set: (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch {} } };
const key = $('key');
key.value = store.get('claudeKey') || '';
let busy = false;

// With a password set (APP_PASSWORD), the server asks every call that changes something for
// this header, which a page on another site can't add (app/auth.py).
const OWN = { 'X-Requested-With': 'fetch' };
// A call made once the session has ended (Log out in another tab, a new password, SESSION_DAYS
// passed) gets 401 {login: true}: the page goes to the login page, which says so, and the call
// never settles, so nothing shows an error on the way out. A question being sent (until the
// server has kept it) or typed, and a new instruction, go along in this tab's sessionStorage
// and are back in their boxes after logging in. A 401 without `login` is about the Claude key.
const loginGone = (status, data) => status === 401 && data?.login === true;
let asking = '';      // the question being sent, until its start event says the server kept it
let leftAt = 0;  // when the page set off for the login page (Log out, or a call before this one)
// A page still here 10 s after the latest start (the server never answered, the browser stopped
// loading) or back from the login page (Back) goes there again at the next call that finds the
// session ended. A time rather than a timer, so a second Log out gets its own full 10 s.
const leave = () => { leftAt = Date.now(); };
addEventListener('pageshow', e => { if (e.persisted) leftAt = 0; });
const toLogin = () => {
  if (Date.now() - leftAt > 10000) {
    leave();
    const question = [asking, $('question').value.trim()].filter(Boolean).join('\n\n');
    try { sessionStorage.setItem('draft', JSON.stringify({ question, rule: $('newrule').value.trim() })); } catch {}
    location.assign('/login?ended=1');
  }
  return new Promise(() => {});
};

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { 'X-API-Key': key.value, ...OWN, ...(opts.headers || {}) } });
  const data = res.status === 204 ? null : await res.json().catch(() => ({
    error: res.status === 413 ? 'File is too large to upload' : `${res.status} ${res.statusText}` }));
  if (loginGone(res.status, data)) return toLogin();
  if (!res.ok) throw Object.assign(new Error(data?.error || res.statusText), { status: res.status, data });
  return data;
}
// A JSON call that changes something.
const send = (path, body, method = 'POST') =>
  api(path, { method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

// [12] -> 'p. 12'; [12, 13, 14, 20] -> 'pp. 12-14, 20'.
function pagesText(pages) {
  const runs = [];
  for (const p of pages) {
    const last = runs[runs.length - 1];
    if (last && p === last[1] + 1) last[1] = p; else runs.push([p, p]);
  }
  return (pages.length > 1 ? 'pp. ' : 'p. ') + runs.map(([a, b]) => a === b ? a : `${a}-${b}`).join(', ');
}
const pageSpan = (a, b) => Array.from({ length: b - a + 1 }, (_, i) => a + i);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
// An icon from the sprite in index.html.
function icon(name, cls = '') {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('class', 'ic' + (cls ? ' ' + cls : ''));
  s.setAttribute('aria-hidden', 'true');
  const u = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  u.setAttribute('href', '#i-' + name);
  s.append(u);
  return s;
}
// A button: label (text or nodes), title, onclick; cls; aria-label for icon-only ones.
function btn(label, onclick, { cls = '', title = '', aria = '', icon: ic = '' } = {}) {
  const b = el('button', cls);
  b.type = 'button';
  if (ic) b.append(icon(ic));
  if (label != null && label !== '') { if (ic) b.append(el('span', '', label)); else b.append(label); }
  if (title) b.title = title;
  if (aria) b.setAttribute('aria-label', aria);
  if (onclick) b.onclick = onclick;
  return b;
}
// Bold names inside a line of text: frag('Filed in ', ['Receipts'], ' automatically').
function frag(...parts) {
  const f = document.createDocumentFragment();
  for (const p of parts) f.append(Array.isArray(p) ? el('b', '', p[0]) : p);
  return f;
}
const plural = (n, one, many = one + 's') => `${n.toLocaleString()} ${n === 1 ? one : many}`;
const isS = matchMedia('(max-width: 820px)');
const isL = matchMedia('(min-width: 1280px)');
const coarse = matchMedia('(pointer: coarse)');
const fine = matchMedia('(pointer: fine)');

// ---- theme, key ----
if (store.get('theme')) document.documentElement.dataset.theme = store.get('theme');
const darkQuery = matchMedia('(prefers-color-scheme: dark)');
const isDark = () => document.documentElement.dataset.theme
  ? document.documentElement.dataset.theme === 'dark' : darkQuery.matches;
$('theme').onclick = () => { store.set('theme', document.documentElement.dataset.theme = isDark() ? 'light' : 'dark'); };
// The dot is green when Claude can be used: a key typed here, or the server's own (/api/models).
function showKey() {
  const typed = !!key.value.trim(), server = !!MODELS.server_key;
  $('keydot').classList.toggle('set', typed || server);
  $('keybtn').title = typed ? 'Claude API key: yours, kept in this browser'
    : server ? 'Claude API key: the server has one (a key typed here is used instead)' : 'Claude API key: none yet';
}
key.oninput = () => { store.set('claudeKey', key.value); fillPickers(); };
const askKey = () => { $('settings').hidden = false; key.focus(); };
$('keybtn').onclick = () => { $('settings').hidden = !$('settings').hidden; if (!$('settings').hidden) key.focus(); };
$('keydone').onclick = () => $('settings').hidden = true;
$('keyclear').onclick = () => { key.value = ''; key.oninput(); key.focus(); };
document.addEventListener('click', e => {
  // The path as dispatched: still right when a handler has already removed the clicked button.
  const path = e.composedPath(), inside = sel => path.some(n => n.matches?.(sel));
  if (!inside('#settings, #keybtn')) $('settings').hidden = true;
  if (!inside('#chats, #chatsbtn, dialog')) { document.body.classList.remove('chatsopen'); syncChatsButton(); }
});
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape' || e.defaultPrevented) return;
  const settings = !$('settings').hidden;
  $('settings').hidden = true;
  if (document.querySelector('dialog[open]')) return;  // Escape closes the dialog only
  if (document.body.classList.contains('chatsopen')) {
    document.body.classList.remove('chatsopen');
    syncChatsButton();
    return;
  }
  if (!settings && view.mode === 'docs') docsEscape(e);
});

// ---- model pickers ----
// Values: 'auto', 'ollama:<model>' or 'claude:<model>'. Each picker remembers its choice.
let MODELS = { local: [], claude: [], auto: {}, default: 'auto' };
showKey();  // a typed key at once; the server's once /api/models says so
const PICKERS = { docModel: 'docModel', chatModel: 'chatModel', rereadModel: 'rereadModel' };
const claudeReady = () => MODELS.server_key || !!key.value.trim();
// "Claude reads every page", beside the pickers that read documents. Shown only for a
// Claude model; off, Claude reads only scanned pages and the text layer does the rest.
const PAGE_BOXES = { docModel: ['allPagesBox', 'allPages'], rereadModel: ['rereadAllBox', 'rereadAll'] };
const PAGES_TIP = 'Off: pages that have a text layer are used as they are, and only scanned pages go to Claude (cheaper). '
  + 'On: Claude reads every page as text and image, which helps with complex tables and layouts but costs more.';
function syncPageBoxes() {
  for (const [sel, [box]] of Object.entries(PAGE_BOXES)) $(box).hidden = !$(sel).value.startsWith('claude:');
}
const claudePages = sel => $(PAGE_BOXES[sel][1]).checked ? 'all' : 'scanned';
for (const [box, input] of Object.values(PAGE_BOXES)) {
  $(box).title = PAGES_TIP;
  $(input).onchange = e => {
    store.set('claudeAllPages', e.target.checked ? '1' : '0');
    for (const [, other] of Object.values(PAGE_BOXES)) $(other).checked = e.target.checked;
  };
}

function fillPicker(sel) {
  const want = sel.value || store.get(PICKERS[sel.id]) || MODELS.default || 'auto';
  sel.replaceChildren();
  const local = MODELS.auto.local || 'local model';
  sel.append(new Option(claudeReady() ? 'Auto (local, Claude if needed)' : 'Auto (local)', 'auto'));
  sel.title = claudeReady()
    ? `Auto uses ${local} and switches to ${MODELS.auto.claude} when Ollama fails or the prompt is too long for it.`
    : `Auto uses ${local}. Add a Claude API key to let it switch to Claude when needed.`;
  const g1 = el('optgroup');
  g1.label = MODELS.local_error ? 'This machine (Ollama not reachable)' : 'This machine';
  for (const n of MODELS.local) g1.append(new Option(n, 'ollama:' + n));
  const g2 = el('optgroup');
  g2.label = claudeReady() ? 'Claude API' : 'Claude API (needs your key)';
  for (const c of MODELS.claude) g2.append(new Option(c.label, 'claude:' + c.id));
  sel.append(g1, g2);
  sel.value = [...sel.options].some(o => o.value === want) ? want : 'auto';
}
const fillPickers = () => { Object.keys(PICKERS).forEach(id => fillPicker($(id))); syncPageBoxes(); showKey(); };
// "Read with Auto (local)": what the file drop overlay says will read a file.
const readerText = () => $('docModel').selectedOptions[0]?.textContent || 'Auto (local)';

// What the file picker offers, from the server (GET /api/models upload_types): never image/*,
// which would offer SVG and GIF. The server still takes any UTF-8 text file whatever its name.
const ACCEPT = '.pdf,.docx,.txt,.md,.markdown,.csv,.json,.html,.xml,.log,.py,.js,.jpg,.jpeg,.png,.webp,'
  + 'text/*,text/markdown,image/jpeg,image/png,image/webp';
function uploadTypes() {
  const heic = !!MODELS.upload_types?.heic;
  $('file').accept = ACCEPT + (heic ? ',.heic,.heif,image/heic,image/heif' : '');
  const hint = `PDF, Word, photos (JPG, PNG, WebP${heic ? ', HEIC' : ''}), Markdown and text`;
  $('addhint').textContent = hint;
  $('avhint').textContent = hint;
}

async function loadModels() {
  try { MODELS = await api('/api/models'); } catch {}
  const all = store.get('claudeAllPages');
  for (const [, input] of Object.values(PAGE_BOXES)) $(input).checked = all ? all === '1' : MODELS.claude_pages === 'all';
  fillPickers();
  uploadTypes();
}
for (const id of Object.keys(PICKERS)) {
  $(id).onchange = e => {
    store.set(PICKERS[id], e.target.value);
    syncPageBoxes();
    if (e.target.value.startsWith('claude:') && !claudeReady()) askKey();
  };
}

// ---- instructions (added to every question's system prompt) ----
async function loadRules() {
  let rules;
  try { rules = await api('/api/instructions'); } catch (err) { $('rules').replaceChildren(el('div', 'muted', err.message)); return; }
  $('rulecount').textContent = rules.filter(r => r.enabled).length || '';
  const box = $('rules');
  box.replaceChildren();
  if (!rules.length) box.append(el('div', 'muted', 'No instructions yet.'));
  for (const r of rules) {
    const row = el('div', 'rule' + (r.enabled ? '' : ' off'));
    const on = Object.assign(el('input'), { type: 'checkbox', checked: r.enabled, title: 'Switch this instruction on or off' });
    on.setAttribute('aria-label', 'Use this instruction');
    const text = Object.assign(el('textarea'), { value: r.text, rows: Math.min(8, Math.max(2, Math.ceil(r.text.length / 70))) });
    text.setAttribute('aria-label', 'Instruction text');
    const saved = el('span', 'saved');
    const flash = msg => { saved.textContent = msg; setTimeout(() => saved.textContent = '', 1500); };
    const save = async patch => {
      try { await send('/api/instructions/' + r.id, patch, 'PATCH'); flash('Saved'); }
      catch (err) { saved.textContent = err.message; }
    };
    on.onchange = async () => { row.classList.toggle('off', !on.checked); await save({ enabled: on.checked }); loadRules(); };
    text.onchange = () => { if (text.value.trim() && text.value.trim() !== r.text) { r.text = text.value.trim(); save({ text: r.text }); } };
    const del = Object.assign(el('button', 'x', '✕'), { type: 'button' });
    del.setAttribute('aria-label', 'Delete instruction');
    del.onclick = async () => {
      if (!del.classList.contains('sure')) {  // second click confirms
        del.classList.add('sure'); del.textContent = 'Delete?';
        setTimeout(() => { del.classList.remove('sure'); del.textContent = '✕'; }, 3000);
        return;
      }
      await api('/api/instructions/' + r.id, { method: 'DELETE' }).catch(() => {});
      loadRules();
    };
    const side = el('div', 'side');
    side.append(del, saved);
    row.append(on, text, side);
    box.append(row);
  }
}
$('rulesbtn').onclick = () => { loadRules(); $('rulesview').showModal(); };
$('rvclose').onclick = () => $('rulesview').close();
$('addrule').onclick = async () => {
  const text = $('newrule').value.trim();
  if (!text) { $('newrule').focus(); return; }
  try {
    await send('/api/instructions', { text });
    $('newrule').value = '';
  } catch (err) { toast(err.message, { error: true }); }
  loadRules();
};

// ---- usage: every model call with its tokens and cost ----
const fmtInt = n => (n || 0).toLocaleString();
const fmtTok = n => n >= 999_500 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M' : n >= 1e4 ? Math.round(n / 1e3) + 'K' : fmtInt(n);
const fmtUsd = c => !c ? '$0' : c < 0.0001 ? '<$0.0001' : c < 0.01 ? '$' + c.toFixed(4) : '$' + c.toFixed(2);
const inputOf = r => r.input_tokens + r.cache_read_tokens + r.cache_write_tokens;
const inputTip = r => `${fmtInt(inputOf(r))} input tokens: ${fmtInt(r.input_tokens)} uncached, `
  + `${fmtInt(r.cache_read_tokens)} read from cache, ${fmtInt(r.cache_write_tokens)} written to cache`;

function usageRow(name, r) {
  const tr = el('tr');
  for (const v of [name, fmtInt(r.calls), fmtTok(inputOf(r)), fmtTok(r.output_tokens), fmtUsd(r.cost_usd)]) {
    const td = el('td');
    td.append(v);
    tr.append(td);
  }
  tr.children[2].title = inputTip(r);
  tr.children[3].title = `${fmtInt(r.output_tokens)} output tokens`;
  tr.children[4].title = `$${(r.cost_usd || 0).toFixed(6)}`;
  return tr;
}

function usageTable(first, rows, total) {
  const t = el('table', 'utable'), head = el('thead'), hr = el('tr');
  for (const h of [first, 'Calls', 'Input', 'Output', 'Cost']) hr.append(el('th', '', h));
  head.append(hr);
  const body = el('tbody');
  for (const [name, r] of rows) body.append(usageRow(name, r));
  t.append(head, body);
  if (total) {
    const foot = el('tfoot');
    foot.append(usageRow('Total', total));
    t.append(foot);
  }
  const wrap = el('div', 'uscroll');
  wrap.append(t);
  return wrap;
}

async function loadUsage() {
  const box = $('usage'), days = $('usagedays').value;
  store.set('usageDays', days);
  let u;
  try { u = await api('/api/usage?days=' + days); } catch (err) { box.replaceChildren(el('div', 'muted', err.message)); return; }
  if (days !== $('usagedays').value) return;  // a newer choice is loading
  const t = u.total;
  const tiles = el('div', 'tiles4');
  for (const [label, value, tip] of [
    ['Cost', fmtUsd(t.cost_usd), `$${t.cost_usd.toFixed(6)} since ${u.since}`],
    ['Model calls', fmtInt(t.calls), `${fmtInt(t.calls)} calls since ${u.since}`],
    ['Input tokens', fmtTok(inputOf(t)), inputTip(t)],
    ['Output tokens', fmtTok(t.output_tokens), `${fmtInt(t.output_tokens)} output tokens`],
  ]) {
    const tile = el('div', 'tile4');
    tile.title = tip;
    tile.append(el('span', 'label', label), el('b', '', value));
    tiles.append(tile);
  }
  const note = el('p', 'muted', 'Every model call is logged: reading pages, analyzing documents and answering. '
    + 'Local models cost nothing. Claude costs come from the list prices in app/llm.py, including each model’s '
    + 'cheaper cache-read price, and are estimates, not your invoice. Input includes cached tokens. Days are UTC.');
  if (!t.calls) {
    box.replaceChildren(tiles, el('div', 'muted', `No model calls in the last ${days} days.`), note);
    return;
  }
  const models = u.by_model.map(m => {
    const name = document.createDocumentFragment();
    name.append(m.label, ' ', el('small', '', m.backend === 'ollama' ? 'local' : 'Claude API'));
    return [name, m];
  });
  const dayName = d => new Date(d + 'T00:00:00Z').toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short', timeZone: 'UTC' });
  box.replaceChildren(tiles,
    el('h3', 'label', 'By model'), usageTable('Model', models, t),
    el('h3', 'label', 'By day'), usageTable('Day', u.by_day.map(d => [dayName(d.day), d])),
    note);
}
const savedDays = store.get('usageDays');
if ([...$('usagedays').options].some(o => o.value === savedDays)) $('usagedays').value = savedDays;
$('usagebtn').onclick = () => { $('usage').replaceChildren(el('div', 'muted', 'Loading…')); $('usageview').showModal(); loadUsage(); };
$('usagedays').onchange = loadUsage;
$('uvclose').onclick = () => $('usageview').close();

// ---- unit fixes: litre/gallon misprints the unit check found in a document ----
// Answers never see the misprinted figure: units.py corrects each pair in the excerpts.
const UNIT_KINDS = {
  'quarts-printed-as-gallons': 'Quarts printed as gallons',
  'imperial-printed-as-us-gallons': 'Imperial gallons printed as U.S. gallons',
  'inconsistent-pair': 'Litres and gallons disagree',
};
const fixesText = n => `${n} unit fix${n === 1 ? '' : 'es'}`;

function unitBadge(id, n, tab = true) {
  const b = Object.assign(el('button', 'ubadge', fixesText(n)), { type: 'button', onclick: e => { e.stopPropagation(); showUnits(id); } });
  b.dataset.units = id;
  if (!tab) b.tabIndex = -1;
  b.title = 'Litre and gallon figures that disagree in this document: see what answers are given instead';
  return b;
}

let unitsShown = 0;  // the latest document asked for: an older answer arriving late is dropped
async function showUnits(id) {
  const box = $('units'), mine = ++unitsShown;
  $('unvtitle').textContent = 'Unit fixes';
  box.replaceChildren(el('div', 'muted', 'Loading…'));
  if (!$('unitsview').open) $('unitsview').showModal();
  let r;
  try { r = await api(`/api/documents/${id}/units`); } catch (err) { r = { error: err.message }; }
  if (mine !== unitsShown) return;
  if (r.error) { box.replaceChildren(el('div', 'muted', r.error)); return; }
  const found = r.findings || [], total = r.total ?? found.length;
  $('unvtitle').textContent = `${fixesText(total)} · ${r.title || r.filename}`;
  const intro = el('p', 'muted', found.length
    ? `${r.filename} prints these litre and gallon figures side by side, and they disagree. The litre figure is trusted: `
      + 'before a passage goes to a model, the gallon figure is replaced as shown, with what the document printed in brackets. '
      + 'The stored text and the original file keep what the document printed.'
      + (total > found.length ? ` Only the first ${fmtInt(found.length)} of ${fmtInt(total)} are kept and listed.` : '')
    : `No litre and gallon figures that disagree were found in ${r.filename}.`);
  const items = found.map(f => {
    const item = el('div', 'finding'), head = el('div', 'head');
    if (f.page != null) {
      head.append(r.pdf ? Object.assign(el('a', 'open', `Page ${f.page}`), {
        href: `/api/documents/${r.id}/file#page=${f.page}`, target: '_blank', rel: 'noopener', title: 'Open the original at this page' })
        : el('span', 'label', `Page ${f.page}`));
    }
    head.append(el('b', '', UNIT_KINDS[f.kind] || f.kind));
    item.append(head);
    if (f.context) item.append(el('div', 'ctx', f.context));
    const dl = el('dl');
    dl.append(el('dt', '', 'Printed'), el('dd', 'was', f.printed), el('dt', '', 'Models read'), el('dd', '', f.corrected));
    item.append(dl);
    return item;
  });
  box.replaceChildren(intro, ...items);
}
$('unvclose').onclick = () => $('unitsview').close();

// ---- upload queue: each file is sent, then read on the server as a background job ----
// A row per upload or re-read: name, state, ✕, a progress bar, details, buttons. Rows are
// updated in place from GET /api/jobs, so buttons keep their focus between polls. On a phone
// the queue sits in Documents (details#uploads); placeUploads (docs.js) moves the same node.
const ACTIVE = new Set(['queued', 'awaiting_confirmation', 'running']);
const jobRows = new Map();  // job id -> row
const gone = new Set();     // jobs whose row was closed: a poll already under way mustn't bring it back
const freshDocs = new Set(); // documents a job just added: their cards flash once

function newRow() {
  const row = el('div', 'job');
  row.tabIndex = -1;
  row.nameEl = el('b');
  row.spin = el('span', 'spin');
  row.state = el('span', 'state');
  row.x = Object.assign(el('button', 'x', '✕'), { type: 'button' });
  row.bar = el('div', 'bar');
  row.fill = el('i');
  row.bar.append(row.fill);
  row.detail = el('div', 'detail');
  row.acts = el('div', 'acts');
  row.append(row.nameEl, row.spin, row.state, row.x, row.bar, row.detail, row.acts);
  $('queue').append(row);
  return row;
}

function jobRow(id) {
  if (!jobRows.has(id)) jobRows.set(id, newRow());
  return jobRows.get(id);
}

function dropRow(row) {
  row.remove();
  for (const [id, r] of jobRows) if (r === row) { jobRows.delete(id); gone.add(id); }
  syncBadge();
}
// A finished row hides after `ms`, but not while the pointer or focus is on it.
function hideLater(row, ms) {
  setTimeout(() => {
    if (!row.isConnected) return;
    if (row.matches(':hover, :focus-within')) hideLater(row, 3000); else dropRow(row);
  }, ms);
}

const needsYou = r => r.classList.contains('bad') || r.classList.contains('dup');
const alerted = new WeakSet();  // rows that opened the phone's Uploads box once already
// The documents button's badge: rows under way, and rows waiting for you (failed, a cost or a
// duplicate to decide), which turn it red; and the phone's Uploads summary.
function syncBadge() {
  let working = 0, waiting = 0, fresh = false;
  for (const r of $('queue').children) {
    if (needsYou(r)) {
      waiting++;
      if (!alerted.has(r)) { alerted.add(r); fresh = true; }
    } else if (r.job ? ACTIVE.has(r.job.status) : !r.spin.hidden) working++;
  }
  const n = working + waiting, badge = $('jobcount');
  badge.textContent = n || '';
  badge.classList.toggle('alert', waiting > 0);
  const summary = ['Uploads', working ? `${working} working` : '', waiting ? `${waiting} need${waiting === 1 ? 's' : ''} you` : '']
    .filter(Boolean).join(' · ');
  $('uploadsum').textContent = summary;
  $('uploads').hidden = !$('queue').children.length;
  if (fresh && isS.matches) $('uploads').open = true;  // once per row that starts needing you
  if (typeof syncDocsButton === 'function') syncDocsButton(working, waiting);
  if (typeof jobsChanged === 'function') jobsChanged();
}

// What a row shows. bar: null (none), 'wait' (moving) or 0..1; lines: [[text or node, class]];
// buttons: [[label, tip, onclick]]; x: the ✕ as [tip, onclick], or null for none.
function paint(row, { name, state, cls = '', spin = false, bar = null, lines = [], buttons = [], x = null }) {
  row.className = 'job' + (cls ? ' ' + cls : '');
  row.nameEl.textContent = name;
  row.title = name;
  row.spin.hidden = !spin;
  row.state.textContent = state;
  row.bar.hidden = bar === null;
  row.bar.classList.toggle('wait', bar === 'wait');
  row.fill.style.width = typeof bar === 'number' ? `${Math.round(bar * 100)}%` : '';
  // What a ghost card shows of it (docs.js): "Working · Reading page 3 of 10".
  const first = lines.find(l => typeof l[0] === 'string' && l[0] && !l[0].startsWith('→'))?.[0];
  row.view = { name, state, bar, needs: cls === 'bad' || cls === 'dup', stage: first && first !== state ? `${state} · ${first}` : state };
  if (row.note) lines = [...lines, [row.note, 'err']];  // a button's error, until the next click
  row.detail.replaceChildren(...lines.filter(l => l[0]).map(([text, c]) => { const d = el('div', c || ''); d.append(text); return d; }));
  const sig = JSON.stringify([buttons.map(b => b[0]), x && x[0]]);
  if (row.sig !== sig) {  // rebuilt only when they change, so focus and hover survive a poll
    row.sig = sig;
    row.acts.replaceChildren(...buttons.map(([label, tip]) => Object.assign(el('button', '', label), { type: 'button', title: tip })));
    row.x.classList.remove('sure');
    row.x.textContent = '✕';
  }
  [...row.acts.children].forEach((b, i) => b.onclick = buttons[i][2]);
  row.x.hidden = !x;
  if (x) { row.x.title = x[0]; row.x.setAttribute('aria-label', x[0]); row.x.onclick = x[1]; }
  syncBadge();
}

function estimateText(e) {
  if (!e) return '';
  const unpriced = e.priced === false ? ' (this model has no list price in app/llm.py: priced like the dearest one)' : '';
  // A long text is analyzed in sections, one call each, plus one combining their notes.
  const [n, read] = e.sections || [1, 1];
  const parts = n > 1 ? `, analysis in ${read < n ? `${read} of ${n}` : n} sections` : '';
  const cost = `about ${fmtTok(e.input_tokens)} input and ${fmtTok(e.output_tokens)} output tokens${parts}, est. ${fmtUsd(e.cost_usd)}${unpriced}`;
  if (e.image) return `${e.model_label} reads the photo: ${cost}`;
  if (e.mode == null) return `${e.model_label} analyzes the text: ${cost}`;
  if (!e.claude_pages) return `No scanned pages, so ${e.model_label} only analyzes the text layer: ${cost}`;
  const which = e.claude_pages < e.pages ? `${e.claude_pages} of ${e.pages} pages (the scanned one${e.claude_pages > 1 ? 's' : ''})`
    : e.pages > 1 ? `all ${e.pages} pages` : 'its one page';
  return `${e.model_label} reads ${which}: ${cost}`;
}

const sure = (b, then) => () => {  // a second click within 3 s confirms
  if (b.classList.contains('sure')) { then(); return; }
  b.classList.add('sure'); b.textContent = 'Cancel?';
  setTimeout(() => { b.classList.remove('sure'); b.textContent = '✕'; }, 3000);
};

async function jobAction(row, id, what) {
  row.note = '';
  try {
    const r = await api(`/api/jobs/${id}/${what}`, { method: 'POST' });
    if (r?.job) showJob(row, r.job);
  } catch (err) {
    if (err.status === 401) askKey();
    if (err.data?.job) showJob(row, err.data.job);  // say it finished meanwhile: the row shows that
    else { row.note = err.message; if (row.job) showJob(row, row.job); }
  }
  startPolling(0);
}

// Where an upload goes, for its row: "→ Tractor › Manuals" or "→ Unfiled · auto-file".
function destText(o) {
  if (o?.folder_id != null) return `→ ${pathText(o.folder_id)}`;
  return `→ Unfiled${o?.auto_file === false ? '' : ' · auto-file'}`;
}

function showJob(row, j) {
  const before = row.job?.status;
  row.job = j;
  const name = (j.kind === 'reread' ? 'Re-read · ' : '') + j.filename;
  const est = estimateText(j.estimate), act = what => () => jobAction(row, j.id, what);
  const switched = j.options?.switched ? `Switched to Claude: ${j.options.switched}` : '';
  const dest = j.kind === 'upload' ? destText(j.options) : '';
  if (j.status === 'cancelled') { dropRow(row); return; }
  if (j.status === 'queued' || j.status === 'running') {
    // Only the job being read moves: a bar while its pages are counted (never quite empty
    // once started), a moving one otherwise. A queued job waits still.
    const running = j.status === 'running', reading = running && j.total > 0 && j.progress < j.total;
    paint(row, { name, state: running ? 'Working' : 'Queued', spin: running,
      bar: !running ? null : reading ? Math.max(j.progress / j.total, 0.03) : 'wait', lines: [[j.stage], [dest], [switched], [est]],
      x: ['Cancel this job', null] });
    row.x.onclick = sure(row.x, act('cancel'));
  } else if (j.status === 'awaiting_confirmation') {
    paint(row, { name, state: 'Confirm cost', cls: 'dup',
      lines: [[switched], [est], [`Jobs estimated at ${fmtUsd(j.estimate?.confirm_at)} or more wait for a go-ahead.`], [dest]],
      buttons: [['Start', 'Read it with Claude at about this cost', act('confirm')],
                ['Cancel', 'Leave the library as it is', act('cancel')]] });
  } else if (j.status === 'failed') {
    const kept = j.pages_kept ? ` (keeps the ${j.pages_kept} page${j.pages_kept > 1 ? 's' : ''} already read)` : '';
    paint(row, { name, state: 'Failed', cls: 'bad', lines: [[j.error, 'err'], [est], [dest]],
      buttons: [['Retry', 'Try this job again' + kept, act('retry')]],
      x: ['Dismiss this failed job', act('cancel')] });
  } else if (j.status === 'done') {
    const r = j.result || {};
    const verb = j.kind === 'reread' ? 'Re-read' : r.replaced?.length ? 'Replaced' : 'Added';
    const show = ['Show', 'Show it in Documents', () => showDocument(j.doc_id)];
    let outcome = '', buttons = [], wait = 4000;
    if (j.kind === 'upload' && j.doc_id) {
      if (r.folder_note) { outcome = r.folder_note; buttons = [show]; }
      else if (r.filed === 'auto') {  // stays until hidden: someone who dropped files and walked away sees where they went
        outcome = frag('Filed in ', [pathText(r.folder_id)], ' automatically');
        buttons = [show, ['Undo', 'Move it to Unfiled', () => moveDocs([j.doc_id], null).then(ok => ok && dropRow(row))]];
        wait = 0;
      } else if (r.filed === 'kept') {
        outcome = `Replaced · stays in ${pathText(r.folder_id)}`;
        buttons = [show];
      } else if (r.filed === 'chosen') {
        outcome = frag('Added to ', [pathText(r.folder_id)]);
        buttons = [show];
      } else if (r.suggested_folder) {
        const where = r.suggested_folder.split('/').join(' › ');
        outcome = `Suggested folder: ${where}`;
        buttons = [[r.suggestion_exists ? `Move to ${where}` : `Move to new folder '${where}'`,
                    r.suggestion_exists ? 'Move it into that folder' : 'Create the folder and move it there',
                    () => applySuggestion(j.doc_id, r.suggested_folder).then(ok => ok && dropRow(row))], show];
        wait = 10000;
      } else buttons = [show];
    } else if (j.doc_id) buttons = [show];
    paint(row, { name, state: `${verb} · ${r.model_label || ''}`, cls: 'ok',
      lines: [[outcome], [r.switched ? `Switched to ${r.model_label}: ${r.switched}` : ''],
              [r.analysis && r.analysis !== 'full' ? `Analysis: ${r.analysis}` : ''],
              [r.unit_fixes ? `${fixesText(r.unit_fixes)}: litre/gallon misprints, corrected in answers` : ''], [est]],
      buttons, x: ['Hide', () => dropRow(row)] });
    if (before !== 'done' && !r.switched && wait) hideLater(row, wait);
  }
}

// Poll while any job is unfinished, or an action asked for news.
let pollTimer = null, polling = false, pollAgain = false;
function startPolling(delay = 300) {
  pollAgain = true;
  if (!polling && !pollTimer) pollTimer = setTimeout(pollJobs, delay);
}

async function pollJobs() {
  pollTimer = null; polling = true; pollAgain = false;
  let list = null, active = false, finished = false;
  try { list = await api('/api/jobs'); } catch {}
  if (list) {
    const ids = new Set(list.map(j => j.id));
    for (const j of list) {
      const row = jobRows.get(j.id);
      // Unknown here and already finished: done before this page saw it (the library shows it).
      if (!row && ((!ACTIVE.has(j.status) && j.status !== 'failed') || gone.has(j.id))) continue;
      if (row?.job && row.job.status !== 'done' && j.status === 'done') {
        finished = true;
        if (j.doc_id) { freshDocs.add(j.doc_id); forgetText(j.doc_id); }
        // Seen finishing here, so the ids it replaced were freed just now, not since given to other uploads.
        if (j.result?.replaced?.length) rescope(j.result.replaced, j.doc_id);
      }
      showJob(row || jobRow(j.id), j);
      active ||= ACTIVE.has(j.status);
    }
    for (const [id, row] of jobRows) if (!ids.has(id) && row.job) dropRow(row);  // dismissed elsewhere
    if (finished) lib.load();
  }
  polling = false;
  if (active || pollAgain || !list) pollTimer = setTimeout(pollJobs, !list ? 5000 : document.hidden ? 4000 : 1500);
}

// The upload itself, with progress (fetch can't report upload progress).
function sendFile(path, body, onProgress) {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open('POST', path);
    x.setRequestHeader('X-API-Key', key.value);
    for (const [k, v] of Object.entries(OWN)) x.setRequestHeader(k, v);
    x.upload.onprogress = e => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    x.onload = () => {
      let data = null;
      try { data = JSON.parse(x.responseText); } catch {}
      if (loginGone(x.status, data)) return toLogin();
      if (x.status >= 200 && x.status < 300) return resolve(data);
      const msg = data?.error || (x.status === 413 ? 'File is too large to upload' : `${x.status} ${x.statusText}`);
      reject(Object.assign(new Error(msg), { status: x.status, data }));
    };
    x.onerror = () => reject(new Error('The upload broke off: is the server reachable?'));
    x.send(body);
  });
}

// target: {folder_id: id or null (Unfiled), auto_file}; uploadTarget() (docs.js) is where the
// page is: the folder shown in Documents, else Unfiled.
async function uploadAll(files, target = uploadTarget()) {
  for (const [i, f] of files.entries()) {
    const done = uploadOne(f, '', null, target);  // its row is in the queue once this returns
    if (!i && typeof revealUploads === 'function') revealUploads();
    await done;
  }
}

// onDuplicate: '' (the server answers 409 if the file is already in the library),
// 'replace' or 'keep'. The row is reused when the user answers the duplicate prompt.
async function uploadOne(f, onDuplicate = '', row = null, target = uploadTarget()) {
  row ||= newRow();
  row.file = f;
  row.target = target;
  const model = $('docModel').value;
  const body = new FormData();
  body.append('file', f);
  body.append('model', model);
  if (model.startsWith('claude:')) body.append('claude_pages', claudePages('docModel'));
  if (onDuplicate) body.append('on_duplicate', onDuplicate);
  if (target.folder_id != null) body.append('folder_id', String(target.folder_id));
  body.append('auto_file', target.auto_file ? '1' : '0');
  const uploading = p => paint(row, { name: f.name, state: p < 1 ? `Uploading ${Math.floor(p * 100)}%` : 'Uploaded',
    spin: true, bar: p < 1 ? p : 'wait', lines: [[destText(target)]] });
  uploading(0);
  try {
    const limit = MODELS.max_upload_mb;  // check before sending megabytes the server will refuse
    if (limit && f.size > limit * 1024 * 1024)
      throw new Error(`${(f.size / 1048576).toFixed(0)} MB is over the ${limit} MB upload limit`);
    const r = await sendFile('/api/upload', body, uploading);
    // A poll that ran between the job's commit and this answer has given it a row of its own.
    const other = jobRows.get(r.job.id);
    if (other && other !== row) other.remove();
    jobRows.set(r.job.id, row);
    showJob(row, r.job);
    startPolling();
  } catch (err) {
    if (err.status === 409 && err.data?.duplicate) askDuplicate(f, row, err.data.duplicate, target);
    else if (err.status === 404 && err.data?.folder_gone) {
      lib.load();
      paint(row, { name: f.name, state: 'Failed', cls: 'bad', lines: [[err.message, 'err']],
        buttons: [['Add to Unfiled', 'Upload it to Unfiled instead', () => uploadOne(f, onDuplicate, row, { folder_id: null, auto_file: target.auto_file })]],
        x: ['Hide', () => dropRow(row)] });
    } else {
      if (err.status === 401) askKey();
      paint(row, { name: f.name, state: 'Failed', cls: 'bad', lines: [[err.message, 'err']], x: ['Hide', () => dropRow(row)] });
    }
  }
}

// A row for something that couldn't be uploaded at all (a folder dropped from the computer).
function failedRow(name, message) {
  const row = newRow();
  paint(row, { name, state: 'Failed', cls: 'bad', lines: [[message, 'err']], x: ['Hide', () => dropRow(row)] });
}

function askDuplicate(f, row, dup, target) {
  const here = target.folder_id != null ? pathText(target.folder_id) : 'Unfiled';
  const there = dup.folder_id !== undefined ? pathText(dup.folder_id) : '';
  const choices = [
    ['Replace', 'replace', `Read this file and delete the copy already in the library.${there ? ` The new copy stays in ${there}.` : ''}`],
    ['Keep both', 'keep', `Add this file to ${here} as another copy`],
    ['Skip', '', 'Leave the library as it is'],
  ];
  let state, text;
  if (dup.in_progress) {  // say the page was reloaded during an upload: the server is still on it
    state = 'Being read';
    text = `The same file is queued or being read right now as “${dup.filename}” (since ${dup.started_at} UTC). `
      + 'It appears in the library when done, even after a reload.';
    choices.shift();  // nothing to replace yet
  } else {
    state = 'Already added';
    const by = dup.model ? `, read by ${dup.model}` : '';
    const copies = dup.copies > 1 ? ` (${dup.copies} copies)` : '';
    text = frag(`The same file is in the library as “${dup.filename}”${copies}`, ...(there ? [' in ', [there]] : []),
      `, added ${dup.created_at} UTC${by}.`);
  }
  paint(row, { name: f.name, state, cls: 'dup', lines: [[text]],
    buttons: choices.map(([label, how, tip]) => [label, tip, () => how ? uploadOne(f, how, row, target) : dropRow(row)]) });
}
$('file').onchange = e => { uploadAll([...e.target.files], fileTarget || uploadTarget()); fileTarget = null; e.target.value = ''; };
// A photo from the phone's camera, named for when it was taken (the server checks its format).
$('camera').onchange = e => {
  const f = e.target.files[0];
  e.target.value = '';
  if (!f) return;
  const ext = { 'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp', 'image/heic': 'heic', 'image/heif': 'heif' }[f.type]
    || (f.name.split('.').pop() || 'jpg').toLowerCase();
  const d = new Date(), p = n => String(n).padStart(2, '0');
  const name = `Photo ${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}.${p(d.getMinutes())}.${ext}`;
  uploadAll([new File([f], name, { type: f.type })], fileTarget || uploadTarget());
  fileTarget = null;
};
let fileTarget = null;  // where the file picker's files go, when not where the page is

// ---- scope: what questions search (all documents when nothing is chosen) ----
// Chosen documents and folders, kept in this browser. A folder is resolved when a question is
// asked, so documents added to it later are searched too. The server drops what was deleted
// meanwhile; so does the page, whenever it lists the whole library. When everything chosen is
// gone, the chip says so rather than turning into every document unasked, and questions wait
// for a new choice. Only how many were chosen is kept then, not their ids (ids are reused).
let scope = { docs: [], folders: [] };
let docNames = null, folderNames = null;  // id -> document / folder, from the last full listing
let scopeGone = +store.get('chatScopeGone') || 0;  // how many chosen items were all deleted
let scopeGoneFolders = store.get('chatScopeGoneF') === '1';
const ints = raw => { try { return (JSON.parse(raw || '[]') || []).filter(Number.isInteger); } catch { return []; } };
scope.docs = ints(store.get('chatScope'));
scope.folders = ints(store.get('chatScopeFolders'));
const docName = id => { const d = docNames?.get(id); return d ? d.title || d.filename : `document ${id}`; };
const folderName = id => folderNames?.get(id)?.name || `folder ${id}`;
const scopeAll = () => !scope.docs.length && !scope.folders.length;
function scopeText() {
  if (scopeGone) return scopeGoneFolders ? `Chosen folder${scopeGone > 1 ? 's' : ''} deleted` : `Chosen document${scopeGone > 1 ? 's' : ''} deleted`;
  const d = scope.docs.length, f = scope.folders.length;
  if (!d && !f) return 'All documents';
  if (f === 1 && !d) return folderName(scope.folders[0]);
  if (d === 1 && !f) return docName(scope.docs[0]);
  return [f ? plural(f, 'folder') : '', d ? plural(d, 'document') : ''].filter(Boolean).join(', ');
}
// "Receipts (folder) and 2 documents": what a question was limited to, under its bubble.
function onlyText(folders, docs) {
  const f = folders.length, d = docs.length;
  const fp = !f ? '' : f === 1 ? `${folders[0]} (folder)` : `${f} folders`;
  const dp = !d ? '' : d === 1 ? docs[0] : `${d} documents`;
  return [fp, dp].filter(Boolean).join(' and ');
}
const GONE = 'The documents or folders chosen for questions were deleted. Choose others with the chip under the question box, or ✕ to ask all documents.';

function setScope(next, gone = 0, goneFolders = false) {
  scope = gone ? { docs: [], folders: [] } : { docs: [...new Set(next.docs || [])], folders: [...new Set(next.folders || [])] };
  scopeGone = gone;
  scopeGoneFolders = !!(gone && goneFolders);
  store.set('chatScope', scope.docs.length ? JSON.stringify(scope.docs) : '');
  store.set('chatScopeFolders', scope.folders.length ? JSON.stringify(scope.folders) : '');
  store.set('chatScopeGone', gone ? String(gone) : '');
  store.set('chatScopeGoneF', gone && goneFolders ? '1' : '');
  showScope();
  if (typeof renderTree === 'function') renderTree();  // the tree marks the folders asked about
}

function scopeTitle() {
  if (scopeGone) return GONE;
  if (scopeAll()) return 'Questions search every document. Click to choose folders or documents.';
  const fn = scope.folders.map(folderName), dn = scope.docs.slice(0, 8).map(docName);
  const fpart = !fn.length ? '' : fn.length === 1 ? `the folder ${fn[0]} and the folders in it, including documents added later`
    : `the folders ${fn.join(', ')} and the folders in them, including documents added later`;
  const dpart = dn.join(', ') + (scope.docs.length > 8 ? ', …' : '');
  return `Questions search only ${[fpart, dpart].filter(Boolean).join('; and ')}. Click to change.`;
}

function showScope() {
  const on = !scopeAll(), b = $('scopebtn'), text = scopeText(), one = !scopeGone && scope.folders.length === 1 && !scope.docs.length;
  $('scopechip').classList.toggle('on', on);
  $('scopechip').classList.toggle('gone', !!scopeGone);
  $('question').placeholder = one ? `Ask about ${folderName(scope.folders[0])}…` : 'Ask about your documents…';
  b.replaceChildren();
  if (one) b.append(icon('folder'));
  b.append(text);
  b.title = scopeTitle();
  b.setAttribute('aria-label', `Ask ${text}. ${b.title}`);
  $('scopeclear').hidden = !on && !scopeGone;
  if ($('log').querySelector('[data-empty]')) emptyState();
}
function pulseScope() {
  const chip = $('scopechip');
  chip.classList.remove('pulse');
  void chip.offsetWidth;
  chip.classList.add('pulse');
}

function knowDocs(docs) {  // the whole library, as just listed
  docNames = new Map(docs.map(d => [d.id, d]));
  pruneScope();
}
function knowFolders(folders) {
  folderNames = new Map(folders.map(f => [f.id, f]));
  pruneScope();
}

function pruneScope() {  // chosen documents and folders deleted meanwhile drop out
  if (!docNames || !folderNames) { showScope(); return; }
  const docs = scope.docs.filter(id => docNames.has(id)), folders = scope.folders.filter(id => folderNames.has(id));
  if (docs.length === scope.docs.length && folders.length === scope.folders.length) { showScope(); return; }
  const all = scope.docs.length + scope.folders.length;
  if (docs.length || folders.length) setScope({ docs, folders });
  else setScope({}, all, scope.folders.length > 0);
}

function rescope(old, now) {  // "Replace" gave a chosen document a new id: questions follow it
  if (now && old.some(id => scope.docs.includes(id))) setScope({ ...scope, docs: scope.docs.map(id => old.includes(id) ? now : id) });
}

// Ask about these from now on: from a folder, a card, the selection, the detail. Back in Chat,
// with the composer focused (fine pointers) and the chip pulsing once.
function askAbout(next) {
  setScope(next);
  for (const d of document.querySelectorAll('dialog[open]')) d.close();
  showView('chat', { focus: false });
  if (!coarse.matches && !isS.matches) $('question').focus();
  pulseScope();
}

$('scopeclear').onclick = () => { setScope({}); $('question').focus(); };

// ---- an answer's sources: the passages it was given, by document ----
const MATCH = { keywords: 'found by keywords', meaning: 'found by meaning', both: 'found by keywords and by meaning',
  opening: 'the start of the document' };

function quoteText(e) {  // a passage, with its "[Page n]" lines opening that page of the kept PDF
  const q = el('div', 'quote');
  e.text.split(/^\[Page (\d+)\]$/m).forEach((part, i) => {
    const n = +part;
    q.append(!(i % 2) ? part : e.has_file && e.pdf && n >= e.page_start && n <= e.page_end
      ? Object.assign(el('a', 'open', `[Page ${n}]`), { href: `/api/documents/${e.doc_id}/file#page=${n}`, target: '_blank',
          rel: 'noopener', title: `Open page ${n} of the original` })
      : `[Page ${part}]`);
  });
  return q;
}

function sourceChip(s, excerpts) {
  if (typeof s === 'string') return el('span', 'chip', s);  // an answer from before sources had pages
  const chip = Object.assign(el('button', 'chip src'), { type: 'button' });
  chip.append(el('span', 'fn', s.filename));
  if (s.pages?.length) chip.append(el('span', 'pg', ', ' + pagesText(s.pages)));
  const n = s.excerpts.length;
  chip.title = `${s.filename}${s.pages?.length ? ', ' + pagesText(s.pages) : ''}: see the passage${n > 1 ? 's' : ''} this answer was given`;
  chip.onclick = () => showSource(s, excerpts);
  return chip;
}

function showSource(s, excerpts) {
  const used = s.excerpts.map(i => excerpts[i]).filter(Boolean), n = used.length;
  $('srctitle').textContent = `${n} passage${n === 1 ? '' : 's'} · ${s.title || s.filename}`;
  const intro = el('p', 'muted', `From ${s.filename}, as the model read ${n === 1 ? 'it' : 'them'} (with any litre/gallon `
    + 'misprints corrected). ' + (s.gone ? 'It has been deleted from the library since. ' : '') + (s.pages?.length ? ''
      : /\.pdf$/i.test(s.filename) ? 'This document was read before page numbers were kept, so its passages have none.'
      : 'This kind of file has no pages.'));
  const parts = [intro];
  if (!s.gone && lib.docs.has(s.doc_id)) {
    parts.push(btn('Show in Documents', () => { $('sourceview').close(); showDocument(s.doc_id); },
      { icon: 'folder', title: 'Open this document in Documents' }));
    parts[1].style.justifySelf = 'start';
  }
  const items = used.map(e => {
    const item = el('div', 'finding'), head = el('div', 'head');
    head.append(el('b', '', e.page_start ? (e.page_start === e.page_end ? `Page ${e.page_start}` : `Pages ${e.page_start}-${e.page_end}`)
      : `Passage ${e.idx + 1}`));
    if (e.has_file) {
      const page = e.pdf && e.page_start;
      head.append(Object.assign(el('a', 'open', page ? `Open page ${page}` : 'Open the file'), {
        href: `/api/documents/${e.doc_id}/file${page ? '#page=' + page : ''}`, target: '_blank', rel: 'noopener',
        title: page ? 'Open the original at this page' : 'Open the original file' }));
    }
    head.append(el('span', 'muted', MATCH[e.match] || ''));
    item.append(head, quoteText(e));
    return item;
  });
  $('source').replaceChildren(...parts, ...items);
  if (!$('sourceview').open) $('sourceview').showModal();
  $('source').scrollTop = 0;
}
$('srcclose').onclick = () => $('sourceview').close();

// ---- saved conversations: the Chats list, and the one shown ----
// The server keeps every question and its answer; the page remembers which conversation is
// open, so a reload comes back to it. Chats shows the list: a column beside the chat on wide
// screens (shown or hidden at will), a drawer below 1280px and while Documents is shown.
let chat = null;     // the open conversation, {id, title}; null for a new chat
let chats = [];      // the list, as last loaded
let chatShown = 0;   // moves whenever another conversation is shown: news of the one before is dropped
let renaming = false;
const drawer = matchMedia('(max-width: 1279px)');
const asDrawer = () => drawer.matches || document.body.classList.contains('docs');

function syncChatsButton() {
  const open = asDrawer() ? document.body.classList.contains('chatsopen') : !document.body.classList.contains('nochats');
  $('chatsbtn').setAttribute('aria-expanded', String(open));
}
if (store.get('chatsOpen') === '0') document.body.classList.add('nochats');
drawer.addEventListener('change', () => { document.body.classList.remove('chatsopen'); syncChatsButton(); });
syncChatsButton();
$('chatsbtn').onclick = e => {
  e.stopPropagation();
  if (asDrawer()) {  // the drawer
    if (document.body.classList.toggle('chatsopen')) loadChats();
  } else {
    store.set('chatsOpen', document.body.classList.toggle('nochats') ? '0' : '1');
  }
  syncChatsButton();
};
$('chatsclose').onclick = () => { document.body.classList.remove('chatsopen'); syncChatsButton(); };

// '2026-11-12 14:02:11' (UTC, as the server keeps time) -> '15:02' today, 'Yesterday', '3 Nov'.
function whenText(utc) {
  const d = new Date(String(utc).replace(' ', 'T') + 'Z'), now = new Date();
  if (isNaN(d)) return '';
  const day = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const days = Math.round((day(now) - day(d)) / 864e5);
  if (days === 0) return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  if (days === 1) return 'Yesterday';
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', ...(d.getFullYear() !== now.getFullYear() ? { year: 'numeric' } : {}) });
}

function setChat(c) {  // where questions go from now on (null: a new conversation, made by the first)
  chat = c ? { id: c.id, title: c.title || '' } : null;
  store.set('chatConversation', chat ? String(chat.id) : '');
  $('convtitle').textContent = chat?.title || 'New conversation';
  $('convtitle').title = chat?.title || '';
  document.title = chat?.title ? `${chat.title} · Document Chatbot` : 'Document Chatbot';
}

let chatsLoading = 0;
async function loadChats() {
  const mine = ++chatsLoading;
  let list;
  try { list = await api('/api/conversations'); } catch (err) {
    if (mine === chatsLoading && !chats.length) $('chatlist').replaceChildren(el('div', 'muted', err.message));
    return;
  }
  if (mine !== chatsLoading) return;
  chats = list;
  const open = chats.find(c => c.id === chat?.id);
  if (open && open.title !== chat.title && !renaming) setChat(open);  // renamed in another tab
  drawChats();
}

function drawChats() {
  if (renaming) return;  // the title being typed stays; drawn when it is done
  $('chatlist').replaceChildren(...(chats.length ? chats.map(chatItem)
    : [el('div', 'muted', 'No saved conversations yet. Every question and its answer are kept here.')]));
}

function chatItem(c) {
  const on = chat?.id === c.id, item = el('div', 'conv' + (on ? ' on' : ''));
  const pick = Object.assign(el('button', 'cbtn'), { type: 'button', title: c.title,
    onclick: () => { showView('chat', { focus: false }); openChat(c.id, true); } });
  pick.append(el('b', '', c.title || 'Untitled'),
    el('small', '', `${whenText(c.updated_at)} · ${c.questions} question${c.questions === 1 ? '' : 's'}`));
  if (on) pick.setAttribute('aria-current', 'true');
  const edit = Object.assign(el('button', 'x', '✎'), { type: 'button', title: 'Rename', onclick: () => renameChat(item, c) });
  edit.setAttribute('aria-label', `Rename “${c.title}”`);
  const del = Object.assign(el('button', 'x', '✕'), { type: 'button', title: 'Delete' });
  del.setAttribute('aria-label', `Delete “${c.title}”`);
  del.onclick = async () => {
    if (!del.classList.contains('sure')) {  // second click confirms
      del.classList.add('sure'); del.textContent = 'Delete?';
      setTimeout(() => { del.classList.remove('sure'); del.textContent = '✕'; }, 3000);
      return;
    }
    if (chat?.id === c.id) newChat(false);
    try { await api('/api/conversations/' + c.id, { method: 'DELETE' }); } catch (err) { if (err.status !== 404) toast(err.message, { error: true }); }
    loadChats();
  };
  item.append(pick, edit, del);
  return item;
}

function renameChat(item, c) {
  renaming = true;
  const input = Object.assign(el('input'), { value: c.title, maxLength: 200 });
  input.setAttribute('aria-label', 'Conversation title');
  let over = false;
  const finish = async save => {  // Enter or leaving the box saves; Escape keeps the old title
    if (over) return;
    over = true;
    const title = input.value.trim();
    if (save && title && title !== c.title) {
      try {
        const r = await send('/api/conversations/' + c.id, { title }, 'PATCH');
        c.title = r.title;
        if (chat?.id === c.id) setChat(r);
      } catch (err) { toast(err.message, { error: true }); }
    }
    renaming = false;
    drawChats();
  };
  input.onkeydown = e => {
    if (e.key === 'Enter') { e.preventDefault(); finish(true); }
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); finish(false); }
  };
  input.onblur = () => finish(true);
  item.replaceChildren(input);
  input.focus();
  input.select();
}

// A saved conversation, shown with its answers as they were given. Picked from the list, it
// also brings back its model and the folders and documents its questions were limited to (a
// reload keeps the page's own choice instead).
async function openChat(id, picked = false) {
  if (picked) { document.body.classList.remove('chatsopen'); syncChatsButton(); }
  if (picked && busy && chat?.id === id) return;  // the one being answered: it carries on
  stopAnswer();
  const mine = ++chatShown;
  let c;
  try { c = await api('/api/conversations/' + id); } catch (err) {
    if (mine !== chatShown) return;
    if (err.status === 404) { newChat(false); loadChats(); return; }  // deleted meanwhile, say in another tab
    setChat({ id, title: chats.find(x => x.id === id)?.title });
    $('log').replaceChildren(el('div', 'empty muted', `Could not load this conversation: ${err.message}`));
    return;
  }
  if (mine !== chatShown) return;
  setChat(c);
  if (picked) {
    if (c.model && [...$('chatModel').options].some(o => o.value === c.model)) {
      $('chatModel').value = c.model;
      store.set('chatModel', c.model);
    }
    // doc_ids [] with no folders: everything chosen for it was deleted. folder_ids is a list (now
    // empty) when folders were chosen, null when none were.
    const gone = Array.isArray(c.doc_ids) && !c.doc_ids.length && !c.folder_ids?.length;
    setScope({ docs: c.doc_ids || [], folders: c.folder_ids || [] }, gone ? 1 : 0, Array.isArray(c.folder_ids));
    pruneScope();
  }
  drawChat(c);
  if (pending(c)) watchChat(id, mine, sign(c));
  drawChats();
  scroll();
}

function drawChat(c) {  // a saved conversation's questions and answers
  $('log').replaceChildren();
  for (let i = 0; i < c.messages.length; i += 2) showExchange(c.messages[i], c.messages[i + 1]);
  if (!c.messages.length) emptyState();
}

// An answer still being written, in another tab or by this page before a reload (the server
// fills it in once it notices the page went), is read again until it is done.
const pending = c => c.messages.some(m => m.status === 'answering');
const sign = c => c.messages.map(m => `${m.id}:${m.status}:${m.content.length}`).join();
function watchChat(id, mine, seen, wait = 1000) {
  setTimeout(async () => {
    if (mine !== chatShown) return;
    if (busy) { watchChat(id, mine, seen, wait); return; }  // this page's own answer is on screen
    let c;
    try { c = await api('/api/conversations/' + id); } catch (err) {
      if (err.status !== 404) watchChat(id, mine, seen, Math.min(wait * 2, 10000));
      return;
    }
    if (mine !== chatShown || busy) { if (mine === chatShown) watchChat(id, mine, seen, wait); return; }
    if (sign(c) !== seen) {
      const pinned = nearBottom();
      drawChat(c);
      if (pinned) scroll();
    }
    if (pending(c)) watchChat(id, mine, sign(c), Math.min(wait * 1.5, 5000));
  }, wait);
}

function newChat(focus = true) {  // a new conversation starts on all documents (picking one brings back its own)
  stopAnswer();
  ++chatShown;
  setChat(null);
  setScope({});
  emptyState();
  drawChats();
  document.body.classList.remove('chatsopen');
  syncChatsButton();
  if (focus) $('question').focus();
}
$('newchat').onclick = $('chatsnew').onclick = () => { showView('chat', { focus: false }); newChat(); };

// ---- chat ----
const IDEAS = ['Summarise everything in my library', 'What are the key dates and amounts?', 'Which documents disagree with each other?'];
function emptyState() {
  const box = el('div', 'empty');
  box.dataset.empty = '1';
  const f = !scopeGone && scope.folders.length === 1 && !scope.docs.length ? scope.folders[0] : null;
  if (f != null) {
    const n = folderTotal(f);
    box.append(el('h2', '', `Ask ${folderName(f)} anything.`),
      el('div', 'muted', `${n == null ? 'The documents' : plural(n, 'document')} in this folder and its subfolders. Documents added later are included.`));
  } else {
    const hint = el('div', 'muted');
    hint.append(el('span', 'wide', 'Add files on the left, or drop them here.'),
      el('span', 'narrow', 'Add files in Documents (the folder button at the top left).'),
      ' Answers cite the files and pages they came from: click a source to read the passage.');
    box.append(el('h2', '', 'Ask your documents anything.'), hint);
  }
  const ideas = el('div', 'ideas');
  for (const t of f != null ? [`Summarise everything in ${folderName(f)}`, ...IDEAS.slice(1)] : IDEAS) ideas.append(Object.assign(el('button', '', t), { type: 'button', onclick: () => ask(t) }));
  box.append(ideas);
  $('log').replaceChildren(box);
}

function turn(cls) {
  $('log').querySelector('.empty')?.remove();
  const t = el('div', 'turn ' + cls);
  $('log').append(t);
  return t;
}
const scroll = () => $('log').scrollTop = $('log').scrollHeight;
const nearBottom = () => { const l = $('log'); return l.scrollHeight - l.scrollTop - l.clientHeight < 80; };

function showQuestion(text, only) {
  const t = turn('user');
  t.append(el('div', '', text));
  if (only) t.append(el('span', 'scoped', 'Only in ' + only));
}

function answerTurn() {
  const bot = turn('bot'), body = el('div', 'body');
  bot.append(body);
  return { bot, body };
}

// What follows an answer: its sources, Copy, the model and its tokens, and how it came about.
// r: an answer as /api/chat returns it, a stream's meta and done events together, or a saved one.
function finishAnswer(bot, r) {
  const foot = el('div', 'foot');
  for (const s of r.sources || []) foot.append(sourceChip(s, r.excerpts || []));
  const copy = Object.assign(el('button', '', 'Copy'), { type: 'button' });
  copy.onclick = async () => { await navigator.clipboard.writeText(r.answer).catch(() => {}); copy.textContent = 'Copied'; setTimeout(() => copy.textContent = 'Copy', 1500); };
  foot.append(copy);
  const u = r.usage;
  if (r.model_label) {
    const tok = el('span', 'tok', u ? `${r.model_label} · ${fmtInt(u.input + u.output)} tokens` : r.model_label);
    if (u) tok.title = `in ${u.input} · out ${u.output} · cache read ${u.cache_read} · cache write ${u.cache_write}`
      + (r.status && r.status !== 'done' ? ' (estimated: the answer ended early)' : '');
    foot.append(tok);
  }
  bot.append(foot);
  if (r.overview) bot.append(el('div', 'switched', 'No passage matched this question, so the model was given the titles and summaries of your documents.'));
  if (r.left_out) bot.append(el('div', 'switched', `This conversation is too long for the local model: it was not given the ${r.left_out === 1 ? 'earliest question and its answer' : `${r.left_out} earliest questions and their answers`}.`));
  if (r.switched) bot.append(el('div', 'switched', `Switched to ${r.model_label}: ${r.switched}`));
  if (r.status === 'stopped') bot.append(el('div', 'switched', 'Stopped. The answer ends here.'));
  if (r.status === 'broken') bot.append(el('div', 'switched bad', `The answer broke off here. ${r.error || ''}`));
}

function showExchange(q, a) {  // a saved question and its answer
  showQuestion(q.content, onlyText((a?.scope_folders || []).map(f => f.name), a?.scope_names || []));
  if (!a) return;
  const { bot, body } = answerTurn();
  if (a.status === 'answering') {  // see watchChat
    body.append(dots());
    bot.append(el('div', 'switched', 'Still being answered. It shows here when it is done.'));
  } else if (!a.content) {  // stopped before its first words, or failed: as the page showed it then
    bot.classList.add('err');
    body.textContent = a.status === 'stopped' ? 'Stopped before the answer began.' : a.error || 'No answer.';
  } else {
    body.textContent = a.content;
    finishAnswer(bot, { ...a, answer: a.content });
  }
}
const dots = () => { const d = el('span', 'dots'); d.append(el('i'), el('i'), el('i')); return d; };

// Answers come from /api/chat/stream as Server-Sent Events, read from fetch's body as they
// arrive, so the text shows as it is written. Without that (an older browser, or a server
// without the route), /api/chat gives the whole answer at once.
const canStream = typeof TextDecoder === 'function' && typeof ReadableStream === 'function' && 'body' in Response.prototype;
let answering = null;  // the answer being written (its AbortController): Stop aborts the fetch
const stopAnswer = () => answering?.abort();
$('stop').onclick = () => { stopAnswer(); $('question').focus(); };

// A stopped answer is kept as far as the page showed it, which the server can't know: it
// notices a closed stream only when a piece fails to go out, and the piece or two sent just
// before never showed. live: the streamed answer being written, from its start event, and
// how many characters of it are shown (code points, as the server counts).
let live = null;
function reportStop(l) {
  if (!l || l.reported) return;  // once: Log out reports it before pagehide, and so does Stop
  l.reported = true;
  fetch('/api/chat/stop', { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json', ...OWN },
    body: JSON.stringify({ conversation_id: l.conversation_id, answer_id: l.answer_id, shown: l.shown }) }).catch(() => {});
}
addEventListener('pagehide', () => reportStop(live));  // a reload or a closed tab stops it too

async function fetchAnswer(req, signal, on) {
  const opts = { method: 'POST', signal, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) };
  if (canStream) {
    const res = await fetch('/api/chat/stream', { ...opts, headers: { ...opts.headers, ...OWN, 'X-API-Key': key.value } });
    const type = res.headers.get('content-type') || '';
    if (type.startsWith('text/event-stream')) return readEvents(res, on);
    if (!((res.status === 404 || res.status === 405) && !type.includes('json'))) {  // not "no such route": the server's answer
      const data = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
      if (loginGone(res.status, data)) return toLogin();
      if (!res.ok) throw Object.assign(new Error(data.error || res.statusText), { status: res.status, data });
      on.meta(data);
      return data;
    }
  }
  const r = await api('/api/chat', opts);
  on.meta(r);
  return r;
}

// The events: start (the conversation, and the answer's id), meta, delta (a piece of text),
// then done, or error ({error, status, kept}: kept when the words so far stand). A stream
// that ends without either broke off on the way.
async function readEvents(res, on) {
  let done = null;
  const handle = block => {
    let name = 'message', data = '';
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      else if (line.startsWith('data:')) data += line.slice(5).replace(/^ /, '');
    }
    if (!data) return;  // a comment
    const d = JSON.parse(data);
    if (name === 'start') on.start(d);
    else if (name === 'meta') on.meta(d);
    else if (name === 'delta') on.delta(d);
    else if (name === 'done') done = d;
    else if (name === 'error') throw Object.assign(new Error(d.error), { status: d.status, kept: d.kept, data: d });
  };
  const reader = res.body?.getReader();
  if (!reader) (await res.text()).split('\n\n').forEach(handle);  // nothing to read as it comes: all of it at once
  else {
    const dec = new TextDecoder();
    let buf = '';
    try {
      for (;;) {
        let part;
        try { part = await reader.read(); } catch (err) {
          if (err.name === 'AbortError') throw err;
          throw Object.assign(new Error('The connection to the server was lost.'), { kept: true });
        }
        if (part.done) break;
        buf += dec.decode(part.value, { stream: true });
        for (let i; (i = buf.indexOf('\n\n')) >= 0; buf = buf.slice(i + 2)) handle(buf.slice(0, i));
      }
    } catch (err) {
      reader.cancel().catch(() => {});
      throw err;
    }
  }
  if (!done) throw Object.assign(new Error('The connection to the server was lost.'), { kept: true });
  return done;
}

function setBusy(on) {
  busy = on;
  $('send').hidden = on; $('stop').hidden = !on;
  if (typeof syncAnswering === 'function') syncAnswering();
}

async function ask(question) {
  if (busy || !question) return;
  setBusy(true);
  const shown = chatShown, ctl = answering = new AbortController();
  showQuestion(question, scopeAll() || scopeGone ? '' : onlyText(scope.folders.map(folderName), scope.docs.map(docName)));
  const { bot, body } = answerTurn();
  body.append(dots());
  scroll();
  let meta = {}, text = '', stopped = false, mine = null;
  // The conversation the server keeps this in, from the start event on (null: it was deleted
  // meanwhile). Its title is the list's business after that: renamed while this is answered,
  // the header keeps the new one.
  const follow = m => {
    if (shown !== chatShown || !m || !('conversation_id' in m)) return;
    if (!m.conversation_id) setChat(null);
    else if (m.conversation_id !== chat?.id) { setChat({ id: m.conversation_id, title: m.title }); loadChats(); }
  };
  try {
    if (scopeGone) throw new Error(GONE);  // never every document unasked
    // A folder with nothing in it is the server's to say (it looks at the folder now): this page's
    // listing may be older than an upload made into it from another tab or device.
    const req = { question, model: $('chatModel').value, ...(scope.docs.length ? { doc_ids: scope.docs } : {}),
                  ...(scope.folders.length ? { folder_ids: scope.folders } : {}), ...(chat ? { conversation_id: chat.id } : {}) };
    asking = question;
    const r = await fetchAnswer(req, ctl.signal, {
      start: s => { asking = ''; live = mine = { conversation_id: s.conversation_id, answer_id: s.answer_id, shown: 0 }; follow(s); },
      meta: m => { meta = m; follow(m); },
      delta: piece => {
        const pinned = nearBottom();
        if (!text) { body.replaceChildren(); bot.classList.add('live'); }
        text += piece;
        if (mine) mine.shown += [...piece].length;
        body.append(piece);
        if (pinned) scroll();
      },
    });
    const pinned = nearBottom();
    bot.classList.remove('live');
    body.textContent = r.answer;
    finishAnswer(bot, { ...meta, ...r });
    follow(r);
    if (pinned) scroll();
  } catch (err) {
    bot.classList.remove('live');
    stopped = err.name === 'AbortError';
    if (stopped || (err.kept && !err.data)) reportStop(mine);  // stopped here, or the connection was lost
    follow(err.data);  // a failed question is kept too: in a new conversation, made for it
    if ((stopped || err.kept) && text) {  // the words so far stand, and are kept with the conversation
      finishAnswer(bot, { ...meta, answer: text, status: stopped ? 'stopped' : 'broken', error: err.message });
    } else {
      bot.classList.add('err');
      body.textContent = stopped ? 'Stopped before the answer began.' : err.message;
      if (/key/i.test(err.message)) askKey();
      if (/^None of the chosen|has no documents yet|have no documents yet/.test(err.data?.error || '')) lib.load();  // the chip catches up
      if (err.status === 404 && shown === chatShown && chat) {  // the conversation was deleted (another tab)
        setChat(null);
        body.textContent += ' Ask again to start a new one.';
      }
    }
  }
  if (answering === ctl) answering = null;
  if (live === mine) live = null;
  asking = '';
  setBusy(false);
  if (shown === chatShown) loadChats();
  if (stopped) setTimeout(loadChats, 1500);  // the server keeps a stopped answer once it notices
}

const qbox = $('question');
qbox.oninput = () => { qbox.setCustomValidity(''); qbox.style.height = 'auto'; qbox.style.height = qbox.scrollHeight + 'px'; };
qbox.onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('ask').requestSubmit(); } };
$('ask').onsubmit = e => {
  e.preventDefault();
  const question = qbox.value.trim();
  if (!question || busy) return;
  const max = MODELS.max_question_chars || 20000, n = [...question].length;
  if (n > max) {  // said here, so the text stays in the box to shorten
    qbox.setCustomValidity(`A question can be ${fmtInt(max)} characters at most; this one has ${fmtInt(n)}.`);
    qbox.reportValidity();
    return;
  }
  qbox.value = ''; qbox.oninput();
  ask(question);
};

// An answer being written is stopped first, while the session still lets the stop report through
// (pagehide's would get 401, and the server would keep more than the page showed). The Claude key
// typed here is forgotten, so whoever logs in next in this browser isn't using it.
$('logoutform').addEventListener('submit', () => {
  leave();
  reportStop(live);
  stopAnswer();
  key.value = ''; store.set('claudeKey', '');
});

// ---- start: after all three scripts have run ----
function start() {
  // Log out, when there is a login (body.login: where the row is short of room, Instructions reads Rules).
  api('/api/session').then(s => { $('logout').hidden = !s.login; document.body.classList.toggle('login', !!s.login); }).catch(() => {});
  // What was being written when the session ended (toLogin), back in its box.
  try {
    const draft = JSON.parse(sessionStorage.getItem('draft') || 'null');
    sessionStorage.removeItem('draft');
    if (draft?.question && !qbox.value) { qbox.value = draft.question; qbox.oninput(); }
    if (draft?.rule) { $('newrule').value = draft.rule; $('rulesbtn').click(); }
  } catch {}
  showScope();
  const lastChat = +store.get('chatConversation');
  if (lastChat) openChat(lastChat); else emptyState();  // the conversation open before a reload
  loadChats();
  loadModels();
  loadRules();
  docsStart();  // Documents: where the URL says, and the library
  pollJobs();  // uploads still queued or being read, and failed ones, also after a reload
}
addEventListener('DOMContentLoaded', start);
