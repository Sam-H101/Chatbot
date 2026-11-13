/* The documents view: the library model (folders and documents as last listed), switching between
   Chat and Documents with the URL's hash, the folder tree, the listing (grid or list), selection,
   the detail panel, menus and sheets, the folder dialogs and move picker, drag and drop, and the
   dialog that chooses what questions search. Shares app.js's global scope. */

// ---- state ----
const lib = { folders: new Map(), docs: new Map(), kids: new Map(), docsIn: new Map(), loadedAt: 0, loaded: false, error: null };
// kids: 'root' | folder id -> [folder ids, by name]; docsIn: 'unfiled' | folder id -> [document ids, newest first]
const view = { mode: 'chat', place: { type: 'root' }, doc: null, q: '', qScope: 'here', flat: false,
  sort: store.get('docsSort') || 'added:desc', relevance: true, layout: store.get('docsView') === 'list' ? 'list' : 'grid',
  filters: { types: new Set(), from: '', to: '', amount: false, due: false, suggested: null, models: new Set(), partial: false }, selecting: false,
  order: [], focusKey: null, server: { key: '', docs: [], hits: [] } };
const sel = { ids: new Set(), anchor: null };  // documents only, never folders
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
const SORTS = {
  name: ['Name', 'A to Z', 'Z to A'], added: ['Date added', 'Oldest first', 'Newest first'],
  date: ['Document date', 'Oldest first', 'Newest first'], type: ['Type', 'A to Z', 'Z to A'],
  size: ['Size', 'Smallest first', 'Largest first'], amount: ['Amount', 'Lowest first', 'Highest first'],
};
const SHORT_SORT = { name: 'Name', added: 'Added', date: 'Date', type: 'Type', size: 'Size', amount: 'Amount' };

// ---- formatting ----
const docTitle = d => d.title || d.filename;
const norm = s => String(s ?? '').normalize('NFD').replace(/\p{M}/gu, '').toLowerCase();
const isoParts = iso => { const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso || ''); return m ? [+m[1], +m[2], +m[3]] : null; };
// Dates from their ISO parts: new Date('2026-11-03') would be UTC midnight, a day early west of it.
function fmtDate(iso, { year = true, month = 'short' } = {}) {
  const p = isoParts(iso);
  if (!p) return '';
  const showYear = year || p[0] !== new Date().getFullYear();
  return new Date(p[0], p[1] - 1, p[2]).toLocaleDateString(undefined, { day: 'numeric', month, ...(showYear ? { year: 'numeric' } : {}) });
}
function fmtAmount(n, cur) {
  if (n == null) return '';
  if (cur) { try { return new Intl.NumberFormat(undefined, { style: 'currency', currency: cur }).format(n); } catch {} }
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + (cur ? ' ' + cur : '');
}
// "Due in 5 days" (accent) within 14 days, "Due 3 Dec" later, "Was due 3 Nov" once past: no alarm,
// since paid or not isn't tracked.
function dueInfo(iso) {
  const p = isoParts(iso);
  if (!p) return null;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const days = Math.round((new Date(p[0], p[1] - 1, p[2]) - today) / 864e5);
  if (days < 0) return { text: `Was due ${fmtDate(iso, { year: false })}`, soon: false };
  if (days === 0) return { text: 'Due today', soon: true };
  if (days <= 14) return { text: `Due in ${plural(days, 'day')}`, soon: true };
  return { text: `Due ${fmtDate(iso, { year: false })}`, soon: false };
}
const utc = s => new Date(String(s).replace(' ', 'T') + 'Z');
function fmtAdded(s, time = false, year = true) {
  const d = utc(s);
  if (isNaN(d)) return '';
  const opts = { day: 'numeric', month: 'short', ...(year || d.getFullYear() !== new Date().getFullYear() ? { year: 'numeric' } : {}) };
  return time ? d.toLocaleString(undefined, { ...opts, hour: '2-digit', minute: '2-digit' }) : d.toLocaleDateString(undefined, opts);
}
const fmtSize = b => b == null ? '—' : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`;
function kindLabel(d) {
  const ext = (/\.([A-Za-z0-9]{1,8})$/.exec(d.filename) || [])[1]?.toLowerCase() || '';
  return { jpeg: 'JPG', markdown: 'MD', heif: 'HEIC', htm: 'HTML' }[ext] || (ext && ext.length <= 4 ? ext.toUpperCase() : 'TXT');
}
const isBill = d => /receipt|bill|invoice|statement/i.test(d.doc_type || '');
// Title case, plural: "manual" -> "Manuals", "utility bill" -> "Utility Bills", "policy" -> "Policies".
function pluralName(t) {
  const words = t.trim().split(/\s+/).map(w => w[0].toUpperCase() + w.slice(1));
  let last = words.pop();
  if (/[^aeiou]y$/i.test(last)) last = last.slice(0, -1) + 'ies';
  else if (/(s|x|z|ch|sh)$/i.test(last)) last += 'es';
  else last += 's';
  return [...words, last].join(' ');
}
// Text with the search words marked, built from text nodes only.
function marked(text, words) {
  const f = document.createDocumentFragment();
  if (!words.length) { f.append(text); return f; }
  const n = norm(text);
  if (n.length !== text.length) { f.append(text); return f; }  // normalising changed the length: no offsets to trust
  const hits = [];
  for (const w of words) for (let i = n.indexOf(w); i >= 0 && w; i = n.indexOf(w, i + w.length)) hits.push([i, i + w.length]);
  hits.sort((a, b) => a[0] - b[0]);
  let at = 0;
  for (const [a, b] of hits) {
    if (a < at) continue;
    f.append(text.slice(at, a), el('mark', '', text.slice(a, b)));
    at = b;
  }
  f.append(text.slice(at));
  return f;
}

// ---- the library model ----
let loadSeq = 0;
lib.load = async () => {
  const mine = ++loadSeq;
  let folders, docs;
  try { [folders, docs] = await Promise.all([api('/api/folders'), api('/api/documents')]); } catch (err) {
    if (mine !== loadSeq) return;
    lib.error = err.message;
    if (!lib.loaded) render(); else toast(`Couldn't refresh your documents: ${err.message}`, { error: true });
    return;
  }
  if (mine !== loadSeq) return;
  lib.error = null;
  lib.loaded = true;
  lib.loadedAt = Date.now();
  lib.folders = new Map(folders.map(f => [f.id, f]));
  lib.docs = new Map(docs.map(d => [d.id, d]));
  rebuild();
  $('count').textContent = docs.length ? plural(docs.length, 'file') : '';
  knowFolders(folders);
  knowDocs(docs);
  checkPlace();
  render();
};

// Maps, paths and counts from the folders' parents and the documents' folders: run after every
// load, and after a change made here before the server has answered.
function rebuild() {
  lib.kids = new Map([['root', []]]);
  lib.docsIn = new Map([['unfiled', []]]);
  for (const f of lib.folders.values()) { lib.kids.set(f.id, []); lib.docsIn.set(f.id, []); }
  for (const f of lib.folders.values()) lib.kids.get(f.parent_id != null && lib.folders.has(f.parent_id) ? f.parent_id : 'root').push(f.id);
  for (const ids of lib.kids.values()) ids.sort((a, b) => collator.compare(lib.folders.get(a).name, lib.folders.get(b).name));
  const walk = (key, names, depth) => {
    for (const id of lib.kids.get(key) || []) {
      const f = lib.folders.get(id);
      f.names = [...names, f.name];
      f.depth = depth;
      if (depth < 64) walk(id, f.names, depth + 1);
    }
  };
  walk('root', [], 0);
  for (const d of [...lib.docs.values()].sort((a, b) => b.id - a.id)) {
    lib.docsIn.get(d.folder_id != null && lib.folders.has(d.folder_id) ? d.folder_id : 'unfiled').push(d.id);
  }
  const total = id => {
    const f = lib.folders.get(id);
    f.count = lib.docsIn.get(id).length;
    f.total = f.count + (lib.kids.get(id) || []).reduce((n, k) => n + total(k), 0);
    return f.total;
  };
  for (const id of lib.kids.get('root')) total(id);
  const types = new Map(), curs = new Set(['GBP', 'EUR', 'USD']);
  for (const d of lib.docs.values()) {
    if (d.doc_type) types.set(norm(d.doc_type), d.doc_type);
    if (d.currency) curs.add(d.currency);
  }
  $('typelist').replaceChildren(...[...types.values()].sort(collator.compare).map(t => new Option(t)));
  $('curlist').replaceChildren(...[...curs].sort().map(c => new Option(c)));
}
const isUnfiled = d => d.folder_id == null || !lib.folders.has(d.folder_id);
const unfiledIds = () => lib.docsIn.get('unfiled') || [];
// A folder and every folder inside it.
function subtree(id) {
  const out = new Set();
  const add = k => { if (out.has(k) || !lib.folders.has(k)) return; out.add(k); for (const c of lib.kids.get(k) || []) add(c); };
  add(id);
  return out;
}
function subtreeDocs(id) { const ids = []; for (const f of subtree(id)) ids.push(...lib.docsIn.get(f)); return ids; }
function pathText(fid) {
  if (fid == null) return 'Unfiled';
  const f = lib.folders.get(fid);
  return f?.names ? f.names.join(' › ') : lib.loaded ? 'a deleted folder' : 'a folder';
}
// One line's worth of a path: the last two folders, after "… ›" when it is deeper. The full path
// goes in the title.
function shortPath(fid) {
  const f = fid != null && lib.folders.get(fid);
  return f?.names && f.names.length > 2 ? `… › ${f.names.slice(-2).join(' › ')}` : pathText(fid);
}
const folderTotal = id => lib.loaded && lib.folders.has(id) ? lib.folders.get(id).total : null;
// An existing folder for a path like "Household/Bills" (each level compared ignoring case), or null.
function findPath(path) {
  let key = 'root', id = null;
  for (const seg of String(path).split('/')) {
    id = (lib.kids.get(key) || []).find(k => norm(lib.folders.get(k).name) === norm(seg.trim())) ?? null;
    if (id == null) return null;
    key = id;
  }
  return id;
}

// ---- places, history and switching views ----
const placeKey = p => p.type === 'folder' ? 'f' + p.id : p.type;
const samePlace = (a, b) => placeKey(a) === placeKey(b);
function placeName(p) {
  return p.type === 'all' ? 'All documents' : p.type === 'unfiled' ? 'Unfiled' : p.type === 'folder'
    ? lib.folders.get(p.id)?.name || 'Folder' : 'Library';
}
function parentPlace(p) {
  if (p.type !== 'folder') return { type: 'root' };
  const f = lib.folders.get(p.id);
  return f?.parent_id != null && lib.folders.has(f.parent_id) ? { type: 'folder', id: f.parent_id } : { type: 'root' };
}
function hashFor(p, doc = null) {
  const base = p.type === 'all' ? '#docs/all' : p.type === 'unfiled' ? '#docs/unfiled' : p.type === 'folder' ? `#docs/f/${p.id}` : '#docs';
  return base + (doc != null ? `/d/${doc}` : '');
}
function parseHash(h) {
  const m = /^#docs(?:\/(all|unfiled|f\/(\d+)))?(?:\/d\/(\d+))?\/?$/.exec(h || '');
  if (!m) return null;
  const place = m[1] === 'all' ? { type: 'all' } : m[1] === 'unfiled' ? { type: 'unfiled' } : m[2] ? { type: 'folder', id: +m[2] } : { type: 'root' };
  return { place, doc: m[3] ? +m[3] : null };
}
// Entries this page pushed carry {app: true}; a detail opened on a phone or a medium screen is one
// (detail: true), so Back closes it. Search, sort, filters, selection and dialogs are not.
let depth = 0;
function nav(hash, how = 'push', extra = {}) {
  if (how === 'push' && (location.hash || '') === (hash || '')) how = 'replace';
  const state = { app: true, n: how === 'push' ? ++depth : depth, ...extra };
  history[how === 'push' ? 'pushState' : 'replaceState'](state, '', hash || location.pathname + location.search);
}

function route() {
  const r = parseHash(location.hash);
  if (!r) { setMode('chat'); render(); return; }
  if (!samePlace(r.place, view.place)) resetPlaceState();
  view.place = r.place;
  view.doc = r.doc;
  store.set('docsAt', hashFor(r.place));
  setMode('docs');
  if (lib.loaded) checkPlace();
  render();
}
addEventListener('popstate', route);

// The place or document shown was deleted (another tab, a folder delete): the nearest place left.
function checkPlace() {
  if (view.place.type === 'folder' && !lib.folders.has(view.place.id)) {
    view.place = { type: 'root' };
    view.doc = null;
    store.set('docsAt', '#docs');
    if (view.mode === 'docs') { nav(hashFor(view.place), 'replace'); toast('That folder no longer exists'); }
  }
  if (view.doc != null && !lib.docs.has(view.doc)) {
    view.doc = null;
    if (view.mode === 'docs') { nav(hashFor(view.place), 'replace'); toast('That document was deleted'); }
  }
}

function resetPlaceState() {
  view.q = '';
  $('dq').value = '';
  view.qScope = 'here';  // each place searches itself first, as its search box says
  view.relevance = true;
  sel.ids.clear();
  sel.anchor = null;
  view.selecting = false;
  view.focusKey = null;
}

function setMode(mode) {
  const docs = mode === 'docs';
  view.mode = mode;
  $('chatview').hidden = docs;
  $('docsview').hidden = !docs;
  document.body.classList.toggle('docs', docs);
  document.body.classList.remove('chatsopen');
  syncChatsButton();
  if (docs) { $('todocs').setAttribute('aria-current', 'page'); $('tochat').removeAttribute('aria-current'); }
  else { $('tochat').setAttribute('aria-current', 'page'); $('todocs').removeAttribute('aria-current'); }
  if (!docs) document.body.classList.remove('hasbar', 'panelopen');
  syncAnswering();
}

// Chat <-> Documents. Both stay mounted: an answer being written keeps going, and the place,
// selection and scroll positions are where they were.
function showView(mode, { focus = true } = {}) {
  if (mode !== view.mode) {
    if (mode === 'docs') { view.doc = null; nav(hashFor(view.place)); } else nav('');
    setMode(mode);
    render();
  }
  if (focus) focusView();
}
function focusView() {
  if (view.mode === 'docs') { if (!focusList()) $('dtitle').focus({ preventScroll: true }); return; }
  if (isS.matches || coarse.matches) $('docsbtn').offsetParent ? $('docsbtn').focus() : $('question').focus();
  else $('question').focus();
}

function openPlace(p, { focus = false } = {}) {
  const same = view.mode === 'docs' && samePlace(p, view.place);
  if (!same) resetPlaceState();
  const wasChat = view.mode !== 'docs';
  view.place = p;
  view.doc = null;
  store.set('docsAt', hashFor(p));
  if (p.type === 'folder') expandTo(p.id);
  nav(hashFor(p), same ? 'replace' : 'push');
  setMode('docs');
  render();
  announce(countText());
  if (focus || wasChat) focusList();
}

// Documents at a document: its folder (Unfiled if none) with its detail open. From job rows'
// Show and the sources dialog's Show in Documents.
function showDocument(id) {
  const d = lib.docs.get(id);
  if (!d) { lib.load(); toast('That document is not in the library any more'); return; }
  openPlace(isUnfiled(d) ? { type: 'unfiled' } : { type: 'folder', id: d.folder_id });
  openDoc(id);
}

let jobCounts = [0, 0];
function syncDocsButton(working = jobCounts[0], waiting = jobCounts[1]) {
  jobCounts = [working, waiting];
  const b = $('docsbtn'), docs = view.mode === 'docs';
  b.querySelector('use').setAttribute('href', docs ? '#i-chat' : '#i-folder');
  $('jobcount').hidden = docs;
  const jobs = working || waiting
    ? `: ${plural(working, 'upload')} under way${waiting ? `, ${waiting} waiting for you` : ''}` : '';
  b.setAttribute('aria-label', docs ? (busy ? 'Chat, an answer is being written' : 'Chat') : 'Documents' + jobs);
  b.title = docs ? 'Chat' : 'Documents';
}
function syncAnswering() {
  const on = busy && view.mode === 'docs';
  $('docsbtn').querySelector('.adot').hidden = !on;
  $('tochat').querySelector('.adot').hidden = !on;
  if (on) $('tochat').setAttribute('aria-label', 'Chat, an answer is being written'); else $('tochat').removeAttribute('aria-label');
  syncDocsButton();
}
$('docsbtn').onclick = () => showView(view.mode === 'docs' ? 'chat' : 'docs');
$('tochat').onclick = () => showView('chat');
$('todocs').onclick = () => showView('docs');

// ---- uploads: where they go, and where the queue sits ----
const autoFileOn = () => store.get('autoFile') !== '0';
function uploadTarget() {
  const p = view.place;
  const fid = view.mode === 'docs' && p.type === 'folder' && lib.folders.has(p.id) ? p.id : null;
  return { folder_id: fid, auto_file: autoFileOn() };
}
function syncInto() {
  const t = uploadTarget(), f = t.folder_id != null && lib.folders.get(t.folder_id);
  $('intoname').textContent = f ? shortPath(f.id) : `Unfiled${t.auto_file ? ' · auto-file' : ''}`;
  $('into').title = f ? pathText(f.id) : '';
  $('autoFileBox').hidden = !!f;
  $('staynote').hidden = !f;
  $('staynote').textContent = f ? `Uploads here stay in ${f.name}` : '';
  $('staynote').title = f ? `Uploads here stay in ${f.name}` : '';
  $('avtitle').textContent = `Add to ${f ? f.name : 'Unfiled'}`;
}
$('autoFile').checked = autoFileOn();
$('autoFile').onchange = e => { store.set('autoFile', e.target.checked ? '1' : '0'); syncInto(); };
$('addfiles').onclick = () => $('file').click();
// The same queue node, so rows keep their state and focus: under the Add box on wider screens,
// in Documents' Uploads box on a phone.
function placeUploads() {
  const home = isS.matches ? $('uploadsbox') : $('queuehome');
  if ($('queue').parentElement !== home) home.append($('queue'));
}
isS.addEventListener('change', () => { placeUploads(); render(); });
isL.addEventListener('change', () => renderPanel());

// ---- toast and live region ----
let toastTimer = null;
function toast(msg, { action = null, error = false, ms = 6000 } = {}) {
  const t = $('toast'), a = $('toastact');
  $('toasttext').textContent = msg;
  a.hidden = !action;
  if (action) { a.textContent = action[0]; a.onclick = () => { hideToast(); action[1](); }; }
  t.classList.toggle('error', error);
  t.hidden = false;
  clearTimeout(toastTimer);
  const arm = () => { toastTimer = setTimeout(() => { if (t.matches(':hover, :focus-within')) arm(); else hideToast(); }, ms); };
  arm();
}
function hideToast() { clearTimeout(toastTimer); $('toast').hidden = true; }
let announceTimer = null;
function announce(msg) {
  clearTimeout(announceTimer);
  $('announce').textContent = '';
  announceTimer = setTimeout(() => { $('announce').textContent = msg; }, 60);
}

// ---- rendering: everything from the state ----
function render() {
  renderTree();
  syncInto();
  if (view.mode !== 'docs') return;
  renderDocs();
  renderPanel();
}

// ---- the folder tree (left column) ----
const treeOpen = new Set(ints(store.get('treeOpen')));
let treeKey = 'root';  // the row with the tree's one tab stop
function saveTree() { store.set('treeOpen', JSON.stringify([...treeOpen].filter(id => lib.folders.has(id)))); }
function expandTo(id) {
  let f = lib.folders.get(lib.folders.get(id)?.parent_id);
  let changed = false;
  for (let n = 0; f && n < 64; n++, f = lib.folders.get(f.parent_id)) if (!treeOpen.has(f.id)) { treeOpen.add(f.id); changed = true; }
  if (changed) saveTree();
}

function renderTree() {
  const t = $('tree'), focused = t.contains(document.activeElement) ? document.activeElement.closest('.tnode')?.dataset.key : null;
  if (!lib.loaded) { t.replaceChildren(el('li', 'tnote', lib.error ? `Couldn't load: ${lib.error}` : 'Loading…')); return; }
  const cur = view.mode === 'docs' ? placeKey(view.place) : null;
  // In Chat, the folders questions are limited to are marked.
  const asked = new Set(view.mode !== 'docs' && !scopeGone ? scope.folders : []);
  const node = o => { const li = treeNode(o, cur); if (asked.has(o.folder)) li.classList.add('inscope'); return li; };
  const root = node({ key: 'root', label: 'Library', ic: 'home', level: 1, count: lib.docs.size, drop: 'root', expanded: true, fixed: true });
  const group = el('ul');
  group.setAttribute('role', 'group');
  const add = (key, level, ul) => {
    for (const id of lib.kids.get(key) || []) {
      const f = lib.folders.get(id), has = lib.kids.get(id).length > 0, open = has && treeOpen.has(id);
      const li = node({ key: 'f' + id, label: f.name, ic: cur === 'f' + id ? 'folder-open' : 'folder', level, count: f.total,
                        drop: 'f' + id, expanded: has ? open : undefined, title: pathText(id), folder: id });
      if (open) {
        const sub = el('ul');
        sub.setAttribute('role', 'group');
        add(id, level + 1, sub);
        li.append(sub);
      }
      ul.append(li);
    }
  };
  add('root', 2, group);
  if (!lib.folders.size) {
    const n = el('li', 'tnote');
    n.setAttribute('role', 'none');
    n.append('No folders yet.', btn('New folder', () => newFolder(null), { cls: 'open' }));
    group.append(n);
  }
  const hasSug = unfiledIds().some(id => lib.docs.get(id).suggested_folder);
  group.append(node({ key: 'unfiled', label: 'Unfiled', ic: 'inbox', level: 2, count: unfiledIds().length, drop: 'unfiled', dot: hasSug }));
  root.append(group);
  t.replaceChildren(node({ key: 'all', label: 'All documents', ic: 'stack', level: 1, count: lib.docs.size, drop: 'all' }), root);
  const items = [...t.querySelectorAll('.tnode')];
  if (!items.some(li => li.dataset.key === treeKey)) treeKey = cur && items.some(li => li.dataset.key === cur) ? cur : 'root';
  for (const li of items) li.tabIndex = li.dataset.key === treeKey ? 0 : -1;
  if (focused) (items.find(li => li.dataset.key === focused) || items.find(li => li.dataset.key === treeKey))?.focus({ preventScroll: true });
}

function treeNode({ key, label, ic, level, count, drop, expanded, title, folder, dot, fixed }, cur) {
  const li = el('li', 'tnode');
  li.setAttribute('role', 'treeitem');
  li.dataset.key = key;
  li.setAttribute('aria-level', String(level));
  if (expanded !== undefined) li.setAttribute('aria-expanded', String(expanded));
  li.setAttribute('aria-selected', String(cur === key));
  li.setAttribute('aria-label', `${label}, ${plural(count, 'document')}`);
  if (dot) li.setAttribute('aria-description', 'has filing suggestions');
  const row = el('div', 'trow');
  row.style.setProperty('--lvl', String(Math.min(level - 1, 8)));
  row.dataset.drop = drop;
  if (title) row.title = title;
  const chev = el('span', 'chev' + (expanded === undefined || fixed ? ' none' : ''));
  chev.append(icon('chev'));
  const n = el('span', 'tcount', String(count));
  if (dot) n.append(el('span', 'tdot'));
  row.append(chev, icon(ic), el('span', 'tname', label), n);
  const askLabel = folder != null ? `Ask about ${label}` : key === 'unfiled' ? `Ask about the ${plural(count, 'Unfiled document')}` : 'Ask all documents';
  const ask = btn('', null, { cls: 'tbtn ask', aria: askLabel, title: askLabel, icon: 'chat' });
  const more = btn('', null, { cls: 'tbtn more', aria: `More for ${label}`, title: 'More actions', icon: 'more' });
  ask.tabIndex = more.tabIndex = -1;
  if ((key === 'unfiled' || folder != null) && !count) { ask.disabled = true; ask.title = `${label} has no documents yet`; }
  row.append(ask, more);
  if (folder != null && fine.matches) row.draggable = true;
  li.append(row);
  return li;
}

function treePlace(key) {
  return key === 'all' ? { type: 'all' } : key === 'root' ? { type: 'root' } : key === 'unfiled' ? { type: 'unfiled' } : { type: 'folder', id: +key.slice(1) };
}
function toggleTree(id, open = !treeOpen.has(id)) {
  if (open) treeOpen.add(id); else treeOpen.delete(id);
  saveTree();
  renderTree();
}
function askPlace(key) {
  if (key === 'all' || key === 'root') askAbout({});
  else if (key === 'unfiled') askAbout({ docs: unfiledIds() });
  else askAbout({ folders: [+key.slice(1)] });
}

$('tree').addEventListener('click', e => {
  const li = e.target.closest('.tnode');
  if (!li || !e.target.closest('.trow')) return;
  const key = li.dataset.key;
  treeKey = key;
  if (e.target.closest('.ask')) { askPlace(key); return; }
  if (e.target.closest('.more')) { placeMenu(key, e.target.closest('.more')); return; }
  if (e.target.closest('.chev') && li.hasAttribute('aria-expanded') && key[0] === 'f') { toggleTree(+key.slice(1)); return; }
  openPlace(treePlace(key));
});
$('tree').addEventListener('contextmenu', e => {
  const li = e.target.closest('.tnode');
  if (!li || !fine.matches) return;
  e.preventDefault();
  placeMenu(li.dataset.key, null, { x: e.clientX, y: e.clientY });
});
let typed = '', typedAt = 0;
function typeAhead(e, items, label, from) {  // the next item whose label starts with what was typed
  if (e.key.length !== 1 || e.ctrlKey || e.metaKey || e.altKey || e.key === ' ') return false;
  typed = Date.now() - typedAt > 700 ? e.key.toLowerCase() : typed + e.key.toLowerCase();
  typedAt = Date.now();
  const i = items.indexOf(from), order = [...items.slice(i + 1), ...items.slice(0, i + 1)];
  const hit = order.find(x => norm(label(x)).startsWith(typed));
  if (hit) hit.focus();
  return true;
}
$('tree').addEventListener('keydown', e => {
  const li = e.target.closest('.tnode');
  if (!li || e.target !== li) return;
  const items = [...$('tree').querySelectorAll('.tnode')], i = items.indexOf(li), key = li.dataset.key;
  const folder = key[0] === 'f' ? +key.slice(1) : null, plain = !e.ctrlKey && !e.metaKey && !e.altKey;
  const go = to => { if (!to) return; treeKey = to.dataset.key; items.forEach(x => x.tabIndex = x === to ? 0 : -1); to.focus(); };
  let done = true;
  if (e.key === 'ArrowDown') go(items[i + 1]);
  else if (e.key === 'ArrowUp') go(items[i - 1]);
  else if (e.key === 'Home') go(items[0]);
  else if (e.key === 'End') go(items[items.length - 1]);
  else if (e.key === 'ArrowRight') {
    if (li.getAttribute('aria-expanded') === 'false' && folder != null) toggleTree(folder, true);
    else if (li.getAttribute('aria-expanded') === 'true') go(li.querySelector('.tnode'));
  } else if (e.key === 'ArrowLeft') {
    if (li.getAttribute('aria-expanded') === 'true' && folder != null) toggleTree(folder, false);
    else go(li.parentElement.closest('.tnode'));
  } else if (e.key === 'Enter' || e.key === ' ') openPlace(treePlace(key));
  else if (e.key === 'F2' && folder != null) renameFolder(folder);
  else if (e.key === 'Delete' && folder != null) deleteFolder(folder);
  else if ((e.key === 'a' || e.key === 'A') && plain && !e.shiftKey) askPlace(key);
  else if (e.key === 'N' && e.shiftKey && plain) newFolder(folder);
  else if ((e.key === 'F10' && e.shiftKey) || e.key === 'ContextMenu') placeMenu(key, li.querySelector('.more'));
  else done = typeAhead(e, items, x => x.querySelector('.tname').textContent, li);
  if (done) e.preventDefault();
});
$('newfolder').onclick = () => newFolder(null);

// ---- the documents view ----
function listing() {
  const p = view.place, q = view.q.trim(), words = norm(q).split(/\s+/).filter(Boolean);
  const L = { folders: [], docs: [], matchFolders: [], hits: [], searching: !!q, words, showPath: false };
  if (!lib.loaded) return L;
  let docs;
  if (L.searching) {
    const within = searchWithin();
    const fields = d => norm([d.title, d.filename, d.doc_type, d.keywords, d.summary, d.vendor,
      d.folder_id != null ? lib.folders.get(d.folder_id)?.name : ''].join(' '));
    const client = [...lib.docs.values()].filter(d => (!within || within.has(d.id)) && words.every(w => fields(d).includes(w)));
    const s = view.server.key === searchKey() ? view.server : { docs: [], hits: [] };
    const ranked = [...new Set([...s.docs.filter(id => lib.docs.has(id) && (!within || within.has(id))), ...client.map(d => d.id)])];
    docs = ranked.map(id => lib.docs.get(id));
    const fwithin = view.qScope === 'here' && p.type === 'folder' ? subtree(p.id) : null;
    L.matchFolders = [...lib.folders.values()].filter(f => (!fwithin || (fwithin.has(f.id) && f.id !== p.id))
      && words.every(w => norm(f.name).includes(w))).slice(0, 5);
    L.hits = s.hits.filter(h => lib.docs.has(h.doc_id));
    L.showPath = true;
  } else if (p.type === 'all') { docs = [...lib.docs.values()].sort((a, b) => b.id - a.id); L.showPath = true; }
  else if (p.type === 'root') { L.folders = lib.kids.get('root').map(id => lib.folders.get(id)); docs = unfiledIds().map(id => lib.docs.get(id)); }
  else if (p.type === 'unfiled') docs = unfiledIds().map(id => lib.docs.get(id));
  else if (lib.folders.has(p.id)) {
    const flat = view.flat && lib.kids.get(p.id).length > 0;
    L.folders = flat ? [] : lib.kids.get(p.id).map(id => lib.folders.get(id));
    docs = (flat ? subtreeDocs(p.id) : lib.docsIn.get(p.id)).map(id => lib.docs.get(id));
    L.showPath = flat;
  } else docs = [];
  L.unfiltered = docs.length;
  docs = docs.filter(d => passes(d));
  if (!(L.searching && view.relevance)) sortDocs(docs);
  if (view.sort === 'name:desc') L.folders.reverse();
  L.docs = docs;
  view.order = docs.map(d => d.id);
  const shown = new Set(view.order);
  for (const id of [...sel.ids]) if (!shown.has(id)) sel.ids.delete(id);  // never act on what can't be seen
  return L;
}
// The documents a search is limited to: the place's (a folder with its subfolders, or Unfiled)
// with "This folder", else everything (null).
function searchWithin() {
  const p = view.place;
  if (view.qScope !== 'here' || p.type === 'all' || p.type === 'root') return null;
  return new Set(p.type === 'unfiled' ? unfiledIds() : lib.folders.has(p.id) ? subtreeDocs(p.id) : []);
}
const searchKey = () => {
  const p = view.place, here = view.qScope === 'here' && (p.type === 'folder' || p.type === 'unfiled');
  return `${view.q.trim()}|${here ? placeKey(p) : 'all'}`;
};
function sortDocs(docs) {
  const [k, dir] = view.sort.split(':'), sign = dir === 'desc' ? -1 : 1;
  const val = {
    name: d => docTitle(d), added: d => d.id, date: d => d.document_date, type: d => d.doc_type || null,
    size: d => d.file_size, amount: d => d.total_amount,
  }[k] || (d => d.id);
  docs.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x == null || x === '') { if (y == null || y === '') return collator.compare(docTitle(a), docTitle(b)); return 1; }  // empty values last, both ways
    if (y == null || y === '') return -1;
    const c = typeof x === 'number' ? x - y : collator.compare(String(x), String(y));
    return c ? c * sign : collator.compare(docTitle(a), docTitle(b));
  });
}
const filtersOn = () => { const f = view.filters; return f.types.size + (f.from || f.to ? 1 : 0) + f.amount + f.due + (f.suggested ? 1 : 0) + f.models.size + f.partial; };
// Read by and analysis: so the documents a local model read, or analyzed only in part, can be
// selected together and re-read with another model.
const readBy = d => d.model || '';
const notFull = d => d.analysis !== 'full';  // partial, none (failed), or '' from before analysis was recorded
function passes(d, f = view.filters) {
  if (f.types.size && !f.types.has(norm(d.doc_type || ''))) return false;
  if (f.models.size && !f.models.has(readBy(d))) return false;
  if (f.partial && !notFull(d)) return false;
  if ((f.from || f.to) && (!d.document_date || (f.from && d.document_date < f.from) || (f.to && d.document_date > f.to))) return false;
  if (f.amount && d.total_amount == null) return false;
  if (f.due && !d.due_date) return false;
  if (f.suggested && !(isUnfiled(d) && d.suggested_folder && (f.suggested === true || norm(d.suggested_folder) === norm(f.suggested)))) return false;
  return true;
}
// One sum per currency, never converted: "£182.40 + €12.00" (three currencies, then "+n"). Amounts
// read without a currency count as the others' when those all have one and the same (a model left
// it out of one of several dollar bills): `assumed` names it, for the tooltip.
function totals(docs) {
  const by = new Map();
  let n = 0, bare = 0;
  for (const d of docs) if (d.total_amount != null) { n++; bare += !d.currency; by.set(d.currency || '', (by.get(d.currency || '') || 0) + d.total_amount); }
  if (!n) return null;
  let assumed = null;
  if (bare && by.size === 2) {
    assumed = [...by.keys()].find(Boolean);
    by.set(assumed, by.get(assumed) + by.get(''));
    by.delete('');
  }
  const parts = [...by].sort((a, b) => b[1] - a[1]).map(([c, v]) => fmtAmount(v, c));
  return { n, bare, assumed, text: parts.slice(0, 3).join(' + ') + (parts.length > 3 ? ` +${parts.length - 3}` : '') };
}
const totalsTitle = t => `Sum of the ${plural(t.n, 'document')} with an amount, as read from them. Different currencies are not converted.`
  + (t.assumed ? ` ${t.bare === 1 ? 'One amount was' : `${t.bare} amounts were`} read without a currency and ${t.bare === 1 ? 'is' : 'are'} counted as ${t.assumed}, like the others.` : '');
function countText() {
  const p = view.place;
  if (!lib.loaded) return '';
  if (p.type === 'all') return plural(lib.docs.size, 'document');
  if (p.type === 'unfiled') return `Unfiled: ${plural(unfiledIds().length, 'document')}`;
  if (p.type === 'root') {
    const n = lib.kids.get('root').length;
    return n ? `${plural(lib.docs.size, 'document')} in ${plural(n, 'folder')} and Unfiled` : plural(lib.docs.size, 'document');
  }
  const f = lib.folders.get(p.id);
  if (!f) return '';
  const k = lib.kids.get(p.id).length;
  if (view.flat && k) return `${plural(f.total, 'document')}, subfolders included`;  // as the listing shows them
  return [plural(f.count, 'document'), k ? plural(k, 'folder') : ''].filter(Boolean).join(' · ');
}

function renderDocs() {
  const L = listing();
  renderHead(L);
  renderTools(L);
  renderChips();
  renderTip();
  renderSuggest();
  renderList(L);
  syncSelection();
}

function renderHead(L) {
  const p = view.place, S = isS.matches;
  // Breadcrumbs: Library and the folders above this one (none at the top and in All documents).
  const ol = $('crumbs').querySelector('ol');
  ol.replaceChildren();
  $('crumbs').hidden = p.type === 'root' || p.type === 'all';
  if (!$('crumbs').hidden) {
    const chain = [];
    if (p.type === 'folder') for (let f = lib.folders.get(lib.folders.get(p.id)?.parent_id), n = 0; f && n < 64; f = lib.folders.get(f.parent_id), n++) chain.unshift(f);
    const crumb = (label, place, drop) => {
      const li = el('li'), b = btn(label, () => openPlace(place));
      b.dataset.drop = drop;
      b.title = label;
      li.append(b);
      return li;
    };
    const items = [crumb('Library', { type: 'root' }, 'root')];
    const keep = S ? 1 : 5;
    let shown = chain;
    if (chain.length > keep + (S ? 1 : 0)) {  // the middle folds into "…"
      const hidden = chain.slice(S ? 0 : 1, chain.length - (S ? 1 : 2));
      shown = S ? chain.slice(-1) : [chain[0], ...chain.slice(-2)];
      const li = el('li'), more = btn('…', () => openMenu({ anchor: more, title: 'Folders above', items: hidden.map((f, i) => ({ label: f.name, icon: 'folder', indent: i, onclick: () => openPlace({ type: 'folder', id: f.id }) })) }), { aria: 'More folders above', title: hidden.map(f => f.name).join(' › ') });
      li.append(more);
      const at = S ? 0 : 1;
      shown.forEach((f, i) => { if (i === at) items.push(li); items.push(crumb(f.name, { type: 'folder', id: f.id }, 'f' + f.id)); });
    } else shown.forEach(f => items.push(crumb(f.name, { type: 'folder', id: f.id }, 'f' + f.id)));
    ol.append(...items);
  }
  const up = parentPlace(p);
  $('dback').hidden = p.type === 'root';
  $('dback').setAttribute('aria-label', `Back to ${placeName(up)}`);
  $('dback').title = `Back to ${placeName(up)}`;
  $('dtitle').textContent = placeName(p);
  const ask = $('daskfolder'), long = ask.querySelector('.long'), short = ask.querySelector('.short');
  ask.disabled = false;
  if (p.type === 'folder') { long.textContent = 'Ask about this folder'; short.textContent = 'Ask'; ask.title = 'Questions search this folder and the folders in it, including documents added later'; }
  else if (p.type === 'unfiled') {
    const n = unfiledIds().length;
    long.textContent = `Ask about these ${n}`; short.textContent = 'Ask';
    ask.title = 'Questions search these documents only; documents added to Unfiled later aren\'t included';
    ask.disabled = !n;
  } else { long.textContent = 'Ask all documents'; short.textContent = 'Ask all'; ask.title = 'Questions search every document'; }
  if (p.type === 'folder' && lib.loaded && !folderTotal(p.id)) {  // the server would refuse: nothing to search
    ask.disabled = true;
    ask.title = `${placeName(p)} has no documents yet`;
  }
  ask.setAttribute('aria-label', long.textContent);  // a narrow phone shows only the icon
  $('dcounttext').textContent = L.searching || filtersOn() ? `${plural(L.docs.length, 'document')} shown` + (filtersOn() && !L.searching ? ` of ${L.unfiltered}` : '') : countText();
  const t = totals(L.docs);
  // Library lists only the Unfiled documents, so its total is theirs. No-break spaces keep the dot
  // with the count, and the total in one piece, when the line wraps.
  const nb = String.fromCharCode(0xa0);
  $('dtotal').textContent = t ? `${nb}· ${p.type === 'root' && !L.searching ? `Unfiled${nb}total` : 'Total'}${nb}${t.text}` : '';
  $('dtotal').title = t ? totalsTitle(t) : '';
  const sub = p.type === 'folder' && lib.kids.get(p.id)?.length > 0;
  $('dflatbox').hidden = !sub;
  $('dflat').checked = view.flat;
}
$('dback').onclick = () => openPlace(parentPlace(view.place), { focus: true });
$('daskfolder').onclick = () => askPlace(placeKey(view.place));
$('dmenu').onclick = () => {
  const p = view.place;
  if (p.type === 'folder') folderMenu(p.id, $('dmenu'), null, true);
  else openMenu({ anchor: $('dmenu'), title: placeName(p), items: [
    { label: 'New folder', icon: 'plus', onclick: () => newFolder(null) },
    { label: 'Select documents', icon: 'check', onclick: () => { view.selecting = true; syncSelection(); } },
  ] });
};
$('dflat').onchange = e => { view.flat = e.target.checked; renderDocs(); };

function renderTools(L) {
  const p = view.place, name = placeName(p), scoped = p.type === 'folder' || p.type === 'unfiled';
  $('dq').placeholder = scoped && view.qScope === 'all' ? 'Search everywhere…' : p.type === 'folder' ? `Search ${name}…`
    : p.type === 'unfiled' ? 'Search Unfiled…' : p.type === 'all' ? 'Search all documents…' : 'Search the library…';
  $('dscope').hidden = !scoped;
  const [here, all] = $('dscope').querySelectorAll('button');
  here.textContent = p.type === 'unfiled' ? 'Unfiled' : 'This folder';
  here.title = p.type === 'unfiled' ? 'Search Unfiled only' : 'Search this folder and the folders in it';
  here.setAttribute('aria-pressed', String(view.qScope === 'here'));
  all.setAttribute('aria-pressed', String(view.qScope === 'all'));
  const t = $('dscopet');
  t.hidden = !scoped;
  t.textContent = view.qScope === 'here' ? `In ${name}` : 'Everywhere';
  t.setAttribute('aria-label', view.qScope === 'here' ? `Searching in ${name}. Switch to searching everywhere` : `Searching everywhere. Switch to searching in ${name}`);
  $('dfiltern').textContent = filtersOn() || '';
  $('dfilter').setAttribute('aria-label', filtersOn() ? `Filter, ${filtersOn()} active` : 'Filter');
  const [k, dir] = view.sort.split(':'), rel = L.searching && view.relevance;
  const arrow = dir === 'desc' ? '↓' : '↑';
  $('dsortlong').textContent = rel ? 'Sort: Relevance' : `Sort: ${SORTS[k][0]} ${arrow}`;
  $('dsortshort').textContent = rel ? 'Relevance' : `${SHORT_SORT[k]} ${arrow}`;
  $('dsort').setAttribute('aria-label', rel ? 'Sorted by relevance. Change the sort' : `Sorted by ${SORTS[k][0].toLowerCase()}, ${SORTS[k][dir === 'desc' ? 2 : 1].toLowerCase()}. Change the sort`);
  for (const b of $('dlayout').querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.v === view.layout));
  // A place with nothing in it (an empty folder, say): no Filter, Sort, layout or Select to try.
  $('dtools').classList.toggle('bare', lib.loaded && !L.unfiltered && !L.folders.length && !L.searching && !filtersOn());
}

// Search: an instant match here on every keystroke, then the server's (titles and text) after a pause.
let searchTimer = null;
$('dq').oninput = () => {
  const was = view.q;
  view.q = $('dq').value;
  if (!was.trim() && view.q.trim()) view.relevance = true;
  renderDocs();
  clearTimeout(searchTimer);
  if (view.q.trim()) searchTimer = setTimeout(serverSearch, 250);
};
$('dq').onkeydown = e => {
  if (e.key !== 'Escape') return;
  e.preventDefault();
  e.stopPropagation();
  if ($('dq').value) clearSearch(); else focusList();
};
function clearSearch() {
  $('dq').value = '';
  view.q = '';
  view.relevance = true;
  renderDocs();
}
async function serverSearch() {
  const k = searchKey(), q = view.q.trim(), p = view.place;
  if (!q) return;
  const where = view.qScope === 'here' && p.type === 'folder' ? `&folder=${p.id}&recursive=1`
    : view.qScope === 'here' && p.type === 'unfiled' ? '&folder=unfiled' : '';
  let docs, hits;
  try {
    [docs, hits] = await Promise.all([api(`/api/documents?q=${encodeURIComponent(q)}&limit=200${where}`),
                                      api(`/api/search?q=${encodeURIComponent(q)}&limit=10${where}`)]);
  } catch (err) { if (err.status === 404) lib.load(); return; }
  if (k !== searchKey()) return;  // typed on meanwhile
  view.server = { key: k, docs: docs.map(d => d.id), hits };
  renderDocs();
  announce(`${plural(view.order.length, 'document')} found`);
}
for (const b of $('dscope').querySelectorAll('button')) b.onclick = () => setSearchScope(b.dataset.v);
$('dscopet').onclick = () => setSearchScope(view.qScope === 'here' ? 'all' : 'here');
function setSearchScope(v) {
  if (view.qScope === v) return;
  view.qScope = v;
  sel.ids.clear();
  renderDocs();
  if (view.q.trim()) serverSearch();
}
for (const b of $('dlayout').querySelectorAll('button')) b.onclick = () => {
  view.layout = isS.matches ? (view.layout === 'grid' ? 'list' : 'grid') : b.dataset.v;
  store.set('docsView', view.layout);
  renderDocs();
  focusList();
};
$('dsort').onclick = () => {
  const [k, dir] = view.sort.split(':'), rel = !!view.q.trim() && view.relevance;
  const items = [];
  if (view.q.trim()) items.push({ label: 'Relevance', radio: true, checked: rel, onclick: () => { view.relevance = true; renderDocs(); announce('Sorted by relevance'); } });
  for (const [key, [label]] of Object.entries(SORTS)) {
    items.push({ label, radio: true, checked: !rel && key === k, onclick: () => setSort(key, key === k ? dir : key === 'name' || key === 'type' ? 'asc' : 'desc') });
  }
  if (!rel) {
    items.push('-');
    items.push({ label: SORTS[k][2], radio: true, checked: dir === 'desc', onclick: () => setSort(k, 'desc') });
    items.push({ label: SORTS[k][1], radio: true, checked: dir === 'asc', onclick: () => setSort(k, 'asc') });
  }
  if (k === 'amount' && !rel) items.push({ note: 'Amounts in different currencies are compared as plain numbers.' });
  openMenu({ anchor: $('dsort'), title: 'Sort documents', items });
};
function setSort(k, dir) {
  view.sort = `${k}:${dir}`;
  view.relevance = false;
  store.set('docsSort', view.sort);
  renderDocs();
  announce(`Sorted by ${SORTS[k][0].toLowerCase()}, ${SORTS[k][dir === 'desc' ? 2 : 1].toLowerCase()}`);
}
$('dfilter').onclick = openFilter;
$('dnewfolder').onclick = () => newFolder(view.place.type === 'folder' ? view.place.id : null);
$('dselect').onclick = () => { view.selecting = true; syncSelection(); };

function renderChips() {
  const f = view.filters, box = $('dchips'), chips = [];
  const chip = (label, off) => chips.push(btn(`${label} ✕`, () => { off(); renderDocs(); }, { cls: 'chip', title: 'Remove this filter' }));
  for (const t of f.types) chip(`type: ${t}`, () => f.types.delete(t));
  if (f.from || f.to) chip(`date: ${dateRangeText(f.from, f.to)}`, () => { f.from = f.to = ''; });
  if (f.amount) chip('with an amount', () => { f.amount = false; });
  if (f.due) chip('has a due date', () => { f.due = false; });
  if (f.suggested) chip(f.suggested === true ? 'has a folder suggestion' : `suggested: ${f.suggested.split('/').join(' › ')}`, () => { f.suggested = null; });
  for (const m of f.models) chip(`read by: ${m || 'not recorded'}`, () => f.models.delete(m));
  if (f.partial) chip('not fully analyzed', () => { f.partial = false; });
  if (chips.length) chips.push(btn('Clear filters', () => { clearFilters(); renderDocs(); }, { cls: 'chip clear' }));
  box.replaceChildren(...chips);
}
function dateRangeText(from, to) {
  const y = s => s.slice(0, 4);
  if (from && to && from.endsWith('-01-01') && to.endsWith('-12-31') && y(from) === y(to)) return y(from);
  return [from ? fmtDate(from) : '…', to ? fmtDate(to) : '…'].join(' to ');
}
function clearFilters() { const f = view.filters; f.types.clear(); f.from = f.to = ''; f.amount = f.due = false; f.suggested = null; f.models.clear(); f.partial = false; }
const noFilters = () => ({ types: new Set(), from: '', to: '', amount: false, due: false, suggested: null, models: new Set(), partial: false });

function renderTip() {
  const p = view.place, box = $('dtip');
  const show = lib.loaded && lib.docs.size > 0 && !lib.folders.size && (p.type === 'root' || p.type === 'unfiled') && !store.get('docsTip');
  box.hidden = !show;
  if (!show) return;
  const card = el('div', 'finding');
  card.append(el('p', '', 'Put related documents in folders, such as Tractor, Receipts and Bills. Make one with New folder, '
    + 'then select documents and choose Move to…, or select several and choose New folder from these.'));
  const row = el('div', 'row');
  row.append(btn('New folder', () => newFolder(null), { icon: 'plus' }),
    btn('Got it', () => { store.set('docsTip', '1'); renderTip(); }));
  card.append(row);
  box.replaceChildren(card);
}

// Unfiled documents with a suggested folder, grouped by it: one click files them all.
let suggestAll = false;
function renderSuggest() {
  const p = view.place, box = $('dsuggest');
  const groups = new Map();
  if (lib.loaded && (p.type === 'root' || p.type === 'unfiled') && !view.q.trim()) {
    for (const id of unfiledIds()) {
      const d = lib.docs.get(id);
      if (!d.suggested_folder) continue;
      const k = norm(d.suggested_folder);
      if (!groups.has(k)) groups.set(k, { path: d.suggested_folder, ids: [] });
      groups.get(k).ids.push(id);
    }
  }
  box.hidden = !groups.size;
  if (!groups.size) { box.replaceChildren(); return; }
  const list = [...groups.values()].sort((a, b) => b.ids.length - a.ids.length);
  const rows = (suggestAll ? list : list.slice(0, 3)).map(g => {
    const fid = findPath(g.path), where = g.path.split('/').join(' › '), n = g.ids.length;
    const row = el('div', 'finding sugrow'), what = el('span', 'what');
    what.append(el('span', 'label', 'Suggested'), el('b', '', where), fid == null ? ' · new folder' : '', ` · ${plural(n, 'document')}`);
    row.append(what,
      btn(fid == null ? `Create and move ${n}` : `Move ${n}`, () => fid == null ? moveDocs(g.ids, null, { path: g.path }) : moveDocs(g.ids, fid),
        { cls: 'primary', title: fid == null ? `Create ${where} and move ${plural(n, 'document')} into it` : `Move ${plural(n, 'document')} to ${where}` }),
      btn('Show', () => { view.filters.suggested = g.path; renderDocs(); }, { title: 'Show only these documents' }));
    return row;
  });
  if (list.length > 3 && !suggestAll) rows.push(btn(`+ ${list.length - 3} more`, () => { suggestAll = true; renderSuggest(); }, { cls: 'open' }));
  box.replaceChildren(...rows);
}

// ---- the listing ----
const thumbFailed = new Set();  // documents whose thumbnail didn't load, this session
function typedIcon(d, cls = '') {
  const f = el('div', 'ficon' + (cls ? ' ' + cls : ''));
  f.append(el('b', '', kindLabel(d)));
  if (cls !== 'mini' && d.doc_type) f.append(el('i', '', d.doc_type));
  return f;
}
function thumbOrIcon(d, size, iconCls = '') {
  if (!d.thumb || thumbFailed.has(d.id + ':' + d.sha256)) return typedIcon(d, iconCls);
  const img = el('img', d.kind === 'pdf' ? 'pdf' : '');
  img.alt = '';
  img.loading = 'lazy';
  img.decoding = 'async';
  img.src = `${d.thumb}&size=${size}`;
  img.onerror = () => { thumbFailed.add(d.id + ':' + d.sha256); img.replaceWith(typedIcon(d, iconCls)); };
  return img;
}
// "Tesco receipt, receipt, 3 November 2026, £42.10, JPG, in Receipts"
function docLabel(d) {
  const parts = [docTitle(d), d.doc_type, d.document_date && fmtDate(d.document_date, { month: 'long' }),
    d.total_amount != null && fmtAmount(d.total_amount, d.currency), kindLabel(d), `in ${pathText(isUnfiled(d) ? null : d.folder_id)}`];
  return parts.filter(Boolean).join(', ');
}
function suggestionChip(d) {
  if (!d.suggested_folder || !isUnfiled(d)) return null;
  const p = view.place;
  if (!(p.type === 'root' || p.type === 'unfiled' || p.type === 'all' || view.q.trim())) return null;
  const where = d.suggested_folder.split('/').join(' › '), exists = findPath(d.suggested_folder) != null;
  const b = btn(exists ? `→ ${where}` : `→ new: ${where}`, e => { e.stopPropagation(); applySuggestion(d.id, d.suggested_folder); },
    { cls: 'sug' + (exists ? '' : ' new'), title: exists ? `Move it to ${where}` : `Create ${where} and move it there` });
  b.tabIndex = -1;
  return b;
}
function moreButton(label) {
  const b = btn('', null, { cls: 'more', aria: `More for ${label}`, title: 'More actions', icon: 'more' });
  b.tabIndex = -1;
  return b;
}

function group(label, ...content) {
  const g = el('section', 'dgroup');
  const lab = el('div', 'glabel');
  if (typeof label === 'string') lab.append(el('span', 'label', label)); else lab.append(label);
  g.append(lab, ...content);
  return g;
}

function folderTile(f, match = false) {
  const t = el('div', 'tile' + (match ? ' match' : ''));
  t.setAttribute('role', 'listitem');
  t.tabIndex = -1;
  t.title = f.names.join(' › ');
  t.dataset.nav = 'f' + f.id;
  t.dataset.drop = 'f' + f.id;
  t.setAttribute('aria-label', `Folder ${f.name}, ${plural(f.total, 'document')}${match ? `, in ${f.names.slice(0, -1).join(' › ') || 'Library'}` : ''}`);
  if (fine.matches) t.draggable = true;
  t.append(icon('folder'));
  if (match) {
    const tt = el('span', 'tt');
    tt.append(el('span', 'tname', f.name), el('span', 'tpath', f.names.slice(0, -1).join(' › ') || 'Library'));
    t.append(tt);
  } else t.append(el('span', 'tname', f.name));
  t.append(el('span', 'tcount', String(f.total)), icon('chev', 'chev'), moreButton(f.name));
  return t;
}
function allTile() {  // a phone's Library starts with All documents
  const t = el('div', 'tile allrow');
  t.setAttribute('role', 'listitem');
  t.tabIndex = -1;
  t.dataset.nav = 'pall';
  t.append(icon('stack'), el('span', 'tname', 'All documents'), el('span', 'tcount', String(lib.docs.size)), icon('chev', 'chev'));
  return t;
}

function docCard(d, L) {
  const c = el('div', 'dcard');
  c.setAttribute('role', 'option');
  c.tabIndex = -1;
  c.dataset.nav = 'd' + d.id;
  c.setAttribute('aria-label', docLabel(d, L.showPath));
  c.setAttribute('aria-selected', String(sel.ids.has(d.id)));
  if (view.doc === d.id) c.setAttribute('aria-current', 'true');
  if (fine.matches) c.draggable = true;
  if (freshDocs.has(d.id)) c.classList.add('fresh');
  const th = el('div', 'thumb'), cb = el('span', 'cbox'), box = el('input');
  box.type = 'checkbox';
  box.tabIndex = -1;
  box.checked = sel.ids.has(d.id);
  box.setAttribute('aria-label', `Select ${docTitle(d)}`);
  cb.append(box);
  th.append(thumbOrIcon(d, 'card'), cb, el('span', 'kindb', kindLabel(d)), moreButton(docTitle(d)));
  const body = el('div', 'cbody'), h = el('h3', 'ctitle');
  h.append(marked(docTitle(d), L.words));
  h.title = docTitle(d);
  body.append(h, el('div', 'cmeta', [d.doc_type, d.document_date ? fmtDate(d.document_date, { year: false }) : `Added ${fmtAdded(d.uploaded_at, false, false)}`].filter(Boolean).join(' · ')));
  const due = (isBill(d) || d.due_date) && dueInfo(d.due_date);
  if (d.total_amount != null || d.vendor || due) {
    const a = el('div', 'camt');
    if (d.total_amount != null) a.append(el('span', 'amt', fmtAmount(d.total_amount, d.currency)));
    if (d.vendor && !norm(docTitle(d)).startsWith(norm(d.vendor))) a.append(el('span', 'vend', d.vendor));  // not twice
    if (due) a.append(el('span', 'due' + (due.soon ? ' soon' : ''), due.text));
    body.append(a);
  }
  const badges = el('div', 'cbadges');
  if (d.unit_fixes) badges.append(unitBadge(d.id, d.unit_fixes, false));
  const chip = suggestionChip(d);
  if (chip) badges.append(chip);
  if (String(d.analysis || '').startsWith('none')) { const na = el('span', 'na', 'Not analyzed'); na.title = d.analysis; badges.append(na); }
  body.append(badges);
  if (L.showPath) {
    const fid = isUnfiled(d) ? null : d.folder_id, cp = el('div', 'cpath', `in ${shortPath(fid)}`);
    cp.title = `in ${pathText(fid)}`;
    body.append(cp);
  }
  c.append(th, body);
  return c;
}

function ghostCard(row) {
  const v = row.view || {}, c = el('div', 'dcard ghost');
  c.dataset.nav = 'g' + ghostId(row);
  c.tabIndex = -1;
  c.setAttribute('role', 'option');
  c.setAttribute('aria-disabled', 'true');
  c.setAttribute('aria-label', `${v.name || 'Upload'}: ${v.stage || ''}`);
  const th = el('div', 'thumb');
  if (!v.needs) th.append(el('span', 'spin'));
  if (v.bar != null) { const bar = el('div', 'bar' + (v.bar === 'wait' ? ' wait' : '')), i = el('i'); if (typeof v.bar === 'number') i.style.width = `${Math.round(v.bar * 100)}%`; bar.append(i); th.append(bar); }
  const body = el('div', 'cbody');
  body.append(el('h3', 'ctitle', v.name || ''), el('div', 'cmeta', v.stage || ''));
  if (v.needs) body.append(el('span', 'needs', 'Needs you'));
  c.append(th, body);
  return c;
}
let ghostSeq = 0;
const ghostId = row => row.gid || (row.gid = ++ghostSeq);
// Uploads on their way into the place shown: queued, being read, or waiting for you.
function ghostRows() {
  const p = view.place, out = [];
  if (view.q.trim()) return out;
  for (const row of $('queue').children) {
    const j = row.job;
    if (j && (j.kind !== 'upload' || j.status === 'done' || j.status === 'cancelled')) continue;
    if (!j && !row.view) continue;
    if (!j && row.classList.contains('ok')) continue;
    const fid = (j ? j.options?.folder_id : row.target?.folder_id) ?? null;
    if (p.type === 'all' || (p.type === 'folder' ? fid === p.id : fid == null)) out.push(row);
  }
  return out;
}

function renderList(L) {
  const box = $('dlist'), active = box.contains(document.activeElement) ? document.activeElement.closest('[data-nav]')?.dataset.nav : null;
  const p = view.place, S = isS.matches;
  if (active) view.focusKey = active;
  box.classList.toggle('anysel', sel.ids.size > 0);
  box.classList.toggle('selmode', sel.ids.size > 0 || view.selecting || S && coarse.matches);
  if (!lib.loaded) { box.replaceChildren(lib.error ? loadError() : skeleton()); return; }
  const ghosts = ghostRows();
  box.dataset.ghosts = ghosts.map(ghostId).join(',');
  const parts = [];
  if (!lib.docs.size && !lib.folders.size && !ghosts.length && !L.searching) { box.replaceChildren(emptyLibrary()); return; }
  if (L.searching) {
    if (L.matchFolders.length) parts.push(group('Matching folders', tilesBlock(L.matchFolders, true)));
    if (L.docs.length) parts.push(group(`Documents · ${L.docs.length}`, docsBlock(L.docs, [], L)));
    if (L.hits.length) parts.push(group('Matching passages', hitsBlock(L.hits)));
    if (!L.docs.length && !L.matchFolders.length && !L.hits.length) parts.push(noResults());
  } else {
    const folders = L.folders;
    if (view.layout === 'list') {
      if (folders.length || L.docs.length || ghosts.length) parts.push(docsBlock(L.docs, ghosts, L, folders));
    } else {
      if (folders.length || (S && p.type === 'root')) {
        const tiles = tilesBlock(folders);
        if (S && p.type === 'root') tiles.prepend(allTile());
        parts.push(group(folders.length ? 'Folders' : '', tiles));
      }
      // The count of what is listed under it (after filters), and the uploads on their way in.
      const coming = ghosts.length ? ` · ${ghosts.length} uploading` : '';
      const label = (p.type === 'root' ? (folders.length ? `Unfiled · ${L.docs.length}` : 'Documents')
        : p.type === 'all' ? `All documents · ${L.docs.length}` : `Documents · ${L.docs.length}`) + coming;
      const hideUnfiled = p.type === 'root' && folders.length && !L.unfiltered && !ghosts.length;
      if (!hideUnfiled && (L.docs.length || ghosts.length)) {
        const lab = document.createDocumentFragment();
        lab.append(el('span', 'label', label));
        const sug = p.type === 'root' ? L.docs.filter(d => d.suggested_folder).length : 0;
        if (sug && !view.filters.suggested) lab.append(btn(`Review ${plural(sug, 'suggestion')}`, () => { view.filters.suggested = true; renderDocs(); }));
        parts.push(group(lab, docsBlock(L.docs, ghosts, L)));
      }
    }
    if (!L.docs.length && !ghosts.length) {
      if (L.unfiltered && filtersOn()) parts.push(noMatches());
      else if (p.type === 'unfiled') parts.push(emptyState2('Everything is filed.', '', [], 'check'));
      else if (p.type === 'folder' && !folders.length) {
        const name = placeName(p);  // a long name is in the title just above: not again in a big heading
        parts.push(emptyState2(name.length > 24 ? 'This folder is empty.' : `${name} is empty.`,
          S ? 'Add files here, or move documents in: select them, then Move to…' : 'Add files here, drop them here, or move documents in: select them, then Move to…',
          [btn('Add files', () => $('file').click(), { cls: 'primary', icon: 'upload' }), btn('New folder inside', () => newFolder(p.id), { icon: 'plus' })]));
      } else if (p.type === 'all') parts.push(emptyLibrary());
    }
  }
  box.replaceChildren(...parts);
  // One tab stop: the item focused last, else the first.
  const items = navItems();
  const keep = items.find(x => x.dataset.nav === view.focusKey) || items[0];
  for (const x of items) x.tabIndex = x === keep ? 0 : -1;
  if (active && keep && keep.dataset.nav === active) keep.focus({ preventScroll: true });
  setTimeout(() => { for (const id of [...freshDocs]) if (box.querySelector(`[data-nav="d${id}"]`)) freshDocs.delete(id); }, 1600);
}
const navItems = () => [...$('dlist').querySelectorAll('[data-nav]')];
function focusList() {
  const items = navItems(), x = items.find(i => i.tabIndex === 0) || items[0];
  if (x) { x.focus({ preventScroll: false }); return true; }
  return false;
}
function tilesBlock(folders, match = false) {
  const t = el('div', 'tiles');
  t.setAttribute('role', 'list');
  t.setAttribute('aria-label', 'Folders');
  t.append(...folders.map(f => folderTile(f, match)));
  return t;
}
// Above 300 documents the cards come in batches of 150, as the end comes into view.
function batched(container, items, make) {
  const first = items.length > 300 ? 150 : items.length;
  container.append(...items.slice(0, first).map(make));
  if (first >= items.length) return;
  let at = first;
  const sentinel = el('div');
  sentinel.style.cssText = 'height:1px;grid-column:1/-1';
  container.append(sentinel);
  const io = new IntersectionObserver(es => {
    if (!es.some(e => e.isIntersecting)) return;
    sentinel.before(...items.slice(at, at + 150).map(make));
    at += 150;
    if (at >= items.length) { io.disconnect(); sentinel.remove(); }
  }, { root: $('dmain'), rootMargin: '600px' });
  io.observe(sentinel);
}
function docsBlock(docs, ghosts, L, folders = []) {
  if (view.layout === 'list') return docTable(docs, ghosts, L, folders);
  const g = el('div', 'dgrid');
  g.setAttribute('role', 'listbox');
  g.setAttribute('aria-multiselectable', 'true');
  g.setAttribute('aria-label', 'Documents');
  g.append(...ghosts.map(ghostCard));
  batched(g, docs, d => docCard(d, L));
  return g;
}

const COLS = [['name', 'Name', 'c-name'], ['folder', 'Folder', 'c-folder'], ['type', 'Type', 'c-type'], ['date', 'Document date', 'c-date'],
  ['amount', 'Amount', 'c-amt num'], ['added', 'Added', 'c-added'], ['size', 'Size', 'c-size num']];
function docTable(docs, ghosts, L, folders) {
  const hasAmt = docs.some(d => d.total_amount != null);
  const t = el('table', 'dtable' + (hasAmt ? ' hasamt' : '') + (L.showPath ? ' haspath' : ''));
  t.setAttribute('role', 'grid');
  t.setAttribute('aria-multiselectable', 'true');
  t.setAttribute('aria-label', 'Documents');
  const head = el('thead'), hr = el('tr'), all = el('input');
  all.type = 'checkbox';
  all.className = 'selall';
  all.setAttribute('aria-label', 'Select all documents shown');
  const th0 = el('th', 'csel');
  th0.append(all);
  hr.append(th0);
  const [sk, sdir] = view.sort.split(':'), rel = L.searching && view.relevance;
  for (const [k, label, cls] of COLS) {
    const th = el('th', cls);
    if (k === 'folder') th.textContent = label;
    else {
      const b = btn(label, () => setSort(k, k === sk && !rel ? (sdir === 'asc' ? 'desc' : 'asc') : k === 'name' || k === 'type' ? 'asc' : 'desc'));
      if (k === sk && !rel) { th.setAttribute('aria-sort', sdir === 'asc' ? 'ascending' : 'descending'); b.append(` ${sdir === 'asc' ? '▲' : '▼'}`); }
      th.append(b);
    }
    hr.append(th);
  }
  const thm = el('th', 'cmore');
  thm.append(el('span', 'sr', 'Actions'));
  hr.append(thm);
  head.append(hr);
  const body = el('tbody');
  for (const row of ghosts) body.append(ghostRow(row));
  for (const f of folders) body.append(folderRow(f));
  batched(body, docs, d => docRow(d, L));
  t.append(head, body);
  const tot = totals(docs);
  if (tot || docs.length) {
    const foot = el('tfoot'), tr = el('tr'), td = el('td');
    td.colSpan = 9;
    td.className = 'tfoot';
    tr.append(td);
    foot.append(tr);
    t.append(foot);
  }
  const wrap = el('div', 'tablewrap');
  wrap.append(t);
  return wrap;
}
function cell(cls, ...content) { const td = el('td', cls); td.append(...content); return td; }
function folderRow(f) {
  const tr = el('tr', 'frow');
  tr.dataset.nav = 'f' + f.id;
  tr.dataset.drop = 'f' + f.id;
  tr.tabIndex = -1;
  tr.setAttribute('aria-label', `Folder ${f.name}, ${plural(f.total, 'document')}`);
  if (fine.matches) tr.draggable = true;
  const tn = el('span', 'tn folder');
  tn.append(icon('folder'));
  const nt = el('span', 'ntext');
  nt.append(el('b', '', f.name), el('small', '', plural(f.total, 'document')));
  const nc = el('span', 'ncell');
  nc.append(tn, nt);
  const more = moreButton(f.name);
  tr.append(cell('csel'), cell('c-name', nc), cell('c-folder', '—'), cell('c-type', 'Folder'), cell('c-date', '—'), cell('c-amt num', '—'),
    cell('c-added nowrap', fmtAdded(f.created_at)), cell('c-size num', '—'), cell('cmore', more));
  return tr;
}
function docRow(d, L) {
  const tr = el('tr');
  tr.dataset.nav = 'd' + d.id;
  tr.tabIndex = -1;
  tr.setAttribute('aria-label', docLabel(d, L.showPath));
  tr.setAttribute('aria-selected', String(sel.ids.has(d.id)));
  if (view.doc === d.id) tr.setAttribute('aria-current', 'true');
  if (fine.matches) tr.draggable = true;
  if (freshDocs.has(d.id)) tr.classList.add('fresh');
  const box = el('input');
  box.type = 'checkbox';
  box.tabIndex = -1;
  box.checked = sel.ids.has(d.id);
  box.setAttribute('aria-label', `Select ${docTitle(d)}`);
  const tn = el('span', 'tn');
  tn.append(thumbOrIcon(d, 'card', 'mini'));
  const nt = el('span', 'ntext'), b = el('b');
  b.append(marked(docTitle(d), L.words));
  b.title = docTitle(d);
  const small = el('small');
  if (docTitle(d) !== d.filename) small.append(el('span', 'fn', d.filename));  // not the title again
  else if (d.vendor) small.append(el('span', 'fn', d.vendor));
  if (d.unit_fixes) small.append(unitBadge(d.id, d.unit_fixes, false));
  const chip = suggestionChip(d);
  if (chip) small.append(chip);
  if (String(d.analysis || '').startsWith('none')) { const na = el('span', 'na', 'Not analyzed'); na.title = d.analysis; small.append(na); }
  const meta = [d.doc_type, d.document_date && fmtDate(d.document_date, { year: false }), d.total_amount != null && fmtAmount(d.total_amount, d.currency)].filter(Boolean).join(' · ');
  if (meta) small.append(el('span', 'narrowmeta', meta));
  nt.append(b, small);
  const nc = el('span', 'ncell');
  nc.append(tn, nt);
  const fid = isUnfiled(d) ? null : d.folder_id;
  const path = btn(shortPath(fid), e => { e.stopPropagation(); openPlace(fid == null ? { type: 'unfiled' } : { type: 'folder', id: fid }); },
    { cls: 'pathbtn', title: pathText(fid) });
  path.tabIndex = -1;
  const type = el('span', 'ell', d.doc_type || '—');
  if (d.doc_type) type.title = d.doc_type;
  const date = cell('c-date nowrap', d.document_date ? fmtDate(d.document_date) : '—');
  const due = (isBill(d) || d.due_date) && dueInfo(d.due_date);
  if (due) { date.append(el('br'), el('span', 'due' + (due.soon ? ' soon' : ''), due.text)); }
  const added = cell('c-added nowrap', fmtAdded(d.uploaded_at));
  added.title = fmtAdded(d.uploaded_at, true);
  tr.append(cell('csel', box), cell('c-name', nc), cell('c-folder', path), cell('c-type', type), date,
    cell('c-amt num', d.total_amount != null ? fmtAmount(d.total_amount, d.currency) : '—'), added,
    cell('c-size num', d.has_file ? fmtSize(d.file_size) : '—'), cell('cmore', moreButton(docTitle(d))));
  return tr;
}
function ghostRow(row) {
  const v = row.view || {}, tr = el('tr', 'ghost');
  tr.dataset.nav = 'g' + ghostId(row);
  tr.tabIndex = -1;
  tr.setAttribute('aria-disabled', 'true');
  const tn = el('span', 'tn');
  tn.append(v.needs ? icon('upload') : el('span', 'spin'));
  const nt = el('span', 'ntext');
  nt.append(el('b', '', v.name || ''), el('small', '', v.stage || ''));
  if (v.needs) nt.append(el('span', 'needs', 'Needs you'));
  const nc = el('span', 'ncell');
  nc.append(tn, nt);
  const td = cell('c-name', nc);
  tr.append(cell('csel'), td, cell('c-folder'), cell('c-type'), cell('c-date'), cell('c-amt'), cell('c-added'), cell('c-size'), cell('cmore'));
  return tr;
}
function hitsBlock(hits) {
  const box = el('div');
  box.style.cssText = 'display:grid;gap:8px';
  for (const h of hits) {
    const d = lib.docs.get(h.doc_id), card = el('div', 'hit'), meta = el('div', 'meta');
    meta.append(el('b', '', docTitle(d)), ` · ${h.page_start ? pagesText(pageSpan(h.page_start, h.page_end)) : `part ${h.idx + 1}`}`);
    const snip = el('div', 'snip');
    h.snippet.split(/\[([^\]]*)\]/).forEach((part, i) => snip.append(i % 2 ? el('mark', '', part) : part));  // the API marks matches as [word]
    const acts = el('div', 'hacts');
    if (d.has_file && d.kind === 'pdf' && h.page_start) {
      acts.append(Object.assign(el('a', 'open', `Open page ${h.page_start} ↗`), { href: `/api/documents/${d.id}/file#page=${h.page_start}`, target: '_blank', rel: 'noopener' }));
    }
    acts.append(btn('Show document', () => openDoc(d.id), { cls: 'open' }));
    card.append(meta, snip, acts);
    box.append(card);
  }
  return box;
}
function noResults() {
  const p = view.place, box = el('div', 'dnone'), here = view.qScope === 'here' && (p.type === 'folder' || p.type === 'unfiled');
  box.append(el('p', '', `Nothing ${here ? `in ${placeName(p)} ` : ''}matches “${view.q.trim()}”.`));
  const row = el('div', 'row');
  if (here) row.append(btn('Search everywhere', () => setSearchScope('all')));
  row.append(btn('Clear search', () => { clearSearch(); $('dq').focus(); }));
  if (filtersOn()) row.append(btn('Clear filters', () => { clearFilters(); renderDocs(); }));
  box.append(row);
  return box;
}
function noMatches() {
  const box = el('div', 'dnone'), row = el('div', 'row');
  box.append(el('p', '', 'No documents match these filters.'));
  row.append(btn('Clear filters', () => { clearFilters(); renderDocs(); }));
  box.append(row);
  return box;
}
function emptyState2(title, text, buttons, ic = '') {
  const box = el('div', 'dempty');
  if (ic) box.append(icon(ic, 'ok'));
  box.append(el('h3', '', title));
  if (text) box.append(el('p', 'muted', text));
  if (buttons.length) { const row = el('div', 'row'); row.append(...buttons); box.append(row); }
  return box;
}
function emptyLibrary() {
  const box = emptyState2('Your library is empty.', 'Add PDFs, Word files, photos of receipts and bills, Markdown and text.',
    [btn('Add files', () => $('file').click(), { cls: 'primary', icon: 'upload' }), btn('New folder', () => newFolder(null), { icon: 'plus' })]);
  if (!isS.matches) box.append(el('p', 'muted', 'or drop files anywhere here'));
  return box;
}
function skeleton() {
  const g = el('div', 'dgrid');
  for (let i = 0; i < 6; i++) { const s = el('div', 'skel'); s.append(el('div', 'thumb'), el('div', 'line'), el('div', 'line s')); g.append(s); }
  return g;
}
function loadError() {
  return emptyState2(`Couldn't load your documents: ${lib.error}`, '', [btn('Try again', () => { lib.error = null; render(); lib.load(); }, { cls: 'primary' })]);
}

// ---- selection (documents only) ----
let delArmed = false, delTimer = null;
const rereadableOf = ids => ids.filter(id => lib.docs.get(id)?.has_file);
function syncSelection() {
  if (view.mode !== 'docs') return;
  const n = sel.ids.size, on = n > 0 || view.selecting, S = isS.matches;
  for (const x of $('dlist').querySelectorAll('[data-nav^="d"]')) {
    const id = +x.dataset.nav.slice(1), s = sel.ids.has(id);
    x.setAttribute('aria-selected', String(s));
    const box = x.querySelector('input[type=checkbox]');
    if (box) box.checked = s;
  }
  $('dlist').classList.toggle('anysel', n > 0);
  $('dlist').classList.toggle('selmode', on || (S && coarse.matches));
  const all = $('dlist').querySelector('.selall');
  if (all) {
    all.checked = view.order.length > 0 && view.order.every(id => sel.ids.has(id));
    all.indeterminate = !all.checked && n > 0;
  }
  const foot = $('dlist').querySelector('td.tfoot');
  if (foot) {
    const docs = (n ? [...sel.ids] : view.order).map(id => lib.docs.get(id)).filter(Boolean), t = totals(docs);
    foot.textContent = n ? `Selected: ${n}${t ? ` · ${t.text}` : ''}` : `${plural(docs.length, 'document')}${t ? ` · Total ${t.text}` : ''}`;
    foot.title = t ? totalsTitle(t) : '';
  }
  $('dtools').hidden = on;
  $('dselbar').hidden = !on;
  $('dselcount').textContent = `${n} selected`;
  const every = view.order.length && view.order.every(id => sel.ids.has(id));
  $('dselall').hidden = !!every || !view.order.length;
  $('dselall').textContent = S ? 'Select all' : `Select all ${view.order.length}`;
  if (!n) disarmDelete(false);
  const acts = bulkButtons(n), a = document.activeElement;
  const act = a?.closest?.('#dselacts, #dbulk') ? a.dataset.act : null;  // keeps its focus through the rebuild
  $('dselacts').replaceChildren(...acts(false));
  $('dbulk').replaceChildren(...acts(true));
  if (act) (S ? $('dbulk') : $('dselacts')).querySelector(`[data-act="${act}"]`)?.focus();
  $('dbulk').hidden = !(on && S && view.doc == null);
  document.body.classList.toggle('hasbar', on && S && view.doc == null);
}
function bulkButtons(n) {
  const ids = [...sel.ids], can = rereadableOf(ids).length;
  return phone => {
    const b = (label, short, ic, onclick, opts = {}) => {
      const x = btn(phone ? short : label, onclick, { icon: ic, title: opts.title || label, cls: opts.cls || '' });
      x.dataset.act = ic;
      if (opts.disabled) x.disabled = true;
      return x;
    };
    const tip = n && can < n
      ? `${n - can} of the selected documents ${n - can > 1 ? 'were' : 'was'} uploaded before originals were kept, so ${n - can > 1 ? 'they' : 'it'} can't be re-read`
      : 'Read the kept originals again with a model of your choice. Each keeps its folder.';
    const del = b(delArmed ? `Really delete ${n}?` : `Delete ${n}`, delArmed ? `Delete ${n}?` : 'Delete', 'trash', () => bulkDelete(), { cls: 'del' + (delArmed ? ' sure' : ''), disabled: !n });
    del.dataset.del = '1';
    const list = [
      b(n === 1 ? 'Ask about it' : `Ask about ${n}`, 'Ask', 'chat', () => askAbout({ docs: [...sel.ids] }), { disabled: !n, title: 'Questions search only the selected documents' }),
      b('Move to…', 'Move', 'move', e => moveDialog([...sel.ids], e.currentTarget), { disabled: !n }),
      b(can ? `Re-read ${can}…` : 'Re-read…', 'Re-read', 'reread', () => openReread(rereadableOf([...sel.ids]), [...sel.ids]), { disabled: !can, title: tip }),
      del,
      b('', 'More', 'more', e => openMenu({ anchor: e.currentTarget, title: plural(n, 'document') + ' selected', items: [
        { label: 'New folder from these', icon: 'plus', disabled: !n, onclick: () => openGroup([...sel.ids]) },
        { label: 'Select all shown', icon: 'check', onclick: selectAll },
        { label: 'Clear the selection', icon: 'close', onclick: clearSelection },
      ] }), { title: 'More actions' }),
    ];
    if (!phone) list[list.length - 1].setAttribute('aria-label', 'More actions');
    if (phone) [list[0], list[1]] = [list[1], list[0]];  // Move · Ask · Re-read · Delete · ⋯ on a phone
    return list;
  };
}
function disarmDelete(render = true) {
  clearTimeout(delTimer);
  if (!delArmed) return;
  delArmed = false;
  if (render) syncSelection();
}
async function bulkDelete() {
  const ids = [...sel.ids];
  if (!ids.length) return;
  if (!delArmed) {  // the second click within 4 s deletes
    delArmed = true;
    syncSelection();
    delTimer = setTimeout(() => disarmDelete(), 4000);
    return;
  }
  disarmDelete(false);
  await deleteDocs(ids);
}
function toggleSel(id) {
  if (sel.ids.has(id)) sel.ids.delete(id); else sel.ids.add(id);
  sel.anchor = id;
  syncSelection();
  announce(`${sel.ids.size} selected`);
}
function selectRange(id) {
  const a = view.order.indexOf(sel.anchor ?? id), b = view.order.indexOf(id);
  if (b < 0) return;
  const [lo, hi] = a < 0 ? [b, b] : [Math.min(a, b), Math.max(a, b)];
  for (let i = lo; i <= hi; i++) sel.ids.add(view.order[i]);
  syncSelection();
  announce(`${sel.ids.size} selected`);
}
function selectAll() {
  for (const id of view.order) sel.ids.add(id);
  syncSelection();
  announce(`${sel.ids.size} selected`);
}
function clearSelection() {
  sel.ids.clear();
  view.selecting = false;
  syncSelection();
  announce('Selection cleared');
}
$('dselclear').onclick = $('dselcancel').onclick = () => { clearSelection(); focusList(); };
for (const box of [$('dselacts'), $('dbulk')]) {
  box.addEventListener('keydown', e => { if (e.key === 'Delete' && e.target.dataset.act === 'trash') { e.preventDefault(); bulkDelete(); } });
}
$('dselall').onclick = selectAll;
const selecting = () => (coarse.matches || isS.matches) && (sel.ids.size > 0 || view.selecting);

$('dlist').addEventListener('mousedown', e => { if (e.shiftKey && e.target.closest('[data-nav]')) e.preventDefault(); });  // no text selection
// Selecting on a phone: a tap on a card's suggestion chip, unit badge or folder link selects the
// card like a tap anywhere else on it, and never files, opens or navigates.
$('dlist').addEventListener('click', e => {
  const b = e.target.closest('button.sug, button.ubadge, button.pathbtn'), item = b?.closest('[data-nav^="d"]');
  if (!item || !selecting()) return;
  e.stopPropagation();
  e.preventDefault();
  toggleSel(+item.dataset.nav.slice(1));
}, true);
// Whatever item gets focus (Tab, a click, a script) becomes the listing's one tab stop.
$('dlist').addEventListener('focusin', e => {
  const item = e.target.closest('[data-nav]');
  if (!item || item.tabIndex === 0) return;
  for (const x of navItems()) x.tabIndex = x === item ? 0 : -1;
  view.focusKey = item.dataset.nav;
});
$('dlist').addEventListener('click', e => {
  const t = e.target, item = t.closest('[data-nav]');
  if (t.closest('button.ubadge, button.sug, button.pathbtn, a, .hit')) return;
  if (t.closest('.selall')) {
    if (t.checked) selectAll(); else clearSelection();
    return;
  }
  if (!item) { if (!t.closest('button, input, a, label, summary')) { if (sel.ids.size) clearSelection(); } return; }
  const key = item.dataset.nav;
  view.focusKey = key;
  for (const x of navItems()) x.tabIndex = x === item ? 0 : -1;
  if (t.closest('.more')) { itemMenu(key, t.closest('.more')); return; }
  if (key[0] === 'g') { revealJob(key); return; }
  if (key === 'pall') { openPlace({ type: 'all' }, { focus: true }); return; }
  if (key[0] === 'f') { openPlace({ type: 'folder', id: +key.slice(1) }, { focus: true }); return; }
  const id = +key.slice(1);
  if (t.closest('.cbox, .csel')) {
    if (e.shiftKey) selectRange(id); else toggleSel(id);
    setTimeout(syncSelection);  // the box's own toggle, put right
    return;
  }
  if (e.shiftKey) selectRange(id);
  else if (e.ctrlKey || e.metaKey) toggleSel(id);
  else if (selecting()) toggleSel(id);
  else openDoc(id);
});
$('dlist').addEventListener('contextmenu', e => {
  const item = e.target.closest('[data-nav]');
  if (!item || !fine.matches || item.dataset.nav[0] === 'g' || item.dataset.nav === 'pall') return;
  e.preventDefault();
  itemMenu(item.dataset.nav, null, { x: e.clientX, y: e.clientY });
});
// Up and down by rows, as laid out: the nearest item in the next row in that direction.
function vertical(item, items, dir) {
  const r = item.getBoundingClientRect(), cx = r.left + r.width / 2;
  let best = null, bestEdge = dir > 0 ? Infinity : -Infinity, bestDx = Infinity;
  for (const o of items) {
    if (o === item) continue;
    const q = o.getBoundingClientRect();
    const edge = dir > 0 ? q.top : q.bottom;
    if (dir > 0 ? q.top < r.bottom - 2 : q.bottom > r.top + 2) continue;
    const dx = Math.abs(q.left + q.width / 2 - cx);
    if (dir > 0 ? edge < bestEdge - 2 || (Math.abs(edge - bestEdge) <= 2 && dx < bestDx) : edge > bestEdge + 2 || (Math.abs(edge - bestEdge) <= 2 && dx < bestDx)) {
      best = o; bestEdge = edge; bestDx = dx;
    }
  }
  return best;
}
$('dlist').addEventListener('keydown', e => {
  const item = e.target.closest('[data-nav]');
  if (!item || e.target !== item) return;
  const items = navItems(), i = items.indexOf(item), key = item.dataset.nav, list = item.tagName === 'TR';
  const mod = e.ctrlKey || e.metaKey, doc = key[0] === 'd' ? +key.slice(1) : null, folder = key[0] === 'f' ? +key.slice(1) : null;
  let next = null, done = true;
  if (e.key === 'ArrowRight' && !list) next = items[i + 1];
  else if (e.key === 'ArrowLeft' && !list) next = items[i - 1];
  else if (e.key === 'ArrowDown') next = list ? items[i + 1] : vertical(item, items, 1);
  else if (e.key === 'ArrowUp' && e.altKey) openPlace(parentPlace(view.place), { focus: true });
  else if (e.key === 'ArrowUp') next = list ? items[i - 1] : vertical(item, items, -1);
  else if (e.key === 'Home') next = items[0];
  else if (e.key === 'End') next = items[items.length - 1];
  else if (e.key === 'Enter') {
    if (key[0] === 'g') revealJob(key);
    else if (key === 'pall') openPlace({ type: 'all' }, { focus: true });
    else if (folder != null) openPlace({ type: 'folder', id: folder }, { focus: true });
    else openDoc(doc);
  } else if (e.key === ' ') { if (doc != null) { if (e.shiftKey) selectRange(doc); else toggleSel(doc); } }
  else if ((e.key === 'a' || e.key === 'A') && mod) selectAll();
  else if (e.key === 'Delete') {
    if (doc != null && !sel.ids.size) { sel.ids.add(doc); syncSelection(); }
    if (sel.ids.size) {
      const armed = delArmed;
      bulkDelete();
      if (!armed) {  // armed: the bar's Delete has focus, and Delete again deletes
        (isS.matches ? $('dbulk') : $('dselacts')).querySelector('[data-act="trash"]')?.focus();
        announce(`Press Delete again to delete ${plural(sel.ids.size, 'document')}`);
      }
    }
  } else if ((e.key === 'F10' && e.shiftKey) || e.key === 'ContextMenu') { if (key[0] !== 'g' && key !== 'pall') itemMenu(key, item.querySelector('.more')); }
  else if (e.key === 'F2') { if (folder != null) renameFolder(folder); else if (doc != null) openDoc(doc, { title: true }); }
  else if (e.key === 'Backspace') { if (view.place.type !== 'root') openPlace(parentPlace(view.place), { focus: true }); }
  else done = false;
  if (next) {
    for (const x of items) x.tabIndex = x === next ? 0 : -1;
    view.focusKey = next.dataset.nav;
    next.focus();
    if (e.shiftKey && next.dataset.nav[0] === 'd') selectRange(+next.dataset.nav.slice(1));
  }
  if (done) e.preventDefault();
});
// "/" searches: only from inside Documents and not while typing (WCAG 2.1.4).
document.addEventListener('keydown', e => {
  if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey || view.mode !== 'docs') return;
  const a = document.activeElement;
  if (!$('docsview').contains(a) || a.matches('input, textarea, select, [contenteditable]')) return;
  e.preventDefault();
  $('dq').focus();
});
// Escape, after menus and dialogs (native): the detail, then the search, then the selection.
function docsEscape() {
  if (view.doc != null) { closeDoc(); return; }
  if (view.q) { clearSearch(); return; }
  if (sel.ids.size || view.selecting) { clearSelection(); focusList(); }
}

// ---- jobs in the listing: ghost cards for uploads, strips for re-reads ----
let jobsTick = 0;
function jobsChanged() {
  if (jobsTick) return;
  jobsTick = requestAnimationFrame(() => {
    jobsTick = 0;
    if (view.mode !== 'docs' || !lib.loaded) return;
    const ghosts = ghostRows(), sig = ghosts.map(ghostId).join(',');
    if (sig !== ($('dlist').dataset.ghosts || '')) { renderDocs(); return; }
    for (const row of ghosts) {
      const old = $('dlist').querySelector(`[data-nav="g${ghostId(row)}"]`);
      if (!old) continue;
      const fresh = old.tagName === 'TR' ? ghostRow(row) : ghostCard(row);
      fresh.tabIndex = old.tabIndex;
      const focused = document.activeElement === old;
      old.replaceWith(fresh);
      if (focused) fresh.focus({ preventScroll: true });
    }
    syncStrips();
  });
}
function rereadRows() {
  const m = new Map();
  for (const row of $('queue').children) { const j = row.job; if (j?.kind === 'reread' && ACTIVE.has(j.status) && j.doc_id) m.set(j.doc_id, row); }
  return m;
}
function strip(row) {
  const s = el('div', 'jobstrip'), v = row.view || {};
  s.append(el('span', '', `Re-reading · ${v.stage || v.state || ''}`));
  const bar = el('div', 'bar' + (v.bar === 'wait' || v.bar == null ? ' wait' : '')), i = el('i');
  if (typeof v.bar === 'number') i.style.width = `${Math.round(v.bar * 100)}%`;
  bar.append(i);
  s.append(bar);
  return s;
}
function syncStrips() {
  const rows = rereadRows();
  for (const card of $('dlist').querySelectorAll('.dcard[data-nav^="d"]')) {
    const row = rows.get(+card.dataset.nav.slice(1)), th = card.querySelector('.thumb');
    th.querySelector('.jobstrip')?.remove();
    if (row) th.append(strip(row));
  }
  syncPanelJob();
}

function revealJob(key) {
  const row = [...$('queue').children].find(r => 'g' + ghostId(r) === key);
  if (!row) return;
  if (isS.matches) $('uploads').open = true;
  row.scrollIntoView({ block: 'nearest' });
  row.focus({ preventScroll: true });
  row.classList.add('flash');
  setTimeout(() => row.classList.remove('flash'), 1200);
}

// ---- moving, filing, deleting ----
const recent = () => { try { return (JSON.parse(store.get('moveRecent') || '[]') || []).filter(k => k === 'unfiled' || Number.isInteger(k)); } catch { return []; } };
function remember(fid) { store.set('moveRecent', JSON.stringify([fid ?? 'unfiled', ...recent().filter(k => k !== (fid ?? 'unfiled'))].slice(0, 5))); }
const docsWord = (ids) => ids.length === 1 && lib.docs.get(ids[0]) ? `'${docTitle(lib.docs.get(ids[0]))}'` : plural(ids.length, 'document');

// Documents into a folder (null: Unfiled) or a path whose missing folders are made. They leave the
// view at once, come back if the call fails, and an Undo puts each back where it was.
async function moveDocs(ids, folderId, { path = null, quiet = false } = {}) {
  ids = ids.filter(id => lib.docs.has(id));
  if (!ids.length) return false;
  const what = docsWord(ids), before = new Map(ids.map(id => [id, { ...lib.docs.get(id) }]));
  if (!path) {
    for (const id of ids) Object.assign(lib.docs.get(id), { folder_id: folderId, filed_by: null, suggested_folder: null });
    rebuild();
    render();
  }
  let r;
  try { r = await send('/api/documents/move', path ? { ids, path } : { ids, folder_id: folderId }); } catch (err) {
    for (const [id, d] of before) if (lib.docs.has(id)) lib.docs.set(id, d);
    rebuild();
    render();
    toast(err.message, { error: true });
    if (err.status === 404) lib.load();
    return false;
  }
  remember(r.folder_id);
  for (const id of ids) sel.ids.delete(id);
  await lib.load();
  const dest = pathText(r.folder_id);
  const msg = r.created?.length ? `Created '${lib.folders.get(r.folder_id)?.name || dest}' and moved ${what} into it` : `Moved ${what} to ${dest}`;
  announce(msg);
  if (!quiet) toast(msg, { action: ['Undo', () => undoMove(r.from, r.created)] });
  return r;
}
// Each document back where it was; folders the move made (`created`, from the top down) go too,
// deepest first, if nothing has been put in them since.
async function undoMove(from, created = []) {
  const back = new Map();
  for (const [id, f] of Object.entries(from)) { const k = f ?? 'null'; if (!back.has(k)) back.set(k, []); back.get(k).push(+id); }
  try {
    for (const [k, ids] of back) await send('/api/documents/move', { ids, folder_id: k === 'null' ? null : +k });
    let removed = 0;
    if (created?.length) {
      const list = await api('/api/folders'), gone = new Set();
      for (const id of [...created].reverse()) {
        const f = list.find(x => x.id === id);
        if (!f || f.total_count || list.some(x => x.parent_id === id && !gone.has(x.id))) continue;
        await api(`/api/folders/${id}?contents=move`, { method: 'DELETE' });
        gone.add(id);
        removed++;
      }
    }
    toast(`Moved ${plural(Object.keys(from).length, 'document')} back${removed ? ` and removed the ${removed === 1 ? 'folder' : 'folders'} made for ${Object.keys(from).length === 1 ? 'it' : 'them'}` : ''}`);
  } catch (err) { toast(err.message, { error: true }); }
  lib.load();
}
// A suggestion chip or button: into the folder if it exists, else made first (the click is the consent).
async function applySuggestion(docId, path) {
  const fid = findPath(path);
  const r = fid != null ? await moveDocs([docId], fid) : await moveDocs([docId], null, { path });
  return !!r;
}
async function moveFolder(id, parentId) {
  const f = lib.folders.get(id);
  if (!f) return;
  const old = f.parent_id ?? null;
  if (old === parentId) return;
  f.parent_id = parentId;
  rebuild();
  render();
  try { await send(`/api/folders/${id}`, { parent_id: parentId }, 'PATCH'); } catch (err) {
    if (lib.folders.has(id)) lib.folders.get(id).parent_id = old;
    rebuild();
    render();
    toast(err.status === 409 ? `${err.message.replace(/"/g, "'")}` : err.message, { error: true });
    if (err.status === 404) lib.load();
    return;
  }
  remember(parentId);
  if (parentId != null) { treeOpen.add(parentId); saveTree(); }
  await lib.load();
  const msg = `Moved '${f.name}' to ${parentId == null ? 'Library' : pathText(parentId)}`;
  announce(msg);
  toast(msg, { action: ['Undo', async () => {
    try { await send(`/api/folders/${id}`, { parent_id: old }, 'PATCH'); toast(`Moved '${f.name}' back`); } catch (err) { toast(err.message, { error: true }); }
    lib.load();
  }] });
}
async function deleteDocs(ids) {
  const what = docsWord(ids), order = view.order, at = Math.max(0, ...ids.map(id => order.indexOf(id)));
  const nextId = order.slice(at + 1).find(id => !ids.includes(id)) ?? [...order.slice(0, at)].reverse().find(id => !ids.includes(id));
  try { await send('/api/documents/delete', { ids }); } catch (err) { toast(err.message, { error: true }); return; }
  for (const id of ids) { sel.ids.delete(id); lib.docs.delete(id); }
  if (ids.includes(view.doc)) {
    view.doc = null;
    if (history.state?.app && history.state.detail) history.back(); else nav(hashFor(view.place), 'replace');
  }
  rebuild();
  render();
  toast(`Deleted ${what}`);
  announce(`Deleted ${what}`);
  view.focusKey = nextId != null ? 'd' + nextId : null;
  focusList();
  lib.load();
}
function moveDialog(ids, anchor) {
  pickFolder({ mode: 'docs', ids }).then(t => { if (t !== undefined) moveDocs(ids, t); });
}

// ---- menus and sheets: one helper, anchored on wider screens, a bottom sheet on a phone ----
let menuState = null;
function openMenu({ anchor = null, at = null, title = '', subtitle = '', items }) {
  const d = $('menusheet');
  if (d.open) { menuState?.anchor?.removeAttribute('aria-expanded'); menuState = null; d.close(); }
  d.replaceChildren(el('div', 'mgrab'));
  const head = el('div', 'mhead'), ht = el('div'), b = el('b', '', title || 'Actions');
  b.id = 'menutitle';
  ht.append(b);
  if (subtitle) ht.append(el('span', 'muted', subtitle));
  head.append(ht, btn('', () => d.close(), { cls: 'icon', aria: 'Close', icon: 'close' }));
  const list = el('div');
  list.setAttribute('role', 'menu');
  list.setAttribute('aria-label', title || 'Actions');
  for (const it of items) {
    if (!it) continue;
    if (it === '-') { list.append(el('hr')); continue; }
    if (it.note) { list.append(el('div', 'mnote', it.note)); continue; }
    const item = el('button', it.danger ? 'danger' : '');
    item.type = 'button';
    item.setAttribute('role', it.radio ? 'menuitemradio' : 'menuitem');
    if (it.radio) {
      item.setAttribute('aria-checked', String(!!it.checked));
      const ck = el('span', 'check');
      if (it.checked) ck.append(icon('check'));
      item.append(ck);
    } else if (it.icon) item.append(icon(it.icon));
    const lab = el('span', '', it.label);
    item.append(lab);
    if (it.indent) item.style.paddingLeft = `calc(10px + ${it.indent} * 14px)`;  // a path, one level per step
    if (it.title) item.title = it.title;
    item.disabled = !!it.disabled;
    item.onclick = () => {
      if (it.sure && !item.classList.contains('sure')) {  // a second click within 3 s
        item.classList.add('sure');
        lab.textContent = it.sure;
        setTimeout(() => { item.classList.remove('sure'); lab.textContent = it.label; }, 3000);
        return;
      }
      if (menuState) menuState.picked = true;
      d.close();
      it.onclick?.();
    };
    list.append(item);
  }
  d.append(head, list, btn('Cancel', () => d.close(), { cls: 'mcancel' }));
  menuState = { anchor, picked: false };
  anchor?.setAttribute('aria-expanded', 'true');  // also keeps its card or row highlighted meanwhile
  d.style.left = d.style.top = '';
  d.showModal();
  if (!isS.matches) {
    const w = d.offsetWidth, h = d.offsetHeight;
    let x, y;
    if (at) { x = at.x; y = at.y; } else if (anchor?.isConnected) {
      const r = anchor.getBoundingClientRect();
      x = r.left; y = r.bottom + 4;
      if (x + w > innerWidth - 8) x = r.right - w;
      if (y + h > innerHeight - 8) y = r.top - h - 4;
    } else { x = (innerWidth - w) / 2; y = (innerHeight - h) / 2; }
    d.style.left = `${Math.max(8, Math.min(x, innerWidth - w - 8))}px`;
    d.style.top = `${Math.max(8, Math.min(y, innerHeight - h - 8))}px`;
  }
  d.querySelector('[role^="menuitem"]:not(:disabled)')?.focus();
}
$('menusheet').addEventListener('click', e => { if (e.target === $('menusheet')) $('menusheet').close(); });
$('menusheet').addEventListener('close', () => {
  if ($('menusheet').open) return;  // reopened for another menu before this event came: that one's state stays
  const s = menuState;
  menuState = null;
  s?.anchor?.removeAttribute('aria-expanded');
  if (!s || document.querySelector('dialog[open]')) return;
  const a = document.activeElement;
  if (s.picked && a && a !== document.body && a.isConnected) return;  // the item moved focus on
  if (s.anchor?.isConnected) {
    const host = s.anchor.closest('[data-nav], .tnode');
    (s.anchor.tabIndex < 0 && host ? host : s.anchor).focus({ preventScroll: true });
  } else focusList();
});
$('menusheet').addEventListener('keydown', e => {
  const items = [...$('menusheet').querySelectorAll('[role^="menuitem"]:not(:disabled)')], i = items.indexOf(document.activeElement);
  const to = x => { if (x) { e.preventDefault(); x.focus(); } };
  if (e.key === 'ArrowDown') to(items[(i + 1) % items.length]);
  else if (e.key === 'ArrowUp') to(items[(i - 1 + items.length) % items.length]);
  else if (e.key === 'Home') to(items[0]);
  else if (e.key === 'End') to(items[items.length - 1]);
  else if (e.key === 'Tab') e.preventDefault();
  else if (typeAhead(e, items, x => x.textContent, document.activeElement)) e.preventDefault();
});

const fileSubtitle = d => d.filename !== docTitle(d) ? d.filename : '';  // a menu's file name, unless it is the title
function docMenuItems(id) {
  const d = lib.docs.get(id);
  if (!d) return [];
  return [
    { label: 'Show details', icon: 'page', onclick: () => openDoc(id) },
    d.has_file ? { label: 'Open original ↗', icon: 'external', onclick: () => window.open(`/api/documents/${id}/file`, '_blank', 'noopener') } : null,
    { label: 'Ask about this', icon: 'chat', onclick: () => askAbout({ docs: [id] }) },
    { label: 'Move to…', icon: 'move', onclick: () => moveDialog([id]) },
    { label: 'Rename…', icon: 'pencil', onclick: () => openDoc(id, { title: true }) },
    { label: 'Re-read…', icon: 'reread', disabled: !d.has_file, title: d.has_file ? '' : "The original wasn't kept, so it can't be re-read", onclick: () => openReread([id], [id]) },
    '-',
    { label: 'Delete', icon: 'trash', danger: true, sure: 'Click again to delete', onclick: () => deleteDocs([id]) },
  ];
}
function folderMenu(id, anchor, at, here = false) {
  const f = lib.folders.get(id);
  if (!f) return;
  openMenu({ anchor, at, title: f.name, subtitle: f.names.length > 1 ? f.names.slice(0, -1).join(' › ') : '', items: [
    here ? null : { label: 'Open', icon: 'folder-open', onclick: () => openPlace({ type: 'folder', id }, { focus: true }) },
    { label: 'Ask about this folder', icon: 'chat', onclick: () => askAbout({ folders: [id] }) },
    { label: 'New folder inside', icon: 'plus', onclick: () => newFolder(id) },
    { label: 'Rename…', icon: 'pencil', onclick: () => renameFolder(id) },
    { label: 'Move to…', icon: 'move', onclick: () => pickFolder({ mode: 'folder', folder: id }).then(t => { if (t !== undefined) moveFolder(id, t); }) },
    '-',
    { label: 'Delete folder…', icon: 'trash', danger: true, onclick: () => deleteFolder(id) },
  ] });
}
function placeMenu(key, anchor, at = null) {
  if (key[0] === 'f') { folderMenu(+key.slice(1), anchor, at); return; }
  const n = unfiledIds().length;
  const items = key === 'root' ? [{ label: 'New folder', icon: 'plus', onclick: () => newFolder(null) }]
    : key === 'unfiled' ? [{ label: 'Open', icon: 'inbox', onclick: () => openPlace({ type: 'unfiled' }, { focus: true }) },
        { label: `Ask about these ${n}`, icon: 'chat', disabled: !n, title: "Documents added to Unfiled later aren't included", onclick: () => askPlace('unfiled') }]
    : [{ label: 'Open', icon: 'stack', onclick: () => openPlace({ type: 'all' }, { focus: true }) },
       { label: 'Ask all documents', icon: 'chat', onclick: () => askAbout({}) }];
  openMenu({ anchor, at, title: placeName(treePlace(key)), items });
}
// The menu of a document that is one of several selected acts on them all, as file managers' do and
// as the selection bar does; each item says how many.
function selectionMenuItems() {
  const ids = [...sel.ids].filter(id => lib.docs.has(id)), n = ids.length, can = rereadableOf(ids);
  return [
    { label: `Ask about these ${n}`, icon: 'chat', onclick: () => askAbout({ docs: ids }) },
    { label: `Move ${n} documents to…`, icon: 'move', onclick: () => moveDialog(ids) },
    { label: can.length ? `Re-read ${can.length}…` : 'Re-read…', icon: 'reread', disabled: !can.length,
      title: can.length < n ? `${n - can.length} of them can't be re-read: their originals weren't kept` : '', onclick: () => openReread(can, ids) },
    { label: 'New folder from these', icon: 'plus', onclick: () => openGroup(ids) },
    '-',
    { label: `Delete ${n} documents`, icon: 'trash', danger: true, sure: `Click again to delete ${n}`, onclick: () => deleteDocs(ids) },
  ];
}
function itemMenu(key, anchor, at = null) {
  if (key[0] === 'f') { folderMenu(+key.slice(1), anchor, at); return; }
  if (key[0] !== 'd') return;
  const id = +key.slice(1), d = lib.docs.get(id);
  if (!d) return;
  if (sel.ids.has(id) && sel.ids.size > 1) openMenu({ anchor, at, title: `${sel.ids.size} documents selected`, items: selectionMenuItems() });
  else openMenu({ anchor, at, title: docTitle(d), subtitle: fileSubtitle(d), items: docMenuItems(d.id) });
}

// ---- dialogs ----
for (const d of document.querySelectorAll('dialog.form')) {
  d.addEventListener('click', e => { if (e.target === d) d.close(); });  // the backdrop
  for (const b of d.querySelectorAll('.dclose')) b.onclick = () => d.close();
}
// Opened from somewhere: focus goes back there when it closes (unless another dialog took it).
function openDialog(d) {
  const opener = document.activeElement;
  d.showModal();
  d.addEventListener('close', () => {
    setTimeout(() => {
      if (document.querySelector('dialog[open]')) return;
      const a = document.activeElement;
      if (a && a !== document.body && a.isConnected && !d.contains(a)) return;
      if (opener?.isConnected && opener !== document.body) opener.focus({ preventScroll: true }); else focusList();
    });
  }, { once: true });
}

// New folder and Rename: one dialog. Resolves with the folder, or null.
let fv = null;
const badName = name => !name ? 'A folder needs a name' : name.length > 100 ? 'Folder names are 100 characters at most'
  : name.includes('/') ? "Folder names can't contain /" : /[\x00-\x1f\x7f]/.test(name) ? "Folder names can't contain control characters" : '';
function folderDialog({ parent = null, folder = null } = {}) {
  return new Promise(resolve => {
    if (fv) fv.resolve(null);
    fv = { parent, folder, resolve };
    const pf = parent != null && lib.folders.get(parent);
    $('fvtitle').textContent = folder ? `Rename '${folder.name}'` : pf ? `New folder in ${pf.name}` : 'New folder';
    $('fvgo').textContent = folder ? 'Rename' : 'Create';
    $('fvname').value = folder ? folder.name : '';
    $('fverr').textContent = '';
    $('fvgo').disabled = !$('fvname').value.trim();
    openDialog($('folderview'));
    $('fvname').focus();
    if (folder) $('fvname').select();
  });
}
$('fvname').oninput = () => { $('fvgo').disabled = !$('fvname').value.trim(); $('fverr').textContent = ''; };
$('folderview').addEventListener('close', () => { if (fv) { const r = fv.resolve; fv = null; r(null); } });
$('fvform').onsubmit = async e => {
  e.preventDefault();
  if (!fv) return;
  const name = $('fvname').value.split(/\s+/).filter(Boolean).join(' '), bad = badName(name);
  if (bad) { $('fverr').textContent = bad; return; }
  const { folder, parent } = fv;
  $('fvgo').disabled = true;
  let f;
  try { f = folder ? await send(`/api/folders/${folder.id}`, { name }, 'PATCH') : await send('/api/folders', { name, parent_id: parent }); } catch (err) {
    $('fverr').textContent = err.status === 404 ? 'That folder was deleted' : err.message;
    $('fvgo').disabled = false;
    if (err.status === 404) lib.load();
    return;
  }
  const r = fv.resolve;
  fv = null;
  $('folderview').close();
  if (parent != null) { treeOpen.add(parent); saveTree(); }
  await lib.load();
  r(f);
};
async function newFolder(parent) {
  const f = await folderDialog({ parent });
  if (!f) return null;
  toast(`Created '${f.name}'`, { action: ['Open', () => openPlace({ type: 'folder', id: f.id }, { focus: true })] });
  announce(`Folder ${f.name} created`);
  const row = $('tree').querySelector(`.tnode[data-key="f${f.id}"] > .trow`);
  if (row) { row.classList.add('flash'); row.scrollIntoView({ block: 'nearest' }); setTimeout(() => row.classList.remove('flash'), 1000); }
  return f;
}
async function renameFolder(id) {
  const f = lib.folders.get(id);
  if (!f) return;
  const r = await folderDialog({ folder: f });
  if (r) { toast(`Renamed to '${r.name}'`); announce(`Renamed to ${r.name}`); }
}

// The move picker: one tree for every width. Resolves with a folder id, null (Unfiled or the top
// level), or undefined when cancelled. mode: 'docs' (moving documents), 'folder' (moving one),
// 'dest' (where a new folder goes).
let pk = null;
function pickFolder({ mode, ids = [], folder = null, title = '' }) {
  return new Promise(resolve => {
    if (pk) pk.resolve(undefined);
    const f = folder != null ? lib.folders.get(folder) : null;
    const current = new Set();
    if (mode === 'docs') { const at = new Set(ids.map(id => { const d = lib.docs.get(id); return d && !isUnfiled(d) ? 'f' + d.folder_id : 'unfiled'; })); if (at.size === 1) current.add([...at][0]); }
    if (mode === 'folder' && f) current.add(f.parent_id != null ? 'f' + f.parent_id : 'root');
    const exclude = mode === 'folder' && f ? subtree(f.id) : new Set();
    const open = new Set();
    for (const k of current) if (k[0] === 'f') { open.add(+k.slice(1)); for (let p = lib.folders.get(+k.slice(1)); p; p = lib.folders.get(p.parent_id)) open.add(p.id); }
    pk = { mode, ids, folder: f, current, exclude, open, chosen: null, resolve };
    $('pktitle').textContent = title || (mode === 'folder' ? `Move folder '${f?.name}' to…` : mode === 'dest' ? 'Put the new folder in…'
      : ids.length === 1 ? `Move '${docTitle(lib.docs.get(ids[0]))}' to…` : `Move ${plural(ids.length, 'document')} to…`);
    $('pkq').value = '';
    $('pkerr').textContent = '';
    drawPicker();
    openDialog($('pickview'));
    ($('pktree').querySelector('.pkrow[tabindex="0"]') || $('pkq')).focus();
  });
}
const firstKey = () => pk.mode === 'docs' ? 'unfiled' : 'root';
const pkTarget = key => key === 'unfiled' || key === 'root' ? null : +key.slice(1);
function pkWhy(key) {
  if (pk.current.has(key)) return '(current)';
  if (key[0] === 'f' && pk.exclude.has(+key.slice(1))) return "A folder can't go inside itself";
  return '';
}
function pkLabel(key) { return key === 'unfiled' ? 'Unfiled' : key === 'root' ? 'Library' : lib.folders.get(+key.slice(1))?.name || ''; }
function drawPicker() {
  const q = norm($('pkq').value.trim());
  const rows = [];
  const row = (key, label, level, { expandable = false, path = '' } = {}) => {
    const r = el('li', 'pkrow');
    r.setAttribute('role', 'treeitem');
    r.dataset.key = key;
    r.tabIndex = -1;
    r.style.setProperty('--lvl', String(Math.min(level, 8)));
    r.setAttribute('aria-level', String(level + 1));
    const why = pkWhy(key);
    if (why) { r.setAttribute('aria-disabled', 'true'); r.title = why; }
    r.setAttribute('aria-selected', String(pk.chosen === key));
    const chev = el('button', 'chev' + (expandable ? '' : ' none'));
    chev.type = 'button';
    chev.tabIndex = -1;
    chev.setAttribute('aria-label', 'Expand');
    chev.append(icon('chev'));
    if (expandable) { const open = q || pk.open.has(+key.slice(1)); r.setAttribute('aria-expanded', String(!!open)); }
    const name = el('span', 'pkname', label);
    r.append(chev, icon(key === 'unfiled' ? 'inbox' : key === 'root' ? 'home' : 'folder'), name);
    if (path) r.append(el('span', 'pkpath', path));
    if (why) r.append(el('span', 'why', why === '(current)' ? '(current)' : ''));
    rows.push(r);
  };
  row(firstKey(), pk.mode === 'docs' ? 'Unfiled (not in a folder)' : 'Library (top level)', 0);
  if (q) {  // matches, with the folders above them
    const keep = new Set();
    for (const f of lib.folders.values()) if (norm(f.name).includes(q)) for (let p = f; p; p = lib.folders.get(p.parent_id)) keep.add(p.id);
    const add = (key, level) => { for (const id of lib.kids.get(key) || []) if (keep.has(id)) { row('f' + id, lib.folders.get(id).name, level, { expandable: lib.kids.get(id).some(k => keep.has(k)) }); add(id, level + 1); } };
    add('root', 0);
    if (rows.length === 1) rows.push(el('li', 'muted', 'No folder matches.'));
  } else {
    const add = (key, level) => {
      for (const id of lib.kids.get(key) || []) {
        const has = lib.kids.get(id).length > 0;
        row('f' + id, lib.folders.get(id).name, level, { expandable: has });
        if (has && pk.open.has(id)) add(id, level + 1);
      }
    };
    add('root', 0);
  }
  $('pktree').replaceChildren(...rows);
  const rec = recent().map(k => k === 'unfiled' ? (pk.mode === 'docs' ? 'unfiled' : null) : lib.folders.has(k) ? 'f' + k : null)
    .filter(k => k && !pkWhy(k)).slice(0, 3);
  if (!q && rec.length) {
    const box = [el('div', 'label', 'Recent')];
    for (const k of rec) {
      const r = el('div', 'pkrow');
      r.dataset.key = k;
      r.tabIndex = -1;
      r.setAttribute('role', 'button');
      r.setAttribute('aria-pressed', String(pk.chosen === k));
      if (pk.chosen === k) r.setAttribute('aria-selected', 'true');
      r.append(icon(k === 'unfiled' ? 'inbox' : 'folder'), el('span', 'pkname', k === 'unfiled' ? 'Unfiled' : pathText(+k.slice(1))));
      box.push(r);
    }
    $('pkrecent').replaceChildren(...box);
  } else $('pkrecent').replaceChildren();
  const items = [...$('pktree').querySelectorAll('.pkrow')];
  const focusKey = pk.chosen || (items.find(r => !r.hasAttribute('aria-disabled'))?.dataset.key);
  for (const r of items) r.tabIndex = r.dataset.key === focusKey ? 0 : -1;
  const go = $('pkgo');
  go.disabled = !pk.chosen || !!pkWhy(pk.chosen);
  go.textContent = pk.mode === 'dest' ? (pk.chosen ? `Choose ${pkLabel(pk.chosen)}` : 'Choose') : pk.chosen ? `Move to ${pkLabel(pk.chosen)}` : 'Move';
}
function pkChoose(key, now = false) {
  if (!key || pkWhy(key)) return;
  pk.chosen = key;
  if (now) { pkDone(); return; }
  drawPicker();
  const r = $('pktree').querySelector(`.pkrow[data-key="${key}"]`) || $('pkrecent').querySelector(`.pkrow[data-key="${key}"]`);
  r?.focus();
}
function pkDone() {
  if (!pk?.chosen || pkWhy(pk.chosen)) return;
  const r = pk.resolve, t = pkTarget(pk.chosen);
  pk = null;
  $('pickview').close();
  r(t);
}
$('pickview').addEventListener('close', () => { if (pk) { const r = pk.resolve; pk = null; r(undefined); } });
$('pkq').oninput = drawPicker;
$('pkq').onkeydown = e => { if (e.key === 'ArrowDown') { e.preventDefault(); $('pktree').querySelector('.pkrow')?.focus(); } };
for (const box of [$('pktree'), $('pkrecent')]) {
  box.addEventListener('click', e => {
    const r = e.target.closest('.pkrow');
    if (!r) return;
    if (e.target.closest('.chev') && r.hasAttribute('aria-expanded')) {
      const id = +r.dataset.key.slice(1);
      if (pk.open.has(id)) pk.open.delete(id); else pk.open.add(id);
      drawPicker();
      $('pktree').querySelector(`.pkrow[data-key="${r.dataset.key}"]`)?.focus();
      return;
    }
    pkChoose(r.dataset.key);
  });
  box.addEventListener('dblclick', e => { const r = e.target.closest('.pkrow'); if (r) pkChoose(r.dataset.key, true); });
}
$('pktree').addEventListener('keydown', e => {
  const r = e.target.closest('.pkrow');
  if (!r) return;
  const items = [...$('pktree').querySelectorAll('.pkrow')], i = items.indexOf(r);
  const to = x => { if (x) { items.forEach(y => y.tabIndex = y === x ? 0 : -1); x.focus(); } };
  const key = r.dataset.key, id = key[0] === 'f' ? +key.slice(1) : null;
  let done = true;
  if (e.key === 'ArrowDown') to(items[i + 1]);
  else if (e.key === 'ArrowUp') { if (i === 0) $('pkq').focus(); else to(items[i - 1]); }
  else if (e.key === 'Home') to(items[0]);
  else if (e.key === 'End') to(items[items.length - 1]);
  else if (e.key === 'ArrowRight' && r.getAttribute('aria-expanded') === 'false') { pk.open.add(id); drawPicker(); $('pktree').querySelector(`[data-key="${key}"]`)?.focus(); }
  else if (e.key === 'ArrowLeft' && r.getAttribute('aria-expanded') === 'true') { pk.open.delete(id); drawPicker(); $('pktree').querySelector(`[data-key="${key}"]`)?.focus(); }
  else if (e.key === 'Enter') pkChoose(key, true);
  else if (e.key === ' ') pkChoose(key);
  else done = typeAhead(e, items, x => x.querySelector('.pkname').textContent, r);
  if (done) e.preventDefault();
});
$('pkgo').onclick = pkDone;
$('pknew').onclick = async () => {
  if (!pk) return;
  const state = pk, parent = state.chosen && state.chosen[0] === 'f' ? +state.chosen.slice(1) : null;
  const f = await folderDialog({ parent });
  if (!f || pk !== state) return;
  if (parent != null) pk.open.add(parent);
  pk.chosen = 'f' + f.id;
  drawPicker();
  $('pktree').querySelector(`[data-key="f${f.id}"]`)?.focus();
};

// Deleting a folder: its contents move up (the default), or everything goes, after a tick and the
// server's count check.
let fd = null;
async function deleteFolder(id) {
  let list;
  try { list = await api('/api/folders'); } catch (err) { toast(err.message, { error: true }); return; }
  const f = list.find(x => x.id === id);
  if (!f) { toast('That folder no longer exists'); lib.load(); return; }
  fd = { id, f, list, mode: 'move' };
  drawFolderDelete();
  openDialog($('fdelview'));
  $('fdelview').querySelector('input[type=radio]:checked, .svfoot .primary')?.focus();
}
function drawFolderDelete(note = '') {
  const { f, list } = fd;
  const byParent = new Map();
  for (const x of list) { const k = x.parent_id ?? 'root'; if (!byParent.has(k)) byParent.set(k, []); byParent.get(k).push(x); }
  const subs = [];
  const walk = k => { for (const c of byParent.get(k) || []) { subs.push(c.id); walk(c.id); } };
  walk(f.id);
  const kids = (byParent.get(f.id) || []).length, total = f.total_count, direct = f.doc_count;
  fd.total = total;
  const parent = f.parent_id != null ? list.find(x => x.id === f.parent_id) : null;
  const body = $('fdbody'), go = $('fdgo'), empty = !total && !kids;
  body.replaceChildren();
  if (note) body.append(el('p', 'err', note));
  if (empty) {
    $('fdtitle').textContent = `Delete the empty folder '${f.name}'?`;
    go.textContent = 'Delete folder';
    go.className = 'primary danger';
    go.disabled = false;
    fd.mode = 'move';
  } else {
    $('fdtitle').textContent = `Delete '${f.name}'?`;
    const holds = [direct || !kids ? plural(direct, 'document') : '', kids ? plural(kids, 'folder') : ''].filter(Boolean).join(' and ');
    body.append(el('p', '', `It holds ${holds}${total !== direct ? ` (${plural(total, 'document')} in all)` : ''}.`));
    const fs = el('fieldset', 'radios'), lg = el('legend', '', 'What happens to what is in it');
    fs.append(lg);
    const radio = (v, text, cls = '') => {
      const l = el('label', cls), r = el('input');
      r.type = 'radio';
      r.name = 'fdmode';
      r.value = v;
      r.checked = fd.mode === v;
      r.onchange = () => { fd.mode = v; fd.sure = false; drawFolderDelete(); $('fdelview').querySelector(`input[value="${v}"]`)?.focus(); };
      l.append(r, el('span', '', text));
      return l;
    };
    fs.append(radio('move', parent ? `Move its contents to ${parent.name}` : 'Move its documents to Unfiled and its folders to the top level'),
      radio('delete', `Delete the folder and the ${plural(total, 'document')} in it`));
    body.append(fs);
    if (fd.mode === 'delete') {
      const l = el('label', 'ck danger'), c = el('input');
      c.type = 'checkbox';
      c.checked = !!fd.sure;
      c.onchange = () => { fd.sure = c.checked; drawFolderDelete(); $('fdelview').querySelector('label.ck input')?.focus(); };
      l.append(c, el('span', '', `I understand: ${plural(total, 'document')}, their files and their text will be deleted. This can't be undone.`));
      body.append(l);
    }
    go.textContent = fd.mode === 'delete' ? `Delete folder and ${plural(total, 'document')}` : 'Delete folder, keep documents';
    go.className = 'primary' + (fd.mode === 'delete' ? ' danger' : '');
    go.disabled = fd.mode === 'delete' && !fd.sure;
  }
  const ids = new Set([f.id, ...subs]);
  if (chats.some(c => c.folder_ids?.some(x => ids.has(x))) || scope.folders.some(x => ids.has(x))) body.append(el('p', 'muted', 'Chats limited to this folder will no longer include it.'));
  const coming = [...$('queue').children].filter(r => { const j = r.job; return j?.kind === 'upload' && ACTIVE.has(j.status) && ids.has(j.options?.folder_id); }).length;
  if (coming) body.append(el('p', 'muted', `${plural(coming, 'upload')} on ${coming === 1 ? 'its' : 'their'} way here will go to ${fd.mode === 'delete' || !parent ? 'Unfiled' : parent.name}.`));
  $('fderr').textContent = '';
}
$('fdgo').onclick = async () => {
  if (!fd) return;
  const { id, f, mode, total } = fd;
  // The place shown goes with the folder: it, or with its documents deleted any folder inside it too
  // (moving its contents up keeps those, at the parent).
  const inside = view.place.type === 'folder' && (mode === 'delete' ? subtree(id).has(view.place.id) : view.place.id === id);
  $('fdgo').disabled = true;
  let r;
  try {
    r = await api(`/api/folders/${id}?contents=${mode}${mode === 'delete' ? `&confirm_documents=${total}` : ''}`, { method: 'DELETE' });
  } catch (err) {
    if (err.status === 409 && err.data?.counts) {  // it changed meanwhile: fresh counts, tick again
      try { fd.list = await api('/api/folders'); fd.f = fd.list.find(x => x.id === id) || f; } catch {}
      fd.sure = false;
      drawFolderDelete(err.message);
    } else { $('fderr').textContent = err.message; $('fdgo').disabled = false; if (err.status === 404) lib.load(); }
    return;
  }
  fd = null;
  $('fdelview').close();
  const parent = r.parent_id;
  if (inside) {  // to where its contents went; an open document stays open if it moved there with them
    view.place = parent != null ? { type: 'folder', id: parent } : { type: 'root' };  // Library lists Unfiled too
    if (mode === 'delete') view.doc = null;
    nav(hashFor(view.place, view.doc), 'replace', view.doc != null && history.state?.detail ? { detail: true } : {});
    store.set('docsAt', hashFor(view.place));
  }
  await lib.load();
  const where = parent != null ? pathText(parent) : 'Unfiled';
  let msg = `Deleted '${f.name}'.`;
  if (mode === 'delete') msg += r.deleted_documents ? ` ${plural(r.deleted_documents, 'document')} deleted.` : '';
  else if (r.moved_documents || r.moved_folders) {
    const moved = [r.moved_documents ? plural(r.moved_documents, 'document') : '', r.moved_folders ? plural(r.moved_folders, 'folder') : ''].filter(Boolean).join(' and ');
    msg += ` ${moved} moved to ${parent != null ? where : r.moved_folders && !r.moved_documents ? 'the top level' : 'Unfiled'}.`;
  }
  for (const x of r.renamed || []) msg += ` '${x.from}' is now '${x.to}': ${parent != null ? where : 'Library'} already had one.`;
  toast(msg);
  announce(msg);
};

// Re-read: from the kept originals, with any model; each keeps its folder and the details edited.
let rr = null;
function openReread(can, all = can) {
  rr = { can };
  const n = all.length, missing = n - can.length;
  $('rrtitle').textContent = n === 1 ? `Re-read '${docTitle(lib.docs.get(all[0]) || { filename: 'document' })}'` : `Re-read ${plural(n, 'document')}`;
  $('rrintro').textContent = `Read again from ${can.length === 1 ? 'its kept original' : 'their kept originals'} with:`;
  $('rrnote').textContent = missing ? `${missing} ${missing === 1 ? 'was' : 'were'} uploaded before originals were kept and can't be re-read.` : '';
  $('rrnote').hidden = !missing;
  $('rrerr').textContent = '';
  $('rrgo').textContent = can.length ? `Re-read ${can.length}` : 'Re-read';
  $('rrgo').disabled = !can.length;
  syncPageBoxes();
  openDialog($('rereadview'));
  $('rereadModel').focus();
}
$('rrgo').onclick = async () => {
  const ids = rr?.can || [], model = $('rereadModel').value;
  if (model.startsWith('claude:') && !claudeReady()) { $('rereadview').close(); askKey(); return; }
  $('rrgo').disabled = true;
  const body = { model, ...(model.startsWith('claude:') ? { claude_pages: claudePages('rereadModel') } : {}) };
  let queued = 0;
  const errors = [];
  for (const id of ids) {
    try {
      const r = await send(`/api/documents/${id}/reread`, body);
      showJob(jobRow(r.job.id), r.job);
      queued++;
    } catch (err) { errors.push(`${lib.docs.get(id)?.filename || id}: ${err.message}`); }
  }
  if (queued) startPolling();
  if (errors.length) { $('rrerr').textContent = errors.join(' · '); $('rrgo').disabled = false; if (!queued) return; }
  else $('rereadview').close();
  sel.ids.clear();
  syncSelection();
  toast(`${plural(queued, 're-read')} queued`);
};

// "New folder from these": named after what they mostly are, made here (or at the top), then filled.
let gv = null;
function openGroup(ids) {
  ids = ids.filter(id => lib.docs.has(id));
  if (!ids.length) return;
  const counts = new Map(), names = new Map();
  for (const id of ids) {
    const raw = (lib.docs.get(id).doc_type || '').trim(), t = norm(raw);
    if (!t) continue;
    counts.set(t, (counts.get(t) || 0) + 1);
    if (!names.has(t)) names.set(t, raw);
  }
  const best = [...counts].sort((a, b) => b[1] - a[1])[0], top = best && [names.get(best[0])];
  gv = { ids, parent: view.place.type === 'folder' ? view.place.id : null };
  $('gvtitle').textContent = `New folder with ${plural(ids.length, 'document')}`;
  $('gvname').value = top ? pluralName(top[0]) : 'New folder';
  $('gverr').textContent = '';
  syncGroupWhere();
  openDialog($('groupview'));
  $('gvname').focus();
  $('gvname').select();
}
const syncGroupWhere = () => { $('gvloc').textContent = gv.parent != null ? pathText(gv.parent) : 'Library (top level)'; };
$('gvchange').onclick = async () => {
  const state = gv, t = await pickFolder({ mode: 'dest' });
  if (t === undefined || gv !== state) return;
  gv.parent = t;
  syncGroupWhere();
};
$('gvname').oninput = () => { $('gverr').textContent = ''; };
$('gvform').onsubmit = async e => {
  e.preventDefault();
  if (!gv) return;
  const name = $('gvname').value.split(/\s+/).filter(Boolean).join(' '), bad = badName(name);
  if (bad) { $('gverr').textContent = bad; return; }
  const { ids, parent } = gv;
  let f;
  try { f = await send('/api/folders', { name, parent_id: parent }); } catch (err) { $('gverr').textContent = err.message; return; }
  gv = null;
  $('groupview').close();
  if (parent != null) { treeOpen.add(parent); saveTree(); }
  const r = await moveDocs(ids, f.id, { quiet: true });
  if (r) toast(`Created '${f.name}' and moved ${plural(ids.length, 'document')} into it`, { action: ['Undo', () => undoMove(r.from, [f.id])] });
};

// Filters: client-side, for the session, shown as chips.
let fdraft = null;
function openFilter() {
  const f = view.filters;
  fdraft = { types: new Set(f.types), from: f.from, to: f.to, amount: f.amount, due: f.due, suggested: f.suggested,
    models: new Set(f.models), partial: f.partial };
  drawFilter();
  openDialog($('filterview'));
  $('fbody').querySelector('button, input')?.focus();
}
function placeDocs() {  // what the place lists before filters (search aside)
  const p = view.place;
  const ids = p.type === 'all' ? [...lib.docs.keys()] : p.type === 'root' || p.type === 'unfiled' ? unfiledIds()
    : lib.folders.has(p.id) ? (view.flat ? subtreeDocs(p.id) : lib.docsIn.get(p.id)) : [];
  return ids.map(id => lib.docs.get(id));
}
function drawFilter() {
  const f = fdraft, body = $('fbody'), docs = placeDocs();
  const types = new Map();
  for (const d of docs) if (d.doc_type) { const k = norm(d.doc_type); types.set(k, (types.get(k) || 0) + 1); }
  const secT = el('div', 'fsec'), chips = el('div', 'chips');
  secT.append(el('span', 'label', 'Type'));
  for (const [t, n] of [...types].sort((a, b) => collator.compare(a[0], b[0]))) {
    const c = btn(`${t} · ${n}`, () => { if (f.types.has(t)) f.types.delete(t); else f.types.add(t); drawFilter(); body.querySelector(`[data-t="${CSS.escape(t)}"]`)?.focus(); }, { cls: 'chip' });
    c.dataset.t = t;
    c.setAttribute('aria-pressed', String(f.types.has(t)));
    chips.append(c);
  }
  if (!types.size) chips.append(el('span', 'muted', 'No types here yet.'));
  secT.append(chips);
  const secD = el('div', 'fsec'), row = el('div', 'daterow');
  secD.append(el('span', 'label', 'Document date'));
  const date = (label, k) => {
    const l = el('label', '', label), i = el('input');
    i.type = 'date';
    i.value = f[k];
    i.onchange = () => { f[k] = i.value; drawFilter(); body.querySelector(`input[data-k="${k}"]`)?.focus(); };
    i.dataset.k = k;
    l.append(i);
    return l;
  };
  row.append(date('From', 'from'), date('To', 'to'));
  const now = new Date(), y = now.getFullYear(), m = now.getMonth();
  const iso = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  const presets = el('div', 'chips');
  for (const [label, a, b] of [['This month', new Date(y, m, 1), new Date(y, m + 1, 0)], ['Last month', new Date(y, m - 1, 1), new Date(y, m, 0)],
    ['This year', new Date(y, 0, 1), new Date(y, 11, 31)], ['Last year', new Date(y - 1, 0, 1), new Date(y - 1, 11, 31)]]) {
    const c = btn(label, () => { f.from = iso(a); f.to = iso(b); drawFilter(); }, { cls: 'chip' });
    c.setAttribute('aria-pressed', String(f.from === iso(a) && f.to === iso(b)));
    presets.append(c);
  }
  secD.append(row, presets);
  const secC = el('div', 'fsec');
  const check = (label, k, on = () => f[k], set = v => { f[k] = v; }) => {
    const l = el('label', 'ck'), c = el('input');
    c.type = 'checkbox';
    c.checked = !!on();
    c.onchange = () => { set(c.checked); drawFilter(); body.querySelector(`input[data-c="${k}"]`)?.focus(); };
    c.dataset.c = k;
    l.append(c, el('span', '', label));
    return l;
  };
  secC.append(check('Only with an amount', 'amount'), check('Has a due date', 'due'),
    check('Has a folder suggestion', 'suggested', () => f.suggested, v => { f.suggested = v ? (typeof f.suggested === 'string' ? f.suggested : true) : null; }));
  // Read by: to find what a model read, select it all and re-read it with another.
  const models = new Map();
  for (const d of docs) models.set(readBy(d), (models.get(readBy(d)) || 0) + 1);
  for (const m of f.models) if (!models.has(m)) models.set(m, 0);
  const secR = el('div', 'fsec'), mchips = el('div', 'chips');
  secR.append(el('span', 'label', 'Read by'));
  for (const [m, n] of [...models].sort((a, b) => collator.compare(a[0], b[0]))) {
    const c = btn(`${m || 'not recorded'} · ${n}`, () => { if (f.models.has(m)) f.models.delete(m); else f.models.add(m); drawFilter(); body.querySelector(`[data-m="${CSS.escape(m)}"]`)?.focus(); }, { cls: 'chip' });
    c.dataset.m = m;
    c.setAttribute('aria-pressed', String(f.models.has(m)));
    mchips.append(c);
  }
  if (!models.size) mchips.append(el('span', 'muted', 'No documents here yet.'));
  secR.append(mchips, check('Not fully analyzed (partly, not at all, or before analyses were recorded)', 'partial'));
  body.replaceChildren(secT, secD, secC, secR);
  const n = docs.filter(d => passes(d, f)).length;
  $('fgo').textContent = `Show ${plural(n, 'document')}`;
}
$('fclear').onclick = () => { fdraft = noFilters(); drawFilter(); };
$('fgo').onclick = () => {
  Object.assign(view.filters, fdraft);
  $('filterview').close();
  renderDocs();
  announce(`${plural(view.order.length, 'document')} shown`);
};

// The phone's Add sheet: the same options node, moved in while it is open.
$('addfab').onclick = () => {
  $('avopts').append($('addopts'));
  $('avcamera').hidden = !coarse.matches;
  syncInto();
  openDialog($('addview'));
  $('avchoose').focus();
};
$('addview').addEventListener('close', () => $('addbox').insertBefore($('addopts'), $('addhint')));
$('avchoose').onclick = () => { $('addview').close(); $('file').click(); };
$('avcamera').onclick = () => { $('addview').close(); $('camera').click(); };
$('avfolder').onclick = () => { $('addview').close(); newFolder(view.place.type === 'folder' ? view.place.id : null); };

// ---- the detail panel ----
const extra = new Map();      // id -> {sha, pages, copies}, from GET /api/documents/<id>
const textCache = new Map();  // id -> {sha, max, data}, from GET /api/documents/<id>/text
let panelFor = null, panelStale = false, previewNode = null, previewKey = '';
// A job finished with this document (a re-read, say): its text is read again, and an open preview
// is drawn again with the next render, though its file, and so its key, is the same.
const forgetText = id => { textCache.delete(id); extra.delete(id); if (panelFor === id) previewKey = ''; };
async function loadText(d, max = 200000) {
  const c = textCache.get(d.id);
  if (c && c.sha === d.sha256 && c.max >= max) return c.data;
  const data = await api(`/api/documents/${d.id}/text?max=${max}`);
  textCache.set(d.id, { sha: d.sha256, max, data });
  return data;
}
async function loadExtra(d) {
  const c = extra.get(d.id);
  if (c && c.sha === d.sha256) return;
  try {
    const r = await api(`/api/documents/${d.id}`);
    extra.set(d.id, { sha: d.sha256, pages: r.pages, copies: r.copies || [] });
    if (view.doc === d.id) renderPanel();
  } catch (err) { if (err.status === 404) lib.load(); }
}

function openDoc(id, { title = false } = {}) {
  if (!lib.docs.has(id)) return;
  const was = view.doc != null;
  view.doc = id;
  sel.anchor = id;
  const push = !was && !isL.matches;
  nav(hashFor(view.place, id), push ? 'push' : 'replace', push || history.state?.detail ? { detail: true } : {});
  renderPanel();
  markCurrent();
  syncSelection();
  if (title) $('docpanel').querySelector('.titlein')?.focus();
  else if (!isL.matches) $('docpanel').querySelector(isS.matches ? '.dphead .back' : '.dphead h2')?.focus();
}
function closeDoc() {
  if (view.doc == null) return;
  const id = view.doc;
  view.doc = null;
  if (history.state?.app && history.state.detail) history.back(); else nav(hashFor(view.place), 'replace');
  renderPanel();
  markCurrent();
  syncSelection();
  const item = $('dlist').querySelector(`[data-nav="d${id}"]`);
  if (item) { for (const x of navItems()) x.tabIndex = x === item ? 0 : -1; view.focusKey = 'd' + id; item.focus({ preventScroll: true }); }
}
function markCurrent() {
  for (const x of $('dlist').querySelectorAll('[aria-current]')) x.removeAttribute('aria-current');
  if (view.doc != null) $('dlist').querySelector(`[data-nav="d${view.doc}"]`)?.setAttribute('aria-current', 'true');
}
const panelTyping = () => { const a = document.activeElement; return $('docpanel').contains(a) && a.matches('input:not([type=checkbox]), textarea'); };
$('docpanel').addEventListener('focusout', () => setTimeout(() => { if (panelStale && !panelTyping()) { panelStale = false; renderPanel(); } }));
$('docpanel').addEventListener('focusin', () => $('docpanel').classList.toggle('typing', panelTyping()));
$('docpanel').addEventListener('focusout', () => setTimeout(() => $('docpanel').classList.toggle('typing', panelTyping())));

function renderPanel() {
  const p = $('docpanel'), d = view.mode === 'docs' && view.doc != null ? lib.docs.get(view.doc) : null;
  document.body.classList.toggle('panelopen', !!d);
  if (!d) { p.hidden = true; p.replaceChildren(); panelFor = null; previewNode = null; previewKey = ''; return; }
  p.hidden = false;
  if (panelFor === d.id && panelTyping()) { panelStale = true; return; }
  const focusField = p.contains(document.activeElement) ? document.activeElement.dataset.field : null;
  const scrollTop = panelFor === d.id ? p.querySelector('.dpbody')?.scrollTop || 0 : 0;
  panelFor = d.id;
  loadExtra(d);
  const x = extra.get(d.id), S = isS.matches;
  // Header: previous/next in the listing's order on wider screens; Back to the place on a phone.
  const head = el('div', 'dphead'), i = view.order.indexOf(d.id);
  const prev = btn('', () => openDoc(view.order[i - 1]), { cls: 'icon nav', aria: 'Previous document', title: 'Previous document', icon: 'chev' });
  prev.querySelector('svg').classList.add('flip');
  const next = btn('', () => openDoc(view.order[i + 1]), { cls: 'icon nav', aria: 'Next document', title: 'Next document', icon: 'chev' });
  prev.disabled = i <= 0;
  next.disabled = i < 0 || i >= view.order.length - 1;
  const h2 = el('h2', '', i >= 0 ? `${i + 1} of ${view.order.length}` : 'Details');
  h2.tabIndex = -1;
  h2.setAttribute('aria-label', `Details of ${docTitle(d)}${i >= 0 ? `, ${i + 1} of ${view.order.length}` : ''}`);
  const back = btn(`‹ ${placeName(view.place)}`, closeDoc, { cls: 'back', aria: `Back to ${placeName(view.place)}` });
  const more = btn('', e => openMenu({ anchor: e.currentTarget, title: docTitle(d), subtitle: fileSubtitle(d), items: docMenuItems(d.id).filter(it => it?.label !== 'Show details') }),
    { cls: 'icon', aria: 'More actions', title: 'More actions', icon: 'more' });
  const close = btn('', closeDoc, { cls: 'icon close', aria: 'Close the details', title: 'Close (Esc)', icon: 'close' });
  head.append(back, prev, next, h2, el('span', 'grow'), more, close);
  const body = el('div', 'dpbody');
  const job = el('div', 'dpjob');
  job.hidden = true;
  body.append(job);
  // Preview: kept between renders while the document's file is the same, so nothing reloads.
  const key = `${d.id}:${d.sha256}:${d.has_file}:${d.kind}:${x?.pages ?? ''}`;
  if (key !== previewKey || !previewNode) { previewNode = preview(d, x); previewKey = key; }
  body.append(previewNode);
  // Title, file and folder lines, actions.
  const tsec = el('div', 'dpsec');
  const title = el('input', 'titlein');
  title.value = docTitle(d);
  title.dataset.field = 'title';
  title.maxLength = 300;
  title.title = docTitle(d);  // in full, where a long one ends in "…"
  title.setAttribute('aria-label', 'Title');
  const terr = el('span', 'err'), saved = el('span', 'saved t');
  terr.setAttribute('role', 'alert');
  title.onkeydown = e => {
    if (e.key === 'Enter') { e.preventDefault(); title.blur(); }
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); title.value = docTitle(d); terr.textContent = ''; title.blur(); }
  };
  title.onchange = () => {
    const v = title.value.trim();
    if (!v) { terr.textContent = "A title can't be empty"; return; }
    terr.textContent = '';
    if (v !== docTitle(d)) patchDoc(d.id, { title: v }, saved);
  };
  const file = el('div', 'fileline');
  file.append(d.filename, ` · ${d.has_file ? fmtSize(d.file_size) : 'original not kept'}`);
  if (x?.pages) file.append(` · ${plural(x.pages, 'page')}`);
  if (d.has_file) file.append(' · ', Object.assign(el('a', 'open', 'Open ↗'), { href: `/api/documents/${d.id}/file`, target: '_blank', rel: 'noopener', title: 'Open the original in a new tab' }));
  const fl = el('div', 'folderline');
  fl.append(icon('folder'));
  if (isUnfiled(d)) fl.append(btn('Unfiled', () => openPlace({ type: 'unfiled' }), { cls: 'open' }));
  else {
    const f = lib.folders.get(d.folder_id), chain = [];
    for (let c = f, n = 0; c && n < 64; c = lib.folders.get(c.parent_id), n++) chain.unshift(c);
    chain.forEach((c, k) => { if (k) fl.append(el('span', 'sep', '›')); fl.append(btn(c.name, () => openPlace({ type: 'folder', id: c.id }), { cls: 'open', title: pathText(c.id) })); });
  }
  fl.append(btn('Move…', () => moveDialog([d.id]), { cls: 'mv' }));
  tsec.append(title, terr, saved, actions(d, 'top'), file, fl);
  if (d.filed_by === 'auto' && !isUnfiled(d)) {
    const a = el('div', 'autonote');
    a.append('Filed automatically · ', btn('Move back to Unfiled', () => moveDocs([d.id], null), { cls: 'open' }));
    tsec.append(a);
  }
  body.append(tsec);
  // The suggested folder, while it is Unfiled.
  if (d.suggested_folder && isUnfiled(d)) {
    const where = d.suggested_folder.split('/').join(' › '), exists = findPath(d.suggested_folder) != null, nested = d.suggested_folder.includes('/');
    const sec = el('div', 'dpsec'), card = el('div', 'finding'), p1 = el('div');
    p1.append('Suggested folder: ', el('b', '', where));
    const row = el('div', 'row');
    row.append(btn(exists ? `Move to ${where}` : nested ? `Create ${where} and move` : `Move to new folder '${where}'`,
      () => applySuggestion(d.id, d.suggested_folder), { cls: 'suggest' }),  // outlined: Ask about this is the panel's one primary
      btn('Dismiss', () => patchDoc(d.id, { suggestion: 'dismiss' })));
    card.append(p1, row);
    sec.append(card);
    body.append(sec);
  }
  body.append(detailsSection(d), aboutSection(d, x));
  p.replaceChildren(head, body, actions(d, 'dock'));
  syncPanelJob();
  body.scrollTop = scrollTop;
  if (focusField) p.querySelector(`[data-field="${focusField}"]`)?.focus({ preventScroll: true });
}

function actions(d, where) {
  const box = el('div', `dacts ${where}`), bar = where === 'dock';  // dock: a phone's bottom bar
  const del = btn('Delete', null, { cls: 'del', icon: 'trash', title: 'Delete this document' });
  del.onclick = () => {
    if (!del.classList.contains('sure')) {  // a second click within 3 s
      del.classList.add('sure');
      del.lastChild.textContent = 'Delete?';
      setTimeout(() => { if (del.isConnected) { del.classList.remove('sure'); del.lastChild.textContent = 'Delete'; } }, 3000);
      return;
    }
    deleteDocs([d.id]);
  };
  const reread = btn('Re-read…', () => openReread([d.id]), { icon: 'reread', title: d.has_file ? 'Read it again with a model of your choice' : "The original wasn't kept, so it can't be re-read" });
  reread.disabled = !d.has_file;
  if (bar) reread.lastChild.textContent = 'Re-read';
  const ask = btn(bar ? 'Ask' : 'Ask about this', () => askAbout({ docs: [d.id] }), { cls: bar ? '' : 'primary', icon: 'chat', title: 'Questions search only this document' });
  const move = btn(bar ? 'Move' : 'Move…', () => moveDialog([d.id]), { icon: 'move', title: 'Move it to another folder' });
  const items = [ask];
  if (d.has_file) {
    const open = Object.assign(el('a', 'btn'), { href: `/api/documents/${d.id}/file`, target: '_blank', rel: 'noopener', title: 'Open the original in a new tab' });
    open.append(icon('external'), el('span', '', bar ? 'Open' : 'Open ↗'));
    items.push(open);
  }
  items.push(move, reread, del);
  box.append(...items);
  return box;
}

function syncPanelJob() {
  const box = $('docpanel').querySelector('.dpjob');
  if (!box) return;
  const row = view.doc != null && rereadRows().get(view.doc);
  box.hidden = !row;
  if (!row) { box.replaceChildren(); return; }
  const v = row.view || {}, line = el('div', 'row');
  line.append(el('span', 'spin'), el('span', '', `Re-reading · ${v.stage || v.state || ''}`));
  const cancel = btn('Cancel', null, { title: 'Cancel this re-read' });
  cancel.onclick = () => {
    if (!cancel.classList.contains('sure')) { cancel.classList.add('sure'); cancel.textContent = 'Cancel?'; setTimeout(() => { if (cancel.isConnected) { cancel.classList.remove('sure'); cancel.textContent = 'Cancel'; } }, 3000); return; }
    jobAction(row, row.job.id, 'cancel');
  };
  line.append(cancel);
  const bar = el('div', 'bar' + (typeof v.bar === 'number' ? '' : ' wait')), i = el('i');
  if (typeof v.bar === 'number') i.style.width = `${Math.round(v.bar * 100)}%`;
  bar.append(i);
  box.replaceChildren(line, bar);
}

function preview(d, x) {
  const box = el('div', 'dpsec dpprev');
  const fail = retry => {
    const f = el('div', 'nofile');
    f.append(el('p', 'muted', "Couldn't load the preview."), btn('Try again', retry));
    return f;
  };
  const lazyText = (label, open = false) => {  // the text as read, loaded when opened
    const det = el('details'), sum = el('summary', '', label), tb = el('div', 'textbox');
    det.append(sum, tb);
    det.open = open;
    let loaded = false;
    const load = async () => {
      if (loaded) return;
      loaded = true;
      tb.textContent = 'Loading…';
      try { const t = await loadText(d); tb.textContent = t.text || '(no text)'; } catch { loaded = false; tb.replaceChildren(fail(load)); }
    };
    det.addEventListener('toggle', () => { if (det.open) load(); });
    if (open) load();
    return det;
  };
  if (!d.has_file) {
    const n = el('div', 'nofile');
    n.append(typedIcon(d, 'big'), el('p', 'muted', "The original wasn't kept for this document (it was added before originals were kept). Its text is still searchable and can be asked about."));
    box.append(n, lazyText('Text as read'));
    return box;
  }
  const fileUrl = `/api/documents/${d.id}/file`;
  if (d.kind === 'pdf' || d.kind === 'image') {
    const heic = /\.hei[cf]$/i.test(d.filename), src = d.kind === 'image' && !heic ? fileUrl : d.thumb && `${d.thumb}&size=large`;
    const a = Object.assign(el('a', 'pbox'), { href: d.kind === 'image' && heic && src ? src : fileUrl, target: '_blank', rel: 'noopener' });
    a.setAttribute('aria-label', d.kind === 'pdf' ? 'Open the PDF in a new tab' : 'Open the photo in a new tab');
    if (src && !thumbFailed.has(d.id + ':' + d.sha256 + ':large')) {
      const img = el('img', d.kind === 'pdf' ? 'pdf' : '');
      img.alt = d.kind === 'pdf' ? `First page of ${docTitle(d)}` : docTitle(d);
      img.decoding = 'async';
      img.src = src;
      img.onerror = () => { thumbFailed.add(d.id + ':' + d.sha256 + ':large'); img.replaceWith(typedIcon(d, 'big')); };
      a.append(img);
    } else a.append(typedIcon(d, 'big'));
    box.append(a);
    if (d.kind === 'pdf') {
      const cap = el('div', 'cap');
      cap.append(x?.pages ? `Page 1 of ${x.pages} · ` : 'Page 1 · ', Object.assign(el('a', 'open', 'Open the PDF ↗'), { href: fileUrl, target: '_blank', rel: 'noopener' }));
      box.append(cap);
    } else box.append(lazyText('Text read from the photo'));
    return box;
  }
  if (d.kind === 'markdown') {
    const seg = el('div', 'seg prow'), fmt = btn('Formatted', null), plain = btn('Plain text', null);
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', 'Show the Markdown');
    seg.append(fmt, plain);
    const area = el('div', 'mdbox'), tools = el('div', 'prow'), all = btn('Show all', null, { cls: 'open' }), note = el('span', 'muted small');
    tools.append(seg, el('span', 'grow'), all);
    let mode = store.get('mdView') === 'plain' ? 'plain' : 'formatted', data = null;
    const draw = () => {
      fmt.setAttribute('aria-pressed', String(mode === 'formatted'));
      plain.setAttribute('aria-pressed', String(mode === 'plain'));
      if (!data) return;
      if (mode === 'formatted') { const m = el('div', 'md'); m.append(mdRender(data.text)); area.replaceChildren(m); }
      else area.replaceChildren(el('pre', 'plain', data.text));
    };
    fmt.onclick = () => { mode = 'formatted'; store.set('mdView', ''); draw(); };
    plain.onclick = () => { mode = 'plain'; store.set('mdView', 'plain'); draw(); };
    all.onclick = () => { const on = area.classList.toggle('all'); all.textContent = on ? 'Show less' : 'Show all'; };
    area.textContent = 'Loading…';
    const load = async () => {
      try { data = await loadText(d); } catch { area.replaceChildren(fail(load)); return; }
      note.textContent = data.truncated ? 'Showing the first 200 KB. Open the file for the rest.' : '';
      draw();
    };
    draw();
    load();
    box.append(tools, area, note);
    return box;
  }
  // Text and Word: the start, then more on request.
  const tb = el('div', 'textbox fade'), more = btn('Show more', null, { cls: 'open' });
  more.hidden = true;
  tb.textContent = 'Loading…';
  let cut = false;  // the server sent only the start
  const load = async (max = 4000) => {
    let t;
    try { t = await loadText(d, max); } catch { tb.replaceChildren(fail(() => load(max))); return; }
    tb.textContent = t.text || '(no text)';
    cut = t.truncated;
    tb.classList.toggle('all', max > 4000);
    // Taller than the box counts too: 900 characters can be more than its 18 lines. (The panel is
    // in the page by now: this runs after an await.)
    const hidden = !tb.classList.contains('all') && (cut || tb.scrollHeight > tb.clientHeight + 1);
    tb.classList.toggle('fade', hidden);
    more.hidden = !hidden;
    if (max > 4000 && t.truncated) box.append(el('span', 'muted small', 'Showing the first 200,000 characters. Open the file for the rest.'));
  };
  // More from the server when it cut the text; otherwise the rest is here already.
  more.onclick = () => { if (cut) load(200000); else { tb.classList.add('all'); tb.classList.remove('fade'); more.hidden = true; } };
  load();
  box.append(tb, more);
  return box;
}

// Key details: each an input that saves on change; a value the user set is kept by re-reads.
function detailsSection(d) {
  const sec = el('div', 'dpsec'), dl = el('dl', 'kv'), edited = new Set(d.user_edited || []);
  sec.append(el('span', 'label', 'Details'));
  // A row: its label, and the input it labels (the amount's row holds two: total and currency).
  const field = (label, key, content, err, input = content) => {
    const dt = el('dt', '', label);
    if (edited.has(key) || (key === 'total_amount' && edited.has('currency'))) {
      const t = el('span', 'edited', 'edited');
      t.title = 'You changed this; re-reading keeps your value';
      dt.append(t);
    }
    const dd = el('dd');
    input.dataset.field = key;
    dt.id = 'kvl-' + key;
    input.setAttribute('aria-labelledby', dt.id);
    dd.append(content);
    if (err) dd.append(err);
    dl.append(dt, dd);
  };
  const saved = el('span', 'saved k');
  const text = (key, value, opts = {}) => {
    const i = el('input', 'kvin' + (opts.cls ? ' ' + opts.cls : ''));
    i.type = opts.type || 'text';
    i.value = value ?? '';
    if (opts.max) i.maxLength = opts.max;
    if (opts.list) i.setAttribute('list', opts.list);
    if (opts.mode) i.inputMode = opts.mode;
    if (opts.placeholder) i.placeholder = opts.placeholder;
    i.onkeydown = e => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); i.value = value ?? ''; i.classList.remove('bad'); i.blur(); } if (e.key === 'Enter') i.blur(); };
    return i;
  };
  const err = () => { const e = el('span', 'err'); e.setAttribute('role', 'alert'); return e; };
  const type = text('doc_type', d.doc_type, { max: 60, list: 'typelist', placeholder: 'e.g. receipt' });
  type.onchange = () => { if (type.value.trim() !== (d.doc_type || '')) patchDoc(d.id, { doc_type: type.value.trim() }, saved); };
  field('Type', 'doc_type', type);
  const dd = text('document_date', d.document_date, { type: 'date' });
  dd.onchange = () => patchDoc(d.id, { document_date: dd.value || null }, saved);
  field('Document date', 'document_date', dd);
  const showMoney = isBill(d) || d.vendor || d.total_amount != null || d.due_date || moneyOpen.has(d.id);
  if (showMoney) {
    const vendor = text('vendor', d.vendor, { max: 200, placeholder: 'Add vendor' });
    vendor.onchange = () => patchDoc(d.id, { vendor: vendor.value.trim() || null }, saved);
    field('Vendor', 'vendor', vendor);
    const amtErr = err(), row = el('div', 'amtrow');
    // 42.1 shows as 42.10; more decimals than two, as read, stay.
    const shown = n => n == null ? '' : Number.isInteger(Math.round(n * 100 * 1e6) / 1e6) ? n.toFixed(2) : String(n);
    const amt = text('total_amount', shown(d.total_amount), { mode: 'decimal', placeholder: 'Add amount' });
    const cur = text('currency', d.currency, { max: 3, list: 'curlist', cls: 'cur', placeholder: 'GBP' });
    cur.setAttribute('aria-label', 'Currency');
    amt.onchange = () => {
      const v = parseAmount(amt.value);
      if (v === null) { amt.classList.remove('bad'); amtErr.textContent = ''; patchDoc(d.id, { total_amount: null }, saved); return; }
      if (typeof v === 'string') { amt.classList.add('bad'); amtErr.textContent = v; return; }
      amt.classList.remove('bad');
      amtErr.textContent = '';
      patchDoc(d.id, { total_amount: v }, saved);
    };
    cur.oninput = () => { const p = cur.selectionStart; cur.value = cur.value.toUpperCase(); cur.setSelectionRange(p, p); };
    cur.onchange = () => {
      const v = cur.value.trim().toUpperCase();
      if (v && !/^[A-Z]{3}$/.test(v)) { cur.classList.add('bad'); amtErr.textContent = 'A three-letter currency code like GBP'; return; }
      cur.classList.remove('bad');
      amtErr.textContent = '';
      patchDoc(d.id, { currency: v || null }, saved);
    };
    row.append(amt, cur);
    field('Total', 'total_amount', row, amtErr, amt);
    cur.dataset.field = 'currency';
    const due = text('due_date', d.due_date, { type: 'date' });
    due.onchange = () => patchDoc(d.id, { due_date: due.value || null }, saved);
    field('Due date', 'due_date', due);
    const info = d.due_date && dueInfo(d.due_date);
    if (info) dl.lastChild.append(el('span', 'due' + (info.soon ? ' soon' : ''), info.text));
  }
  sec.append(dl);
  if (!showMoney) sec.append(btn('+ Add vendor, amount or due date', () => { moneyOpen.add(d.id); renderPanel(); $('docpanel').querySelector('[data-field="vendor"]')?.focus(); }, { cls: 'open kvmore' }));
  sec.append(saved);
  // Summary and keywords: a keyword searches everywhere.
  const out = [sec];
  if (d.summary) { const s = el('div', 'dpsec'); s.append(el('span', 'label', 'Summary'), el('p', 'summary', d.summary)); out.push(s); }
  const kws = (d.keywords || '').split(',').map(s => s.trim()).filter(Boolean);
  if (kws.length) {
    const s = el('div', 'dpsec'), chips = el('div', 'chips');
    for (const k of kws) chips.append(btn(k, () => { $('dq').value = k; view.q = k; view.qScope = 'all'; view.relevance = true; if (!isL.matches) closeDoc(); renderDocs(); serverSearch(); }, { cls: 'chip', title: `Search everywhere for ${k}` }));
    s.append(el('span', 'label', 'Keywords'), chips);
    out.push(s);
  }
  const frag2 = document.createDocumentFragment();
  frag2.append(...out);
  return frag2;
}
const moneyOpen = new Set();
// A total as typed: a number, null for none, or what is wrong with it (a string). A dot is the
// decimal point, and so is a comma when there is no dot ("42,10"). Commas grouping thousands are
// dropped where they can't be decimals: before a dot ("1,234.56", "£2,000.00") or more than one
// ("12,000,000"). A lone comma before three digits ("1,500") could be either, so it is asked
// about, as the server's reading of an analysis leaves such an amount out (jobs._details).
function parseAmount(typed) {
  let raw = String(typed).replace(/[\s£$€¥]/g, '');
  if (!raw) return null;
  if (/^-?\d{1,3}(,\d{3})+(\.\d{1,4})?$/.test(raw) && (raw.includes('.') || raw.split(',').length > 2)) raw = raw.replace(/,/g, '');
  if (/^-?\d{1,12},\d{3}$/.test(raw)) {
    const whole = raw.replace(',', ''), part = parseFloat(raw.replace(',', '.'));
    return `Is that ${whole} or ${part.toFixed(2)}? Write ${whole}, or use a dot for decimals`;
  }
  if (!/^-?\d{1,12}([.,]\d{1,4})?$/.test(raw)) return 'An amount like 1234.56';
  return parseFloat(raw.replace(',', '.'));
}
function aboutSection(d, x) {
  const s = el('div', 'dpsec'), dl = el('dl', 'about');
  const row = (k, v) => { const dd = el('dd'); dd.append(v); dl.append(el('dt', '', k), dd); };
  row('Read by', d.model || '—');
  row('Analysis', d.analysis || '—');
  row('Characters', fmtInt(d.char_count));
  row('Added', fmtAdded(d.uploaded_at, true));
  if (d.unit_fixes) row('Unit fixes', unitBadge(d.id, d.unit_fixes));
  const copies = (x?.copies || []).filter(c => lib.docs.has(c.id));
  if (copies.length) {
    const f = document.createDocumentFragment();
    copies.forEach((c, k) => {
      if (k) f.append(', ');
      f.append(btn(`Also in ${pathText(c.folder_id)}`, () => showDocument(c.id), { cls: 'open' }));
    });
    row('Copies', f);
  }
  s.append(el('span', 'label', 'About'), dl);
  return s;
}
async function patchDoc(id, body, saved = null) {
  let d;
  try { d = await send(`/api/documents/${id}`, body, 'PATCH'); } catch (err) {
    toast(err.message, { error: true });
    if (err.status === 404) lib.load();
    return null;
  }
  const old = lib.docs.get(id);
  lib.docs.set(id, { ...old, ...d });
  if (d.created?.length || 'folder_id' in body || body.suggestion) await lib.load(); else { rebuild(); render(); }
  if (saved) {  // the panel may have been drawn again meanwhile: its own "Saved" for that part
    const target = $('docpanel').querySelector('title' in body ? '.saved.t' : '.saved.k');
    if (target) { target.textContent = 'Saved'; setTimeout(() => { if (target.isConnected) target.textContent = ''; }, 1500); }
  }
  if (body.suggestion === 'dismiss') toast('Suggestion dismissed');
  return d;
}

// ---- drag and drop ----
// Inside the page (fine pointers): documents and folders onto folders. From the computer: files
// onto a folder, or wherever the page is. dragover decides; only a valid target highlights.
let drag = null, dropEl = null, hoverKey = '', hoverAt = 0;
function dragDest(key) {  // a drop target's folder: null for Unfiled / the top level, undefined for none
  if (key === 'root' || key === 'unfiled') return null;
  if (key && key[0] === 'f') return +key.slice(1);
  return undefined;
}
function validDrop(dr, key) {
  if (!dr || !key) return false;
  if (dr.docs) {
    if (key === 'all') return false;
    const fid = dragDest(key);
    if (fid === undefined || (fid != null && !lib.folders.has(fid))) return false;
    return dr.docs.some(id => { const d = lib.docs.get(id); return d && (isUnfiled(d) ? null : d.folder_id) !== fid; });
  }
  if (dr.folder != null) {
    if (key === 'all' || key === 'unfiled') return false;
    const fid = dragDest(key), f = lib.folders.get(dr.folder);
    if (!f || fid === undefined) return false;
    if (fid != null && subtree(dr.folder).has(fid)) return false;
    return (f.parent_id ?? null) !== fid;
  }
  return false;
}
function highlight(t) {
  if (dropEl === t) return;
  dropEl?.classList.remove('dropon');
  dropEl = t;
  t?.classList.add('dropon');
}
function startDrag(e, label, els) {
  if (!fine.matches) { e.preventDefault(); return; }
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData('application/x-docchat', JSON.stringify(drag));
  e.dataTransfer.setData('text/plain', label);
  $('dragpill').textContent = label;
  e.dataTransfer.setDragImage($('dragpill'), 14, 14);
  setTimeout(() => els.forEach(x => x.classList.add('dragging')));
}
$('dlist').addEventListener('dragstart', e => {
  const item = e.target.closest?.('[data-nav]');
  if (!item) return;
  const key = item.dataset.nav;
  if (key[0] === 'g' || key === 'pall') { e.preventDefault(); return; }
  if (key[0] === 'f') {
    const f = lib.folders.get(+key.slice(1));
    drag = { folder: f.id };
    startDrag(e, f.name, [item]);
  } else {
    const id = +key.slice(1), ids = sel.ids.has(id) ? [...sel.ids] : [id];
    drag = { docs: ids };
    startDrag(e, ids.length === 1 ? docTitle(lib.docs.get(id)) : plural(ids.length, 'document'),
      ids.map(i => $('dlist').querySelector(`[data-nav="d${i}"]`)).filter(Boolean));
  }
});
$('tree').addEventListener('dragstart', e => {
  const li = e.target.closest?.('.tnode');
  if (!li || li.dataset.key[0] !== 'f') { e.preventDefault(); return; }
  const f = lib.folders.get(+li.dataset.key.slice(1));
  drag = { folder: f.id };
  startDrag(e, f.name, [li.querySelector('.trow')]);
});
function endDrag() {
  drag = null;
  highlight(null);
  hideDropzone();
  for (const x of document.querySelectorAll('.dragging')) x.classList.remove('dragging');
}
document.addEventListener('dragend', endDrag);
const isFiles = e => !drag && [...(e.dataTransfer?.types || [])].includes('Files');
function autoScroll(e) {
  for (const box of [$('tree'), $('dmain'), $('pktree')]) {
    if (!box.offsetParent) continue;
    const r = box.getBoundingClientRect();
    if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) continue;
    if (e.clientY < r.top + 40) box.scrollTop -= 12;
    else if (e.clientY > r.bottom - 40) box.scrollTop += 12;
  }
}
function hoverExpand(t) {  // a collapsed tree folder held over for 700 ms opens
  const li = t?.closest('.tnode');
  const key = li?.getAttribute('aria-expanded') === 'false' ? li.dataset.key : '';
  if (key !== hoverKey) { hoverKey = key; hoverAt = Date.now(); return; }
  if (key && Date.now() - hoverAt > 700) { hoverKey = ''; toggleTree(+key.slice(1), true); }
}
function fileDest(t) {  // where dropped files go: a target's folder, else where the page is
  const key = t?.dataset.drop;
  if (key) { const fid = dragDest(key); return { folder_id: fid ?? null, auto_file: autoFileOn() }; }
  return uploadTarget();
}
function showDropzone(target) {
  const z = $('dropzone'), host = view.mode === 'docs' ? $('docsview') : document.querySelector('main');
  if (z.parentElement !== host) host.append(z);
  const f = target.folder_id != null && lib.folders.get(target.folder_id);
  const t = $('droptext');
  t.replaceChildren('Drop to add to ', el('strong', '', f ? f.name : 'Unfiled'));
  if (!f && target.auto_file) t.append(' · filed automatically when it matches a folder');
  $('dropnote').textContent = `Read with ${readerText()}`;
  z.hidden = false;
  $('addbox').classList.add('over');
}
function hideDropzone() { $('dropzone').hidden = true; $('addbox').classList.remove('over'); }
document.addEventListener('dragover', e => {
  const files = isFiles(e);
  if (!drag && !files) return;
  const t = e.target.closest?.('[data-drop]') || null;
  autoScroll(e);
  hoverExpand(t);
  if (files) {
    e.preventDefault();  // a dropped file never opens in the browser
    e.dataTransfer.dropEffect = 'copy';
    showDropzone(fileDest(t));
    highlight(t);
    return;
  }
  if (t && validDrop(drag, t.dataset.drop)) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; highlight(t); }
  else highlight(null);
});
document.addEventListener('dragleave', e => { if (!e.relatedTarget) { highlight(null); hideDropzone(); } });
document.addEventListener('drop', e => {
  const t = e.target.closest?.('[data-drop]') || null;
  if (drag) {
    const dr = drag, key = t?.dataset.drop;
    if (key && validDrop(dr, key)) {
      e.preventDefault();
      const fid = dragDest(key);
      if (dr.docs) moveDocs(dr.docs, fid); else moveFolder(dr.folder, fid);
    }
    endDrag();
    return;
  }
  if (![...(e.dataTransfer?.types || [])].includes('Files')) return;
  e.preventDefault();
  const target = fileDest(t);
  highlight(null);
  hideDropzone();
  const files = [];
  const items = [...(e.dataTransfer.items || [])];
  if (items.length && items[0].webkitGetAsEntry) {
    for (const it of items) {
      if (it.kind !== 'file') continue;
      const entry = it.webkitGetAsEntry(), f = it.getAsFile();
      if (entry?.isDirectory) failedRow(entry.name, "Folders can't be uploaded: open it and drop the files inside");
      else if (f) files.push(f);
    }
  } else files.push(...e.dataTransfer.files);
  if (files.length) uploadAll(files, target);
});

// ---- what questions search: folders and documents, chosen in a tree ----
let sv = null;
function openScope() {
  sv = { docs: new Set(scope.docs), folders: new Set(scope.folders), open: new Set() };
  for (const id of [...sv.folders, ...[...sv.docs].map(i => lib.docs.get(i)?.folder_id)]) {
    for (let f = lib.folders.get(lib.folders.get(id)?.parent_id); f; f = lib.folders.get(f.parent_id)) sv.open.add(f.id);
  }
  $('scopeq').value = '';
  $('embedstatus').textContent = '';
  drawScope();
  openDialog($('scopeview'));
  showEmbedStatus();
  if (!lib.loaded || Date.now() - lib.loadedAt > 30000) lib.load().then(() => { if (sv) drawScope(); });
  $('scopeq').focus();
}
async function showEmbedStatus() {  // search by meaning: how far the vectors are
  let s;
  try { s = await api('/api/search/status'); } catch { $('embedstatus').textContent = ''; return; }
  $('embedstatus').textContent = s.error
    ? `Search by meaning (${s.model}) is paused: ${s.error}. Questions are searched by keywords meanwhile, and it tries again shortly.`
    : s.embedded < s.chunks
      ? `Questions are searched by keywords and by meaning (${s.model}). ${fmtInt(s.embedded)} of ${fmtInt(s.chunks)} passages are indexed by meaning so far; the rest follow in the background.`
      : `Questions are searched by keywords and by meaning (${s.model}).`;
}
// The ticked folder a folder or document is inside, if any (it is then included already).
function tickedAbove(fid) {
  for (let f = lib.folders.get(fid), n = 0; f && n < 64; f = lib.folders.get(f.parent_id), n++) if (sv.folders.has(f.id)) return f;
  return null;
}
function svCounts() {
  const top = [...sv.folders].filter(id => lib.folders.has(id) && !tickedAbove(lib.folders.get(id).parent_id));
  const inFolders = new Set();
  for (const id of top) for (const d of subtreeDocs(id)) inFolders.add(d);
  const docs = [...sv.docs].filter(id => lib.docs.has(id) && !inFolders.has(id));
  return { top, docs, n: inFolders.size };
}
function drawScope() {
  if (!sv) return;
  const q = norm($('scopeq').value.trim()), list = $('scopelist'), rows = [];
  const folderRowS = (f, level, path = '') => {
    const li = el('li', 'srow'), above = tickedAbove(f.parent_id), on = sv.folders.has(f.id) || !!above;
    li.style.setProperty('--lvl', String(Math.min(level, 8)));
    if (on) li.classList.add('on');
    if (above) li.classList.add('in');
    const has = lib.kids.get(f.id).length + lib.docsIn.get(f.id).length > 0;
    const chev = el('button', 'chev' + (has && !q ? '' : ' none'));
    chev.type = 'button';
    chev.append(icon('chev'));
    chev.setAttribute('aria-expanded', String(sv.open.has(f.id)));
    chev.setAttribute('aria-label', `${sv.open.has(f.id) ? 'Collapse' : 'Expand'} ${f.name}`);
    chev.onclick = () => { if (sv.open.has(f.id)) sv.open.delete(f.id); else sv.open.add(f.id); drawScope(); list.querySelector(`[data-sf="${f.id}"] .chev`)?.focus(); };
    if (!has || q) chev.tabIndex = -1;
    const lab = el('label'), box = el('input'), txt = el('span', 'stext');
    box.type = 'checkbox';
    box.checked = on;
    box.disabled = !!above;
    box.onchange = () => { if (box.checked) sv.folders.add(f.id); else sv.folders.delete(f.id); drawScope(); list.querySelector(`[data-sf="${f.id}"] input`)?.focus(); };
    txt.append(el('b', '', f.name), el('small', '', [path, plural(f.total, 'document'), above ? `in ${above.name}` : ''].filter(Boolean).join(' · ')));
    lab.append(box, icon('folder'), txt);
    li.dataset.sf = f.id;
    li.append(chev, lab);
    rows.push(li);
  };
  const docRowS = (d, level, path = '') => {
    const li = el('li', 'srow'), above = isUnfiled(d) ? null : tickedAbove(d.folder_id), on = sv.docs.has(d.id) || !!above;
    li.style.setProperty('--lvl', String(Math.min(level, 8)));
    if (on) li.classList.add('on');
    if (above) li.classList.add('in');
    const chev = el('span', 'chev none'), lab = el('label'), box = el('input'), txt = el('span', 'stext');
    box.type = 'checkbox';
    box.checked = on;
    box.disabled = !!above;
    box.onchange = () => { if (box.checked) sv.docs.add(d.id); else sv.docs.delete(d.id); drawScope(); list.querySelector(`[data-sd="${d.id}"] input`)?.focus(); };
    txt.append(el('b', '', docTitle(d)), el('small', '', [path, d.filename, d.doc_type, above ? `in ${above.name}` : ''].filter(Boolean).join(' · ')));
    lab.append(box, icon(d.kind === 'image' ? 'image' : 'page'), txt);  // its name in line with the folders' names
    li.dataset.sd = d.id;
    li.append(chev, lab);
    rows.push(li);
  };
  if (!lib.loaded) rows.push(el('li', 'muted', lib.error || 'Loading…'));
  else if (q) {  // a flat list of what matches, with where each is
    const words = q.split(/\s+/);
    const hitF = [...lib.folders.values()].filter(f => words.every(w => norm(f.name).includes(w)));
    const hitD = [...lib.docs.values()].filter(d => words.every(w => norm([d.title, d.filename, d.doc_type, d.keywords].join(' ')).includes(w)));
    for (const f of hitF) folderRowS(f, 0, f.names.slice(0, -1).join(' › ') || 'Library');
    for (const d of hitD) docRowS(d, 0, pathText(isUnfiled(d) ? null : d.folder_id));
    if (!rows.length) rows.push(el('li', 'muted', 'Nothing matches that filter.'));
  } else {
    const add = (key, level) => {
      for (const id of lib.kids.get(key) || []) {
        const f = lib.folders.get(id);
        folderRowS(f, level);
        if (sv.open.has(id)) { add(id, level + 1); for (const d of lib.docsIn.get(id)) docRowS(lib.docs.get(d), level + 1); }
      }
    };
    add('root', 0);
    const un = unfiledIds();
    if (un.length) {
      rows.push(Object.assign(el('li', 'label sglabel', lib.folders.size ? `Unfiled · ${un.length}` : `Documents · ${un.length}`), {}));
      for (const id of un) docRowS(lib.docs.get(id), 0);
    }
    if (!lib.docs.size && !lib.folders.size) rows.push(el('li', 'muted', 'Your library is empty.'));
  }
  list.replaceChildren(...rows);
  $('scopeall').hidden = !q;
  const { top, docs, n } = svCounts();
  const fpart = top.length ? `${plural(top.length, 'folder')} (${plural(n, 'document')})` : '', dpart = docs.length ? plural(docs.length, 'document') : '';
  $('scopecount').textContent = fpart || dpart ? [fpart, dpart].filter(Boolean).join(' and ') : 'Nothing ticked: questions search every document';
  const use = $('scopeuse');
  use.disabled = !top.length && !docs.length;
  use.textContent = use.disabled ? 'Tick folders or documents' : top.length === 1 && !docs.length ? `Ask ${lib.folders.get(top[0]).name}`
    : `Ask ${[top.length ? plural(top.length, 'folder') : '', docs.length ? (docs.length === 1 && !top.length ? 'this one' : plural(docs.length, 'document')) : ''].filter(Boolean).join(' and ')}`;
}
$('scopebtn').onclick = openScope;
$('scopeq').oninput = drawScope;
$('scopeall').onclick = () => {
  const q = norm($('scopeq').value.trim()), words = q.split(/\s+/);
  for (const f of lib.folders.values()) if (words.every(w => norm(f.name).includes(w))) sv.folders.add(f.id);
  for (const d of lib.docs.values()) if (words.every(w => norm([d.title, d.filename, d.doc_type, d.keywords].join(' ')).includes(w))) sv.docs.add(d.id);
  drawScope();
};
$('scopenone').onclick = () => { sv.docs.clear(); sv.folders.clear(); drawScope(); };
$('scopeevery').onclick = () => { setScope({}); $('scopeview').close(); $('question').focus(); };
$('scopeuse').onclick = () => { const { top, docs } = svCounts(); sv = null; askAbout({ docs, folders: top }); };
$('svclose').onclick = () => $('scopeview').close();
$('scopeview').addEventListener('close', () => { sv = null; });

// ---- start (from app.js's start) ----
function docsStart() {
  placeUploads();
  const saved = parseHash(store.get('docsAt'));
  if (saved) view.place = saved.place;
  history.replaceState({ app: true, n: 0 }, '', location.href);
  route();
  lib.load();
  // Back after a while (another tab may have changed things): a fresh listing.
  const fresh = () => { if (!document.hidden && lib.loaded && Date.now() - lib.loadedAt > 30000) lib.load(); };
  addEventListener('focus', fresh);
  document.addEventListener('visibilitychange', fresh);
  // The sticky table header sits under the toolbar, whatever height the toolbar wraps to.
  const ro = new ResizeObserver(() => {
    const h = $('dtools').hidden ? $('dselbar').offsetHeight : $('dtools').offsetHeight;
    if (h) $('dmain').style.setProperty('--toolsh', `${h}px`);
  });
  ro.observe($('dtools'));
  ro.observe($('dselbar'));
  // Once the toolbar sticks, a rule under it separates it from what scrolls beneath.
  $('dmain').addEventListener('scroll', () => $('dtools').classList.toggle('stuck', $('dmain').scrollTop > $('dhead').offsetHeight - 1), { passive: true });
}
// A phone shows no queue beside the listing: files just added open the Uploads box, in view.
function revealUploads() {
  if (!isS.matches || view.mode !== 'docs') return;
  $('uploads').open = true;
  if (!$('uploads').hidden) $('uploads').scrollIntoView({ block: 'nearest' });
}
