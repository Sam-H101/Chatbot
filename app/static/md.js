/* Markdown preview renderer for kept .md/.markdown files: mdRender(src, opts) -> DocumentFragment.
   Everything is built with createElement, textContent and a few whitelisted attributes: raw HTML
   and entities stay literal text, links open only http(s) and mailto, and images never become
   <img> (no request leaves the page until a link is clicked). A linear line scanner and an inline
   tokenizer with bounded look-ahead, so a hostile file can't make it hang. No globals but mdRender. */
function mdRender(src, opts) {
  'use strict';
  opts = opts || {};
  const LIMIT = opts.limit || 200000, MAXDEPTH = 6;
  let text = String(src == null ? '' : src);
  const cut = text.length > LIMIT;
  if (cut) text = text.slice(0, LIMIT);
  text = text.replace(/^﻿/, '').replace(/\r\n?/g, '\n');  // a BOM (Windows editors) would hide a first heading
  const SVG = 'http://www.w3.org/2000/svg';
  const PUNCT = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');
  const frag = document.createDocumentFragment();
  let pid = 0;

  function E(tag, cls, txt) {
    const e = document.createElement(tag);
    if (cls) e.setAttribute('class', cls);
    if (txt != null) e.textContent = txt;
    return e;
  }

  // Leading tabs count to the next multiple of 4 columns.
  function expandLead(line) {
    if (line.indexOf('\t') < 0) return line;
    let col = 0, i = 0;
    for (; i < line.length; i++) {
      if (line[i] === ' ') col++;
      else if (line[i] === '\t') col += 4 - (col % 4);
      else break;
    }
    return ' '.repeat(col) + line.slice(i);
  }
  function indentOf(l) { let i = 0; while (l[i] === ' ') i++; return i; }
  function blank(l) { return l.trim() === ''; }
  function dedent(l, n) { let i = 0; while (i < n && l[i] === ' ') i++; return l.slice(i); }

  // ---- links ----
  function safeUrl(raw) {
    let u;
    try { u = new URL(String(raw).trim()); } catch (e) { return null; }
    return u.protocol === 'http:' || u.protocol === 'https:' || u.protocol === 'mailto:' ? u.href : null;
  }
  function anchor(href) {
    const a = E('a');
    a.setAttribute('href', href);
    a.setAttribute('target', '_blank');
    a.setAttribute('rel', 'noopener noreferrer nofollow');
    a.setAttribute('referrerpolicy', 'no-referrer');
    a.setAttribute('title', href);
    return a;
  }
  function unescapeMd(s) {
    let out = '';
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '\\' && i + 1 < s.length && PUNCT.has(s[i + 1])) { out += s[i + 1]; i++; } else out += s[i];
    }
    return out;
  }
  function linkNode(label, url) {
    const safe = safeUrl(url);
    const box = safe ? anchor(safe) : E('span');
    inline(label, box, true);
    return box;
  }
  function imageChip(alt, url, inLink) {
    const safe = safeUrl(url), web = !!safe && /^https?:/i.test(safe);
    const chip = web && !inLink ? anchor(safe) : E('span');
    chip.setAttribute('class', 'mdimg');
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('class', 'ic');
    const use = document.createElementNS(SVG, 'use');
    use.setAttribute('href', '#i-image');
    svg.append(use);
    chip.append(svg, E('span', null, 'Image: ' + (unescapeMd(alt).trim() || 'no description')));
    if (web) chip.append(E('small', null, new URL(safe).host));
    return chip;
  }

  // ---- inline: a tokenizer, then emphasis matched with a bounded stack ----
  function bracketPairs(s) {  // '[' index -> its matching ']' (one pass, so no rescanning)
    const close = new Map(), open = [];
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (c === '\\') { i++; continue; }
      if (c === '[') open.push(i);
      else if (c === ']' && open.length) close.set(open.pop(), i);
    }
    return close;
  }
  function linkAt(s, i, pairs) {  // s[i] === '[': [text](url "title")
    const j = pairs.get(i);
    if (j == null || s[j + 1] !== '(') return null;
    let k = j + 2;
    while (s[k] === ' ' || s[k] === '\n') k++;
    let url;
    if (s[k] === '<') {
      const e = s.indexOf('>', k);
      if (e < 0 || e - k > 2048 || s.slice(k, e).indexOf('\n') >= 0) return null;
      url = s.slice(k + 1, e);
      k = e + 1;
    } else {
      const st = k, lim = Math.min(s.length, k + 2048);
      let depth = 0;
      for (; k < lim; k++) {
        const c = s[k];
        if (c === '\\') { k++; continue; }
        if (c === ' ' || c === '\n') break;
        if (c === '(') depth++;
        else if (c === ')') { if (!depth) break; depth--; }
      }
      url = s.slice(st, k);
    }
    while (s[k] === ' ' || s[k] === '\n') k++;
    if (s[k] === '"' || s[k] === "'" || s[k] === '(') {
      const q = s[k] === '(' ? ')' : s[k], e = s.indexOf(q, k + 1);
      if (e < 0 || e - k > 1024) return null;
      k = e + 1;
      while (s[k] === ' ' || s[k] === '\n') k++;
    }
    if (s[k] !== ')') return null;
    return { label: s.slice(i + 1, j), url: unescapeMd(url), end: k + 1 };
  }
  function autolinkAt(s, i) {  // <https://…> and <name@example.com>
    const e = s.indexOf('>', i);
    if (e < 0 || e - i > 2048) return null;
    const body = s.slice(i + 1, e);
    if (!body || /[\s<]/.test(body)) return null;
    if (/^[A-Za-z][A-Za-z0-9+.-]{1,31}:/.test(body)) return { href: body, label: body, end: e + 1 };
    const at = body.indexOf('@');
    if (at > 0 && body.indexOf('@', at + 1) < 0 && body.indexOf('.', at) > at + 1 && !body.endsWith('.'))
      return { href: 'mailto:' + body, label: body, end: e + 1 };
    return null;
  }
  function bareUrlAt(s, i) {
    const head = s.slice(i, i + 8).toLowerCase();
    const len = head.startsWith('https://') ? 8 : head.startsWith('http://') ? 7 : 0;
    if (!len) return null;
    let k = i + len;
    const lim = Math.min(s.length, i + 2048);
    while (k < lim && !/[\s<]/.test(s[k])) k++;
    while (k > i + len && '.,;:!?)*_~\'"'.indexOf(s[k - 1]) >= 0) k--;
    if (k <= i + len) return null;
    return { href: s.slice(i, k), end: k };
  }

  function inline(s, parent, inLink) {
    const toks = [];
    const pairs = bracketPairs(s);
    const failed = {};  // code span lengths with no closing run from that position on
    let buf = '';
    const flush = () => { if (buf) { toks.push({ t: 'text', v: buf }); buf = ''; } };
    const node = n => { flush(); toks.push({ t: 'node', n }); };
    const n = s.length;
    let i = 0;
    while (i < n) {
      const c = s[i];
      if (c === '\\') {
        const d = s[i + 1];
        if (d === '\n') { node(E('br')); i += 2; while (s[i] === ' ') i++; continue; }
        if (d !== undefined && PUNCT.has(d)) { buf += d; i += 2; continue; }
        buf += c; i++; continue;
      }
      if (c === '\n') {  // two or more trailing spaces make a hard break, else a soft one
        let k = buf.length;
        while (k > 0 && buf[k - 1] === ' ') k--;
        const spaces = buf.length - k;
        buf = buf.slice(0, k);
        if (spaces >= 2) node(E('br')); else buf += '\n';
        i++;
        while (s[i] === ' ') i++;
        continue;
      }
      if (c === '`') {
        let j = i;
        while (s[j] === '`') j++;
        const len = j - i;
        let close = -1;
        if (!(failed[len] <= j)) {
          const run = '`'.repeat(len);
          for (let p = j; ;) {
            const q = s.indexOf(run, p);
            if (q < 0) { failed[len] = j; break; }
            let e = q;
            while (s[e] === '`') e++;
            if (e - q === len) { close = q; break; }
            p = e;
          }
        }
        if (close >= 0) {
          let code = s.slice(j, close).replace(/\n/g, ' ');
          if (code.length > 2 && code[0] === ' ' && code[code.length - 1] === ' ' && code.trim()) code = code.slice(1, -1);
          node(E('code', null, code));
          i = close + len;
        } else { buf += s.slice(i, j); i = j; }
        continue;
      }
      if (c === '*' || c === '_' || c === '~') {
        let j = i;
        while (s[j] === c) j++;
        const cnt = j - i;
        if (c === '~' && cnt !== 2) { buf += s.slice(i, j); i = j; continue; }
        const before = i > 0 ? s[i - 1] : ' ', after = j < n ? s[j] : ' ';
        const ws = ch => /\s/.test(ch), pu = ch => PUNCT.has(ch);
        const left = !ws(after) && (!pu(after) || ws(before) || pu(before));
        const right = !ws(before) && (!pu(before) || ws(after) || pu(after));
        const open = c === '_' ? left && (!right || pu(before)) : left;
        const close = c === '_' ? right && (!left || pu(after)) : right;
        flush();
        toks.push({ t: 'delim', ch: c, n: cnt, open, close });
        i = j;
        continue;
      }
      if (c === '[' || (c === '!' && s[i + 1] === '[')) {
        const img = c === '!';
        const r = linkAt(s, img ? i + 1 : i, pairs);
        if (r && (img || !inLink)) {
          node(img ? imageChip(r.label, r.url, inLink) : linkNode(r.label, r.url));
          i = r.end;
          continue;
        }
      }
      if (c === '<' && !inLink) {
        const r = autolinkAt(s, i), safe = r && safeUrl(r.href);
        if (safe) {
          const a = anchor(safe);
          a.textContent = r.label;
          node(a);
          i = r.end;
          continue;
        }
      }
      if ((c === 'h' || c === 'H') && !inLink && (i === 0 || !/[A-Za-z0-9]/.test(s[i - 1]))) {
        const r = bareUrlAt(s, i), safe = r && safeUrl(r.href);
        if (safe) {
          const a = anchor(safe);
          a.textContent = r.href;
          node(a);
          i = r.end;
          continue;
        }
      }
      buf += c;
      i++;
    }
    flush();
    emphasis(toks);
    build(toks, parent);
  }

  function emphasis(toks) {
    const stack = [];  // opener tokens' indexes
    for (let k = 0; k < toks.length; k++) {
      const t = toks[k];
      if (t.t !== 'delim') continue;
      if (t.close) {
        for (let matched = true; t.n > 0 && matched;) {
          matched = false;
          for (let s = stack.length - 1, steps = 0; s >= 0 && steps < 50; s--, steps++) {
            const o = toks[stack[s]];
            if (o.ch !== t.ch || o.n === 0 || (o.pairs || 0) >= 3) continue;
            if (t.ch === '~' && (o.n < 2 || t.n < 2)) continue;
            const use = t.ch === '~' || (o.n >= 2 && t.n >= 2) ? 2 : 1;
            const id = ++pid;
            (o.opens || (o.opens = [])).push({ tag: t.ch === '~' ? 'del' : use === 2 ? 'strong' : 'em', id });
            (t.closes || (t.closes = [])).push(id);
            o.n -= use; t.n -= use; o.pairs = (o.pairs || 0) + 1;
            stack.length = s + (o.n > 0 ? 1 : 0);  // openers inside it stay literal
            matched = true;
            break;
          }
        }
        if (t.n > 0 && t.open) stack.push(k);
      } else if (t.open) stack.push(k);
      if (stack.length > 400) stack.splice(0, stack.length - 400);
    }
  }

  function build(toks, parent) {
    const stack = [parent], byId = new Map();
    let cur = parent;
    for (const t of toks) {
      if (t.t === 'text') cur.append(t.v);
      else if (t.t === 'node') cur.append(t.n);
      else {
        if (t.closes) {
          for (const id of t.closes) {
            const e = byId.get(id);
            if (!e || stack.indexOf(e) < 0) continue;
            while (stack.length > 1 && stack.pop() !== e) { /* unwinds to it */ }
            cur = stack[stack.length - 1];
          }
        }
        if (t.n > 0) cur.append(t.ch.repeat(t.n));
        if (t.opens) {
          for (let q = t.opens.length - 1; q >= 0; q--) {
            const e = E(t.opens[q].tag);
            cur.append(e);
            stack.push(e);
            byId.set(t.opens[q].id, e);
            cur = e;
          }
        }
      }
    }
  }

  // ---- blocks ----
  const reFence = /^ {0,3}(`{3,}|~{3,})(.*)$/;
  const reFenceEnd = /^ {0,3}(`{3,}|~{3,}) *$/;
  const reBullet = /^( {0,3})([-*+])( +|$)/;
  const reOrdered = /^( {0,3})(\d{1,9})([.)])( +|$)/;
  const reQuote = /^ {0,3}>/;
  const reSetext1 = /^ {0,3}=+ *$/;
  const reSetext2 = /^ {0,3}-+ *$/;

  function isHr(l) {
    const m = /^ {0,3}([-*_])/.exec(l);
    if (!m) return false;
    const s = l.replace(/[ \t]/g, '');
    if (s.length < 3) return false;
    for (let i = 0; i < s.length; i++) if (s[i] !== m[1]) return false;
    return true;
  }
  function atx(l) {
    const m = /^ {0,3}(#{1,6})(?: +|$)/.exec(l);
    if (!m) return null;
    const rest = l.slice(m[0].length);
    let end = rest.length;
    while (end > 0 && rest[end - 1] === ' ') end--;
    let k = end;
    while (k > 0 && rest[k - 1] === '#') k--;
    if (k < end && (k === 0 || rest[k - 1] === ' ')) end = k;
    return { level: m[1].length, text: rest.slice(0, end).trim() };
  }
  function listStart(l) {
    let m = reBullet.exec(l), ordered = false, start = 1, marker;
    if (!m) {
      m = reOrdered.exec(l);
      if (!m) return null;
      ordered = true; start = +m[2]; marker = m[3];
    } else marker = m[2];
    const lead = m[1].length, mark = ordered ? m[2].length + 1 : 1, gap = (ordered ? m[4] : m[3]).length;
    const empty = gap === 0 || blank(l.slice(m[0].length));
    const content = empty || gap > 4 ? lead + mark + 1 : lead + mark + gap;
    return { ordered, start, marker, lead, content, empty };
  }
  function splitRow(l) {
    let s = l.trim();
    if (s[0] === '|') s = s.slice(1);
    if (s.endsWith('|') && !s.endsWith('\\|')) s = s.slice(0, -1);
    const cells = [];
    let cur = '';
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '\\' && s[i + 1] === '|') { cur += '|'; i++; continue; }
      if (s[i] === '|') { cells.push(cur.trim()); cur = ''; continue; }
      cur += s[i];
    }
    cells.push(cur.trim());
    return cells;
  }
  function delimRow(l) {
    if (l.indexOf('-') < 0 || !/^[ |:-]+$/.test(l)) return null;
    const out = [];
    for (const c of splitRow(l)) {
      if (!/^:?-+:?$/.test(c)) return null;
      const L = c[0] === ':', R = c[c.length - 1] === ':';
      out.push(L && R ? 'center' : R ? 'right' : L ? 'left' : '');
    }
    return out;
  }
  function tableAt(lines, i) {
    const l = lines[i];
    if (l.indexOf('|') < 0 || i + 1 >= lines.length) return null;
    const al = delimRow(lines[i + 1]);
    if (!al) return null;
    const head = splitRow(l);
    return head.length === al.length ? { head, al } : null;
  }
  function interrupts(lines, i) {
    const l = lines[i];
    if (atx(l) || reFence.test(l) || reQuote.test(l) || isHr(l)) return true;
    const st = listStart(l);
    if (st && !st.empty && (!st.ordered || st.start === 1)) return true;
    return !!tableAt(lines, i);
  }

  function heading(level, txt, parent) {
    const h = E('h' + Math.min(6, level + 2), 'm' + level);
    inline(txt, h, false);
    parent.append(h);
  }
  function para(txt, parent) {
    const p = E('p');
    inline(txt.replace(/^ +/gm, '').replace(/ +$/, ''), p, false);
    parent.append(p);
  }
  function codeBlock(body, info, parent, cls) {
    const wrap = E('div', 'codewrap' + (cls ? ' ' + cls : ''));
    if (info) wrap.append(E('div', 'label', info));
    const pre = E('pre'), code = E('code');
    code.textContent = body;
    pre.append(code);
    wrap.append(pre);
    parent.append(wrap);
  }
  function table(lines, i, t, parent) {
    const wrap = E('div', 'uscroll'), tb = E('table', 'utable'), thead = E('thead'), tr = E('tr');
    t.head.forEach((c, k) => {
      const th = E('th');
      if (t.al[k]) th.setAttribute('align', t.al[k]);
      inline(c, th, false);
      tr.append(th);
    });
    thead.append(tr);
    tb.append(thead);
    const body = E('tbody');
    i += 2;
    while (i < lines.length && !blank(lines[i]) && lines[i].indexOf('|') >= 0 && !interruptsTable(lines[i])) {
      const cells = splitRow(lines[i]), row = E('tr');
      for (let k = 0; k < t.head.length; k++) {
        const td = E('td');
        if (t.al[k]) td.setAttribute('align', t.al[k]);
        if (cells[k]) inline(cells[k], td, false);
        row.append(td);
      }
      body.append(row);
      i++;
    }
    tb.append(body);
    wrap.append(tb);
    parent.append(wrap);
    return i;
  }
  function interruptsTable(l) { return !!(atx(l) || reFence.test(l) || reQuote.test(l) || isHr(l)); }

  function list(lines, i, first, parent, depth) {
    const box = E(first.ordered ? 'ol' : 'ul');
    if (first.ordered && first.start !== 1) box.setAttribute('start', String(first.start));
    while (i < lines.length) {
      const st = listStart(lines[i]);
      if (!st || isHr(lines[i]) || st.ordered !== first.ordered || st.marker !== first.marker || st.lead > first.lead + 3) break;
      const need = Math.min(st.content, st.lead + 2);  // nested by 2 or more spaces
      const item = [lines[i].slice(Math.min(lines[i].length, st.content))];
      let j = i + 1;
      while (j < lines.length) {
        const l = lines[j];
        if (blank(l)) {
          let k = j + 1;
          while (k < lines.length && blank(lines[k])) k++;
          if (k < lines.length && indentOf(lines[k]) >= need) { for (; j < k; j++) item.push(''); continue; }
          break;
        }
        if (indentOf(l) >= need) { item.push(dedent(l, st.content)); j++; continue; }
        if (listStart(l) || interrupts(lines, j)) break;
        if (blank(item[item.length - 1])) break;
        item.push(l.trim());  // a lazy continuation of the item's paragraph
        j++;
      }
      const li = E('li');
      const task = /^\[([ xX])\](?: +|$)/.exec(item[0]);
      if (task) {
        li.setAttribute('class', 'task');
        const box2 = E('input');
        box2.setAttribute('type', 'checkbox');
        box2.setAttribute('disabled', '');
        if (task[1] !== ' ') box2.setAttribute('checked', '');
        li.append(box2);
        item[0] = item[0].slice(task[0].length);
      }
      blocks(item, li, depth + 1);
      box.append(li);
      i = j;
      while (i < lines.length && blank(lines[i])) {  // a blank line between items keeps the list going
        let k = i;
        while (k < lines.length && blank(lines[k])) k++;
        const nx = k < lines.length && listStart(lines[k]);
        if (nx && nx.ordered === first.ordered && nx.marker === first.marker) i = k; else break;
      }
    }
    parent.append(box);
    return i;
  }

  function blocks(lines, parent, depth) {
    let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (blank(l)) { i++; continue; }
      const fence = reFence.exec(l);
      if (fence && !(fence[1][0] === '`' && fence[2].indexOf('`') >= 0)) {
        const lead = indentOf(l), mark = fence[1], body = [];
        i++;
        while (i < lines.length) {
          const e = reFenceEnd.exec(lines[i]);
          if (e && e[1][0] === mark[0] && e[1].length >= mark.length) { i++; break; }
          body.push(dedent(lines[i], lead));
          i++;
        }
        codeBlock(body.join('\n'), fence[2].trim().split(' ')[0], parent);
        continue;
      }
      const h = atx(l);
      if (h) { heading(h.level, h.text, parent); i++; continue; }
      if (isHr(l)) { parent.append(E('hr')); i++; continue; }
      // Quotes and lists nest 6 deep at most: deeper ones are paragraphs, their markers left as text.
      const nest = depth < MAXDEPTH;
      if (nest && reQuote.test(l)) {
        const inner = [];
        while (i < lines.length) {
          const q = /^ {0,3}> ?/.exec(lines[i]);
          if (q) inner.push(lines[i].slice(q[0].length));
          else if (!blank(lines[i]) && inner.length && !blank(inner[inner.length - 1]) && !interrupts(lines, i)) inner.push(lines[i]);
          else break;
          i++;
        }
        const bq = E('blockquote');
        blocks(inner, bq, depth + 1);
        parent.append(bq);
        continue;
      }
      const st = nest && listStart(l);
      if (st) { i = list(lines, i, st, parent, depth); continue; }
      if (indentOf(l) >= 4) {
        const body = [];
        while (i < lines.length && (blank(lines[i]) || indentOf(lines[i]) >= 4)) { body.push(dedent(lines[i], 4)); i++; }
        while (body.length && blank(body[body.length - 1])) body.pop();
        codeBlock(body.join('\n'), '', parent);
        continue;
      }
      const t = tableAt(lines, i);
      if (t) { i = table(lines, i, t, parent); continue; }
      const p = [l];
      i++;
      let done = false;
      while (i < lines.length && !blank(lines[i])) {
        if (reSetext1.test(lines[i]) || reSetext2.test(lines[i])) {
          heading(reSetext1.test(lines[i]) ? 1 : 2, p.join('\n').trim(), parent);
          i++;
          done = true;
          break;
        }
        if (interrupts(lines, i)) break;
        p.push(lines[i]);
        i++;
      }
      if (!done) para(p.join('\n'), parent);
    }
  }

  const lines = text.split('\n').map(expandLead);
  let from = 0;
  if (lines[0] === '---') {  // YAML front matter at the very top
    for (let k = 1; k < Math.min(lines.length, 200); k++) {
      if (lines[k] === '---' || lines[k] === '...') {
        codeBlock(lines.slice(1, k).join('\n'), 'Front matter', frag, 'front');
        from = k + 1;
        break;
      }
    }
  }
  blocks(from ? lines.slice(from) : lines, frag, 0);
  if (cut) frag.append(E('p', 'mdnote', 'Showing the first 200 KB. Open the file for the rest.'));
  return frag;
}
