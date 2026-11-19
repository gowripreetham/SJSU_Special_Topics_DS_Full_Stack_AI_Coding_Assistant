/* DS Kit — a tiny declarative dashboard runtime shared by all 16 projects.
 * The Python backend describes each page as a list of JSON "blocks"; this file renders them.
 * No build step, no framework: vanilla JS + a vendored Chart.js. */
(function () {
  const $ = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const SERIES = () => [1, 2, 3, 4, 5, 6, 7, 8].map(i => css('--s' + i));
  const fmt = (v, d) => {
    if (v == null || v === '') return '—';
    if (typeof v !== 'number') return esc(v);
    if (!isFinite(v)) return '—';
    if (d != null && !Number.isInteger(v)) return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
    if (Number.isInteger(v)) return v.toLocaleString();
    const a = Math.abs(v);
    return v.toLocaleString(undefined, { maximumFractionDigits: a >= 100 ? 1 : a >= 1 ? 3 : 4 });
  };

  // ---------- markdown-lite ----------
  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>')
      .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, '$1<i>$2</i>')
      .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }
  function md(src) {
    const lines = String(src || '').replace(/\r/g, '').split('\n'); let out = '', i = 0;
    while (i < lines.length) {
      let l = lines[i];
      if (/^```/.test(l)) { let buf = []; i++; while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]); i++; out += '<pre class="code">' + esc(buf.join('\n')) + '</pre>'; continue; }
      const h = l.match(/^(#{1,4})\s+(.*)/); if (h) { const n = Math.min(h[1].length, 3); out += `<h${n}>${inline(h[2])}</h${n}>`; i++; continue; }
      if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
        const row = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
        out += '<table><tr>' + row(l).map(c => `<th>${inline(c)}</th>`).join('') + '</tr>'; i += 2;
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) { out += '<tr>' + row(lines[i]).map(c => `<td>${inline(c)}</td>`).join('') + '</tr>'; i++; }
        out += '</table>'; continue;
      }
      if (/^\s*([-*]|\d+\.)\s+/.test(l)) {
        const ord = /^\s*\d+\./.test(l); out += ord ? '<ol>' : '<ul>';
        while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) { out += '<li>' + inline(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, '')) + '</li>'; i++; }
        out += ord ? '</ol>' : '</ul>'; continue;
      }
      if (/^>\s?/.test(l)) { out += '<blockquote>' + inline(l.replace(/^>\s?/, '')) + '</blockquote>'; i++; continue; }
      if (!l.trim()) { i++; continue; }
      let p = [l]; i++;
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|```|\s*([-*]|\d+\.)\s|>|\s*\|)/.test(lines[i])) p.push(lines[i++]);
      out += '<p>' + inline(p.join(' ')) + '</p>';
    }
    return out;
  }

  // ---------- tooltip for non-Chart.js marks ----------
  const tip = $('div', 'tip'); document.addEventListener('DOMContentLoaded', () => document.body.appendChild(tip));
  function hover(el, text) {
    el.addEventListener('mousemove', e => { tip.style.display = 'block'; tip.innerHTML = text; tip.style.left = Math.min(e.clientX + 12, innerWidth - 330) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; });
    el.addEventListener('mouseleave', () => tip.style.display = 'none');
  }

  // ---------- color scales ----------
  const hex2rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => { const A = hex2rgb(a), B = hex2rgb(b); return 'rgb(' + A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',') + ')'; };
  function seqColor(t) { t = Math.max(0, Math.min(1, t)); const dark = document.documentElement.dataset.theme === 'dark'; return dark ? mix('#16283f', '#86b6ef', t) : mix('#e8f1fc', '#104281', t); }
  function divColor(t) { t = Math.max(-1, Math.min(1, t)); const mid = css('--neutral'); return t < 0 ? mix(mid, '#2a78d6', -t) : mix(mid, '#eb6834', t); }
  const inkOn = t => Math.abs(t) > 0.55 ? '#fff' : css('--ink');

  // ---------- block renderers ----------
  const R = {};
  const charts = [];
  function card(b, el) { const c = $('div', 'card'); if (b.title) c.appendChild($('div', 'ctitle', esc(b.title))); if (b.sub) c.appendChild($('div', 'csub', inline(b.sub))); el.appendChild(c); return c; }

  R.h = (b, el) => { el.appendChild($('h1', 'pagetitle', esc(b.text))); if (b.sub) el.appendChild($('p', 'pagesub', inline(b.sub))); };
  R.md = (b, el) => { const c = b.card === false ? el : card(b, el); c.appendChild($('div', 'md', md(b.text))); };
  R.html = (b, el) => { const c = b.card === false ? el : card(b, el); c.appendChild($('div', '', b.html)); };
  R.callout = (b, el) => { el.appendChild($('div', 'callout ' + (b.kind || 'info') + (b.err ? ' kit-error' : ''), (b.title ? `<b class="t">${esc(b.title)}</b>` : '') + md(b.text))); };
  R.code = (b, el) => { const c = card(b, el); c.appendChild($('pre', 'code', esc(b.text))); };
  R.kpis = (b, el) => {
    const g = $('div', 'kpis');
    for (const k of b.items) {
      const d = k.delta != null ? ` <span class="d ${k.delta_good === false ? 'down' : 'up'}" style="font-size:12px">${esc(k.delta)}</span>` : '';
      g.appendChild($('div', 'kpi', `<div class="l">${esc(k.label)}</div><div class="v">${typeof k.value === 'number' ? fmt(k.value, k.digits) : esc(k.value)}${d}</div>${k.hint ? `<div class="h">${esc(k.hint)}</div>` : ''}`));
    }
    el.appendChild(g);
  };
  R.grid = (b, el) => { const g = $('div', 'grid g' + (b.cols || 2)); el.appendChild(g); render(g, b.items); };
  R.tabs = (b, el) => {
    const bar = $('div', 'tabs'), body = $('div'); el.appendChild(bar); el.appendChild(body);
    const show = i => { [...bar.children].forEach((t, j) => t.classList.toggle('on', i === j)); body.innerHTML = ''; render(body, b.tabs[i].blocks); };
    b.tabs.forEach((t, i) => { const e = $('div', 'tab', esc(t.title)); e.onclick = () => show(i); bar.appendChild(e); }); show(0);
  };
  R.steps = (b, el) => {
    const s = $('div', 'steps');
    b.items.forEach((it, i) => s.appendChild($('div', 'step', `<div class="n">${it.n ?? i + 1}</div><div><div class="tt">${esc(it.title)} ${it.status ? `<span class="pill ${it.kind || 'good'}">${esc(it.status)}</span>` : ''}</div><div class="md">${md(it.text)}</div></div>`)));
    el.appendChild(s);
  };
  R.table = (b, el) => {
    const c = card(b, el), wrap = $('div', 'tscroll'); if (b.maxh) wrap.style.maxHeight = b.maxh + 'px';
    const t = $('table', 't'); let rows = b.rows.slice(), sortCol = -1, asc = true;
    const isNum = j => rows.length && rows.every(r => r[j] == null || typeof r[j] === 'number');
    const cell = v => {
      if (v && typeof v === 'object' && v.pill) return `<span class="pill ${v.kind || ''}">${esc(v.pill)}</span>`;
      if (v && typeof v === 'object' && v.bar != null) return `<div style="display:flex;align-items:center;gap:8px"><div style="flex:1;height:8px;background:var(--neutral);border-radius:4px"><div style="width:${Math.max(0, Math.min(100, v.bar * 100))}%;height:8px;border-radius:4px;background:var(--s1)"></div></div><span>${esc(v.text ?? '')}</span></div>`;
      if (v && typeof v === 'object' && v.code) return `<code style="font:12px var(--mono)">${esc(v.code)}</code>`;
      return typeof v === 'number' ? fmt(v, b.digits) : inline(v ?? '');
    };
    const draw = () => {
      t.innerHTML = '<thead><tr>' + b.cols.map((h, j) => `<th class="${isNum(j) ? 'num' : ''}" data-j="${j}">${esc(h)}${sortCol === j ? (asc ? ' ▲' : ' ▼') : ''}</th>`).join('') + '</tr></thead><tbody>' +
        rows.map(r => '<tr>' + r.map((v, j) => `<td class="${typeof v === 'number' ? 'num' : ''}">${cell(v)}</td>`).join('') + '</tr>').join('') + '</tbody>';
      t.querySelectorAll('th').forEach(th => th.onclick = () => {
        const j = +th.dataset.j; asc = sortCol === j ? !asc : true; sortCol = j;
        const key = v => (v && typeof v === 'object') ? (v.text ?? v.pill ?? v.code ?? '') : v;
        rows.sort((a, c2) => { const x = key(a[j]), y = key(c2[j]); return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1); }); draw();
      });
    };
    draw(); wrap.appendChild(t); c.appendChild(wrap);
  };
  R.heatmap = (b, el) => {
    const c = card(b, el), xs = b.xs, ys = b.ys, z = b.z; const flat = z.flat().filter(v => v != null);
    const lo = b.min ?? Math.min(...flat), hi = b.max ?? Math.max(...flat), amax = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
    const g = $('div', 'hm'); g.style.gridTemplateColumns = `minmax(60px,${b.labw || 130}px) repeat(${xs.length},minmax(${b.cellw || 16}px,1fr))`;
    ys.forEach((y, i) => {
      g.appendChild($('div', 'lab', esc(y)));
      xs.forEach((x, j) => {
        const v = z[i][j]; const t = v == null ? 0 : b.diverging ? v / amax : (v - lo) / ((hi - lo) || 1);
        const d = $('div', 'c', b.show === false || v == null ? '' : fmt(v, b.digits ?? 2));
        d.style.background = v == null ? 'transparent' : b.diverging ? divColor(t) : seqColor(t); d.style.color = inkOn(b.diverging ? t : t * 1.2 - .1);
        hover(d, `<b>${esc(y)}</b> × <b>${esc(x)}</b><br>${fmt(v, b.digits ?? 3)}`); g.appendChild(d);
      });
    });
    g.appendChild($('div')); xs.forEach(x => g.appendChild($('div', 'xlab', esc(x))));
    c.appendChild(g);
  };
  R.chart = (b, el) => {
    const c = card(b, el), wrap = $('div', 'chartwrap'); wrap.style.height = (b.height || 280) + 'px'; const cv = $('canvas'); wrap.appendChild(cv); c.appendChild(wrap);
    const S = SERIES(), grid = css('--grid'), ink2 = css('--ink2'), surface = css('--surface');
    const kind = b.kind || 'line', horiz = kind === 'hbar', isBar = kind === 'bar' || horiz || kind === 'stackedbar';
    const type = isBar ? 'bar' : kind === 'area' ? 'line' : kind;
    const datasets = (b.series || []).map((s, i) => {
      const col = s.color ? (/^\d$/.test(String(s.color)) ? S[s.color - 1] : s.color.startsWith('--') ? css(s.color) : s.color) : S[i % 8];
      const o = { label: s.name || '', data: s.data, borderColor: col, backgroundColor: col, ...(s.extra || {}) };
      if (type === 'line') Object.assign(o, { borderWidth: s.width ?? 2, pointRadius: s.points ? 3 : 0, pointHoverRadius: 5, tension: s.smooth ? .3 : 0, borderDash: s.dash ? [6, 4] : [], fill: s.fill ?? (kind === 'area' ? 'origin' : false), backgroundColor: s.fill != null || kind === 'area' ? col + '33' : col, spanGaps: true, stepped: s.stepped || false });
      if (type === 'bar') Object.assign(o, { borderRadius: 4, borderWidth: 0, maxBarThickness: 34, backgroundColor: s.colors ? s.colors.map(k => /^\d$/.test(String(k)) ? S[k - 1] : k.startsWith('--') ? css(k) : k) : col });
      if (type === 'scatter' || type === 'bubble') Object.assign(o, { pointRadius: s.r ?? 3.5, pointHoverRadius: 6, backgroundColor: col + (s.alpha || 'b0'), borderColor: surface, borderWidth: s.ring === false ? 0 : 1, showLine: !!s.line, ...(s.line ? { borderColor: col, borderWidth: 2, pointRadius: 0 } : {}) });
      if (type === 'doughnut') Object.assign(o, { backgroundColor: S, borderColor: surface, borderWidth: 2 });
      return o;
    });
    const ax = (title, o = {}) => ({ title: { display: !!title, text: title, color: ink2, font: { size: 11.5 } }, grid: { color: grid, drawTicks: false }, border: { display: false }, ticks: { color: ink2, font: { size: 11 }, maxRotation: 0, autoSkipPadding: 14 }, ...o });
    const lin = type === 'scatter' || b.linear_x;
    const opts = {
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 }, indexAxis: horiz ? 'y' : 'x',
      interaction: type === 'line' ? { mode: 'index', intersect: false } : { mode: 'nearest', intersect: true },
      plugins: { legend: { display: b.legend ?? (datasets.filter(d => d.label).length > 1 || type === 'doughnut'), position: 'bottom', labels: { color: ink2, boxWidth: 10, boxHeight: 10, font: { size: 11.5 }, filter: it => !!it.text && !it.text.startsWith('_') } }, tooltip: { backgroundColor: '#16181d', padding: 9, filter: it => !(it.dataset.label || '').startsWith('_') } },
      scales: type === 'doughnut' ? {} : {
        x: ax(b.x, { type: horiz ? 'linear' : lin ? (b.logx ? 'logarithmic' : 'linear') : 'category', stacked: kind === 'stackedbar' || !!b.stacked, ...(horiz ? { beginAtZero: true } : { grid: { display: lin || !!b.xgrid, color: grid } }) }),
        y: ax(b.y, { type: horiz ? 'category' : b.logy ? 'logarithmic' : 'linear', stacked: kind === 'stackedbar' || !!b.stacked, beginAtZero: isBar && !horiz, ...(horiz ? { grid: { display: false }, ticks: { color: ink2, font: { size: 11 }, autoSkip: false } } : {}), ...(b.ymin != null ? { min: b.ymin } : {}), ...(b.ymax != null ? { max: b.ymax } : {}), ...(b.reverse_y ? { reverse: true } : {}) }),
      },
    };
    if (b.xmin != null) opts.scales.x.min = b.xmin; if (b.xmax != null) opts.scales.x.max = b.xmax;
    if (b.onclick && Kit.custom[b.onclick]) opts.onClick = (ev, items, ch) => Kit.custom[b.onclick](ev, items, ch, b);
    const ch = new Chart(cv, { type, data: { labels: b.labels, datasets }, options: opts }); charts.push(ch); cv._chart = ch;
    if (b.note) c.appendChild($('div', 'csub', inline(b.note))).style.margin = '8px 0 0';
  };
  R.form = (b, el) => {
    const c = card(b, el), f = $('div', 'form'), res = $('div', 'formresult'), vals = {};
    const read = () => { const o = {}; for (const fl of b.fields) { const e = vals[fl.name]; o[fl.name] = fl.type === 'checkbox' ? e.checked : (fl.type === 'number' || fl.type === 'range') ? parseFloat(e.value) : e.value; } return o; };
    const go = async () => {
      btn.disabled = true; const old = btn.textContent; btn.textContent = 'Running…';
      try { const r = await Kit.api(b.action, read()); res.innerHTML = ''; render(res, r.blocks || []); if (b.then && Kit.custom[b.then]) Kit.custom[b.then](r, read()); }
      catch (e) { res.innerHTML = `<div class="callout crit kit-error"><b class="t">Request failed</b>${esc(e.message)}</div>`; }
      btn.disabled = false; btn.textContent = old;
    };
    for (const fl of b.fields) {
      const w = $('div', 'field' + (fl.wide || fl.type === 'textarea' ? ' wide' : '')); let e;
      if (fl.type === 'select') { e = $('select'); for (const o of fl.options) { const op = $('option'); const [v, t] = Array.isArray(o) ? o : [o, o]; op.value = v; op.textContent = t; if (String(v) === String(fl.value)) op.selected = true; e.appendChild(op); } }
      else if (fl.type === 'textarea') { e = $('textarea'); e.rows = fl.rows || 3; e.value = fl.value ?? ''; }
      else { e = $('input'); e.type = fl.type || 'text'; for (const k of ['min', 'max', 'step', 'placeholder']) if (fl[k] != null) e[k] = fl[k]; if (fl.type === 'checkbox') e.checked = !!fl.value; else e.value = fl.value ?? ''; }
      e.dataset.name = fl.name; vals[fl.name] = e;
      const lab = $('label', '', esc(fl.label || fl.name)); if (fl.type === 'range') { const rv = $('span', 'rv', e.value); lab.appendChild(rv); e.addEventListener('input', () => rv.textContent = e.value); }
      w.appendChild(lab); w.appendChild(e); f.appendChild(w);
      if (b.auto) e.addEventListener(fl.type === 'range' ? 'change' : 'input', () => { clearTimeout(f._t); f._t = setTimeout(go, 250); });
      if (fl.type !== 'textarea') e.addEventListener('keydown', ev => { if (ev.key === 'Enter') go(); });
    }
    const bw = $('div', 'field'), btn = $('button', 'btn', esc(b.submit || 'Run')); btn.onclick = go; bw.appendChild(btn); f.appendChild(bw);
    c.appendChild(f); c.appendChild(res); c._form = { vals, go }; if (b.id) Kit.forms[b.id] = c._form;
    if (b.result) render(res, b.result); if (b.run_on_load) go();
  };
  R.quiz = (b, el) => {
    const c = card(b, el), q = $('div', 'quiz'); let score = 0, done = 0; const sc = $('div', 'csub', '');
    b.questions.forEach((qq, qi) => {
      const d = $('div', 'q', `<div class="qt">${qi + 1}. ${inline(qq.q)}</div>`);
      qq.options.forEach((o, oi) => { const op = $('div', 'opt', inline(o)); op.onclick = () => { if (d.classList.contains('done')) return; d.classList.add('done'); done++; if (oi === qq.answer) { op.classList.add('right'); score++; } else { op.classList.add('wrong'); d.querySelectorAll('.opt')[qq.answer].classList.add('right'); } sc.textContent = `Score: ${score} / ${done} answered (${b.questions.length} total)`; }; d.appendChild(op); });
      d.appendChild($('div', 'why', '💡 ' + inline(qq.why || ''))); q.appendChild(d);
    });
    c.appendChild(q); c.appendChild(sc);
  };
  R.custom = (b, el) => { const c = b.card === false ? el : card(b, el); const fn = Kit.custom[b.name]; if (fn) fn(c, b.props || {}, b); else c.appendChild($('div', 'callout warn kit-error', 'Missing custom block: ' + esc(b.name))); };

  function render(el, blocks) {
    for (const b of blocks || []) {
      if (!b) continue; const w = $('div', 'block'); el.appendChild(w);
      try { (R[b.t] || ((bb, e) => e.appendChild($('div', 'callout warn kit-error', 'Unknown block ' + esc(bb.t)))))(b, w); }
      catch (e) { console.error(e); w.appendChild($('div', 'callout crit kit-error', 'Render error: ' + esc(e.message))); }
    }
  }

  // ---------- app shell ----------
  const Kit = window.Kit = {
    custom: {}, forms: {}, render, md, esc, fmt, $, hover, css, seqColor, divColor, SERIES, meta: null,
    async api(path, body) {
      if (window.KIT_STATIC) { const k = path; if (k in window.KIT_STATIC) return JSON.parse(JSON.stringify(window.KIT_STATIC[k])); throw new Error('This static export has no back end for ' + path); }
      const r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!r.ok) { let m = r.status + ' ' + r.statusText; try { const j = await r.json(); m = j.detail || m; } catch (_) { } throw new Error(m); }
      return r.json();
    },
    async go(pid) {
      const main = document.getElementById('main'); charts.splice(0).forEach(c => c.destroy()); Kit.forms = {};
      main.innerHTML = '<div class="loading">Loading…</div>';
      document.querySelectorAll('.navitem').forEach(n => n.classList.toggle('on', n.dataset.id === pid));
      try { const p = await Kit.api('api/page/' + pid); main.innerHTML = ''; render(main, p.blocks); window.scrollTo(0, 0); document.body.dataset.page = pid; document.body.dataset.ready = '1'; }
      catch (e) { main.innerHTML = `<div class="callout crit kit-error"><b class="t">Could not load page</b>${esc(e.message)}</div>`; }
    },
    route() { const pid = location.hash.slice(1) || Kit.meta.pages[0].id; document.body.dataset.ready = ''; Kit.go(pid); },
  };
  async function boot() {
    const saved = localStorage.getItem('kit-theme'); if (saved) document.documentElement.dataset.theme = saved;
    const meta = Kit.meta = await Kit.api('api/meta'); document.title = meta.title;
    const side = document.getElementById('side');
    side.innerHTML = `<div id="brand"><div class="num">PROJECT ${esc(meta.number)}</div><div class="name">${esc(meta.title)}</div><div class="sub">${esc(meta.subtitle || '')}</div></div>`;
    let g = null;
    for (const p of meta.pages) { if (p.group !== g) { g = p.group; side.appendChild($('div', 'navgroup', esc(g))); } const a = $('a', 'navitem', esc(p.title)); a.dataset.id = p.id; a.href = '#' + p.id; side.appendChild(a); }
    const foot = $('div', '', ''); foot.id = 'sidefoot';
    foot.innerHTML = `<a href="${meta.hub ? '../../' : '#'}" style="${meta.hub ? '' : 'display:none'}">← All projects</a><button class="themebtn">◐ Theme</button><span>${esc(meta.footer || '')}</span>`;
    foot.querySelector('button').onclick = () => { const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = t; localStorage.setItem('kit-theme', t); Kit.route(); };
    side.appendChild(foot);
    if (meta.custom_js) await new Promise((ok) => { const s = document.createElement('script'); s.src = 'web/custom.js'; s.onload = ok; s.onerror = ok; document.head.appendChild(s); });
    window.addEventListener('hashchange', Kit.route); Kit.route();
  }
  document.addEventListener('DOMContentLoaded', boot);
})();
