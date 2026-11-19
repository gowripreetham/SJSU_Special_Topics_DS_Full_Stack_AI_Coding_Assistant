// Zenith board: kanban + list views, drag & drop, quick add, filters, live SSE sync.
(function () {
  const { $, esc, api } = Kit; const COLS = [['todo', 'To do'], ['doing', 'In progress'], ['done', 'Done']];
  const PR = { urgent: ['crit', '▲▲ urgent'], high: ['warn', '▲ high'], medium: ['info', '● medium'], low: ['', '▽ low'] };
  const st = document.createElement('style'); st.textContent = `
  .zbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;align-items:center}
  .zbar input,.zbar select{padding:8px 10px;border:1px solid var(--axis);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit}
  .zquick{flex:1;min-width:260px}.zcols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
  .zcol{background:var(--neutral);border-radius:12px;padding:10px;min-height:220px}.zcol.over{outline:2px dashed var(--accent)}
  .zcol h4{margin:2px 6px 10px;font-size:13px;display:flex;justify-content:space-between}.zcol h4 span{color:var(--muted);font-weight:500}
  .zcard{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:grab}
  .zcard:hover{border-color:var(--accent)}.zcard.dragging{opacity:.4}.zcard .tt{font-weight:600}.zcard.done .tt{text-decoration:line-through;color:var(--muted)}
  .zmeta{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px;font-size:11.5px;color:var(--ink2);align-items:center}
  .zx{float:right;border:0;background:none;color:var(--muted);cursor:pointer;font-size:15px}.zx:hover{color:var(--crit)}
  .zlive{font-size:12px;color:var(--ink2)}.zlive i{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--muted);margin-right:5px}.zlive.on i{background:var(--good)}
  @media(max-width:900px){.zcols{grid-template-columns:1fr}}`; document.head.appendChild(st);

  Kit.custom.todo = function (el) {
    let tasks = [], view = 'board', q = '', fp = '', es = null;
    el.innerHTML = `<div class="zbar"><input class="zquick" placeholder='Quick add — e.g. "Email professor tomorrow #assignment !urgent"  (Enter)'>
      <input class="zsearch" placeholder="Search…" style="width:150px"><select class="zprio"><option value="">All priorities</option>${Object.keys(PR).map(p => `<option>${p}</option>`).join('')}</select>
      <button class="btn ghost zview">List view</button><span class="zlive"><i></i><span>connecting…</span></span></div><div class="zbody"></div>`;
    const body = el.querySelector('.zbody'), live = el.querySelector('.zlive'), today = new Date().toISOString().slice(0, 10);
    const vis = () => tasks.filter(t => (!fp || t.priority === fp) && (!q || (t.title + ' ' + t.tags.join(' ')).toLowerCase().includes(q)));
    const cardHtml = t => {
      const od = t.due && t.due < today && t.status !== 'done';
      return `<button class="zx" title="Delete" data-del="${t.id}">×</button><div class="tt">${esc(t.title)}</div><div class="zmeta"><span class="pill ${PR[t.priority][0]}">${PR[t.priority][1]}</span>${t.due ? `<span class="pill ${od ? 'crit' : ''}">${od ? '⚠ overdue · ' : '📅 '}${t.due}</span>` : ''}${t.tags.map(g => `<span>#${esc(g)}</span>`).join('')}</div>`;
    };
    function draw() {
      if (view === 'board') {
        body.innerHTML = '<div class="zcols">' + COLS.map(([k, n]) => `<div class="zcol" data-s="${k}"><h4>${n}<span>${vis().filter(t => t.status === k).length}</span></h4></div>`).join('') + '</div>';
        for (const t of vis()) { const c = $('div', 'zcard' + (t.status === 'done' ? ' done' : ''), cardHtml(t)); c.draggable = true; c.dataset.id = t.id; c.ondragstart = e => { e.dataTransfer.setData('text/plain', t.id); c.classList.add('dragging'); }; c.ondragend = () => c.classList.remove('dragging'); body.querySelector(`[data-s=${t.status}]`).appendChild(c); }
        body.querySelectorAll('.zcol').forEach(col => { col.ondragover = e => { e.preventDefault(); col.classList.add('over'); }; col.ondragleave = () => col.classList.remove('over'); col.ondrop = e => { e.preventDefault(); col.classList.remove('over'); move(+e.dataTransfer.getData('text/plain'), col.dataset.s); }; });
      } else {
        body.innerHTML = '<div class="card"><table class="t"><thead><tr><th></th><th>Task</th><th>Priority</th><th>Due</th><th>Tags</th><th>Status</th><th></th></tr></thead><tbody>' + vis().map(t => `<tr><td><input type="checkbox" data-chk="${t.id}" ${t.status === 'done' ? 'checked' : ''}></td><td>${esc(t.title)}</td><td><span class="pill ${PR[t.priority][0]}">${PR[t.priority][1]}</span></td><td>${t.due || '—'}</td><td>${t.tags.map(g => '#' + esc(g)).join(' ')}</td><td>${t.status}</td><td><button class="zx" data-del="${t.id}">×</button></td></tr>`).join('') + '</tbody></table></div>';
      }
      body.querySelectorAll('[data-del]').forEach(b => b.onclick = async () => { tasks = tasks.filter(t => t.id != b.dataset.del); draw(); await fetch('api/tasks/' + b.dataset.del, { method: 'DELETE' }); });
      body.querySelectorAll('[data-chk]').forEach(b => b.onchange = () => move(+b.dataset.chk, b.checked ? 'done' : 'todo'));
    }
    async function move(id, status) { const t = tasks.find(x => x.id === id); if (!t || t.status === status) return; t.status = status; draw(); await fetch('api/tasks/' + id, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) }); }
    el.querySelector('.zquick').onkeydown = async e => { if (e.key !== 'Enter' || !e.target.value.trim()) return; const text = e.target.value; e.target.value = ''; const t = await api('api/quick', { text }); if (!tasks.find(x => x.id === t.id)) tasks.push(t); draw(); };
    el.querySelector('.zsearch').oninput = e => { q = e.target.value.toLowerCase(); draw(); };
    el.querySelector('.zprio').onchange = e => { fp = e.target.value; draw(); };
    el.querySelector('.zview').onclick = e => { view = view === 'board' ? 'list' : 'board'; e.target.textContent = view === 'board' ? 'List view' : 'Board view'; draw(); };
    api('api/tasks').then(t => { tasks = t; draw(); });
    es = new EventSource('api/events');
    es.onopen = () => { live.classList.add('on'); live.lastChild.textContent = 'live sync on'; };
    es.onerror = () => { live.classList.remove('on'); live.lastChild.textContent = 'reconnecting…'; };
    es.onmessage = m => { const ev = JSON.parse(m.data); if (ev.kind === 'deleted') tasks = tasks.filter(t => t.id !== ev.id); else { const i = tasks.findIndex(t => t.id === ev.task.id); if (i >= 0) tasks[i] = ev.task; else tasks.push(ev.task); } draw(); };
    window.addEventListener('hashchange', () => es && es.close(), { once: true });
  };
})();
