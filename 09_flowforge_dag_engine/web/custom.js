// FlowForge canvas: drag nodes, click an output port then a node to connect, click an edge to delete, edit params, run.
(function () {
  const { $, esc } = Kit; const NW = 150, NH = 58;
  const st = document.createElement('style'); st.textContent = `.ff-bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px}.ff-bar select{padding:7px 9px;border:1px solid var(--axis);border-radius:8px;background:var(--surface);color:var(--ink);font:inherit;max-width:330px}
  .ff-wrap{display:grid;grid-template-columns:minmax(0,1fr) 230px;gap:14px}.ff-svg{width:100%;height:auto;aspect-ratio:1210/360;background:var(--neutral);border-radius:10px;user-select:none}
  .ff-side{background:var(--neutral);border-radius:10px;padding:12px;font-size:13px}.ff-side .field{margin-bottom:9px}.ff-node{cursor:grab}.ff-edge{cursor:pointer}.ff-edge:hover{stroke:var(--crit)}
  @media(max-width:900px){.ff-wrap{grid-template-columns:1fr}}`; document.head.appendChild(st);
  const COL = { pending: '--axis', ok: '--good', cached: '--s1', failed: '--crit', skipped: '--warn' };
  Kit.custom.dag = function (el, props) {
    const specs = props.specs, tpl = props.templates; let g = { nodes: [], edges: [] }, sel = null, linking = null, status = {}, seq = 1;
    el.innerHTML = `<div class="ff-bar"><select class="tpl">${Object.keys(tpl).map(k => `<option>${esc(k)}</option>`).join('')}</select><select class="add"><option value="">+ add node…</option>${Object.values(specs).map(s => `<option value="${s.kind}">${esc(s.label)} (${s.inputs.join(',') || '∅'} → ${s.output})</option>`).join('')}</select>
      <button class="btn run">▶ Run pipeline</button><button class="btn ghost val">Validate</button><button class="btn ghost del">Delete selected</button><span class="csub hint" style="margin:0"></span></div>
      <div class="ff-wrap"><svg class="ff-svg" viewBox="0 0 1210 360" preserveAspectRatio="xMinYMin meet"></svg><div class="ff-side"></div></div><div class="out" style="margin-top:14px"></div>`;
    const svg = el.querySelector('svg'), side = el.querySelector('.ff-side'), out = el.querySelector('.out'), hint = el.querySelector('.hint');
    const node = id => g.nodes.find(n => n.id === id);
    function draw() {
      let h = '<defs><marker id="ffa" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--muted)"/></marker></defs>';
      g.edges.forEach((e, i) => { const a = node(e.src), b = node(e.dst); if (!a || !b) return; const x1 = a.x + NW, y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2, mx = (x1 + x2) / 2; h += `<path class="ff-edge" data-e="${i}" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2 - 3},${y2}" fill="none" stroke="var(--muted)" stroke-width="2" marker-end="url(#ffa)"><title>click to delete</title></path>`; });
      for (const n of g.nodes) {
        const s = specs[n.kind] || { label: n.kind, output: '?' }, stt = status[n.id], c = `var(${COL[stt ? stt.status : 'pending']})`;
        h += `<g class="ff-node" data-n="${n.id}" transform="translate(${n.x},${n.y})"><rect width="${NW}" height="${NH}" rx="9" fill="var(--surface)" stroke="${sel === n.id ? 'var(--accent)' : c}" stroke-width="${sel === n.id ? 3 : 2}"/><rect width="6" height="${NH}" rx="3" fill="${c}"/>
          <text x="14" y="20" font-size="12.5" font-weight="700" fill="var(--ink)">${esc(n.id)}</text><text x="14" y="36" font-size="11" fill="var(--ink2)">${esc(s.label)}</text><text x="14" y="50" font-size="10" fill="${c}">${stt ? stt.status + ' · ' + stt.ms.toFixed(0) + ' ms' : '→ ' + s.output}</text>
          <circle class="ff-port" data-p="${n.id}" cx="${NW}" cy="${NH / 2}" r="6" fill="${linking === n.id ? 'var(--s2)' : 'var(--accent)'}" stroke="var(--surface)" stroke-width="2" style="cursor:crosshair"><title>click, then click the target node</title></circle></g>`;
      }
      svg.innerHTML = h;
      svg.querySelectorAll('.ff-edge').forEach(p => p.onclick = () => { g.edges.splice(+p.dataset.e, 1); status = {}; draw(); });
      svg.querySelectorAll('.ff-port').forEach(p => p.onmousedown = ev => { ev.stopPropagation(); linking = p.dataset.p; hint.textContent = `connecting from "${linking}" — click the target node`; draw(); });
      svg.querySelectorAll('.ff-node').forEach(gn => gn.onmousedown = ev => {
        const id = gn.dataset.n, n = node(id);
        if (linking && linking !== id) { if (!g.edges.some(e => e.src === linking && e.dst === id)) g.edges.push({ src: linking, dst: id }); linking = null; hint.textContent = ''; status = {}; draw(); return; }
        sel = id; const pt = svg.createSVGPoint(), toSvg = e => { pt.x = e.clientX; pt.y = e.clientY; return pt.matrixTransform(svg.getScreenCTM().inverse()); }, s0 = toSvg(ev), ox = n.x, oy = n.y;
        const mv = e => { const p = toSvg(e); n.x = Math.max(0, Math.min(1210 - NW, ox + p.x - s0.x)); n.y = Math.max(0, Math.min(360 - NH, oy + p.y - s0.y)); draw(); };
        const up = () => { removeEventListener('mousemove', mv); removeEventListener('mouseup', up); }; addEventListener('mousemove', mv); addEventListener('mouseup', up); draw(); panel();
      });
    }
    function panel() {
      const n = node(sel); if (!n) { side.innerHTML = '<b>Inspector</b><p class="csub">Select a node to edit its parameters. Drag to move. Click a blue port, then another node, to connect them.</p>'; return; }
      const s = specs[n.kind], stt = status[n.id];
      side.innerHTML = `<b>${esc(n.id)}</b> <span class="pill info">${esc(s.label)}</span><div class="csub">${s.inputs.join(', ') || 'no input'} → ${s.output}${s.variadic ? ' (variadic)' : ''}</div>` + s.params.map(p => `<div class="field"><label>${p.name}</label>${p.kind === 'select' ? `<select data-k="${p.name}">${p.options.map(o => `<option ${String(n.params[p.name] ?? p.default) === o ? 'selected' : ''}>${o}</option>`).join('')}</select>` : `<input data-k="${p.name}" value="${esc(n.params[p.name] ?? p.default)}">`}</div>`).join('') +
        (stt ? `<div class="callout ${stt.status === 'failed' ? 'crit' : stt.status === 'skipped' ? 'warn' : 'good'}" style="font-size:12px"><b class="t">${stt.status}${stt.key ? ' · cache key ' + stt.key : ''}</b>${esc(stt.error || stt.summary || '')}</div>` : '');
      side.querySelectorAll('[data-k]').forEach(i => i.onchange = () => { const p = s.params.find(q => q.name === i.dataset.k); n.params[p.name] = p.kind === 'number' ? +i.value : i.value; status = {}; draw(); });
    }
    const call = async path => { const r = await Kit.api(path, g); status = r.nodes || {}; draw(); panel(); out.innerHTML = ''; Kit.render(out, r.blocks); document.body.dataset.ran = '1'; };
    el.querySelector('.run').onclick = () => call('api/run'); el.querySelector('.val').onclick = () => call('api/validate');
    el.querySelector('.del').onclick = () => { if (!sel) return; g.nodes = g.nodes.filter(n => n.id !== sel); g.edges = g.edges.filter(e => e.src !== sel && e.dst !== sel); sel = null; status = {}; draw(); panel(); };
    el.querySelector('.add').onchange = e => { const k = e.target.value; if (!k) return; let id; do { id = k + '_' + seq++; } while (node(id)); g.nodes.push({ id, kind: k, x: 30 + (seq * 37) % 300, y: 20 + (seq * 53) % 250, params: {} }); e.target.value = ''; sel = id; status = {}; draw(); panel(); };
    const loadTpl = k => { g = JSON.parse(JSON.stringify(tpl[k])); sel = null; linking = null; status = {}; draw(); panel(); call('api/run'); };
    el.querySelector('.tpl').onchange = e => loadTpl(e.target.value); loadTpl(Object.keys(tpl)[0]);
  };
})();
