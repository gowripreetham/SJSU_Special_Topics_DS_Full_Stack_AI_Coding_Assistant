(function () {
  const { $, esc } = Kit;
  // Basket builder: click items to add them; recommendations refresh live.
  Kit.custom.basket = function (el, props) {
    const sel = new Set(props.start || []);
    el.innerHTML = `<div class="csub">Click products to build a basket (${props.items.length} most frequent of 169 shown).</div><div class="chips" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px"></div><div class="out"></div>`;
    const chips = el.querySelector('.chips'), out = el.querySelector('.out');
    const run = async () => { const r = await Kit.api('api/recommend', { items: [...sel] }); out.innerHTML = ''; Kit.render(out, r.blocks); };
    for (const it of props.items) {
      const c = $('span', 'pill', esc(it)); c.style.cssText = 'cursor:pointer;padding:5px 11px;font-size:12.5px;border:1px solid var(--grid)';
      const paint = () => { c.className = 'pill' + (sel.has(it) ? ' info' : ''); c.style.borderColor = sel.has(it) ? 'var(--accent)' : 'var(--grid)'; };
      c.onclick = () => { sel.has(it) ? sel.delete(it) : sel.add(it); paint(); run(); }; paint(); chips.appendChild(c);
    }
    run();
  };
  // Rule network: items are nodes, rules are arrows (antecedent item -> consequent); simple force layout.
  Kit.custom.rulegraph = function (el, props) {
    const cv = $('canvas'); const H = 560; cv.style.cssText = `width:100%;height:${H}px`; el.appendChild(cv);
    const nodes = {}, links = [];
    for (const e of props.edges) for (const a of e.a) { nodes[a] = nodes[a] || { id: a, deg: 0 }; nodes[e.c] = nodes[e.c] || { id: e.c, deg: 0 }; nodes[a].deg++; nodes[e.c].deg++; links.push({ s: nodes[a], t: nodes[e.c], lift: e.lift }); }
    const N = Object.values(nodes); let W = 900, seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    N.forEach((n, i) => { const a = 2 * Math.PI * i / N.length; n.x = 450 + 220 * Math.cos(a) + rnd() * 20; n.y = H / 2 + 200 * Math.sin(a) + rnd() * 20; n.vx = n.vy = 0; });
    for (let it = 0; it < 420; it++) {
      for (const a of N) for (const b of N) if (a !== b) { const dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 40, f = 15000 / d2; a.vx += dx / Math.sqrt(d2) * f; a.vy += dy / Math.sqrt(d2) * f; }
      for (const l of links) { const dx = l.t.x - l.s.x, dy = l.t.y - l.s.y, d = Math.sqrt(dx * dx + dy * dy) + .1, f = (d - 160) * .012; l.s.vx += dx / d * f; l.s.vy += dy / d * f; l.t.vx -= dx / d * f; l.t.vy -= dy / d * f; }
      for (const n of N) { n.vx += (450 - n.x) * .004; n.vy += (H / 2 - n.y) * .006; n.x += Math.max(-8, Math.min(8, n.vx)); n.y += Math.max(-8, Math.min(8, n.vy)); n.vx *= .55; n.vy *= .55; n.x = Math.max(60, Math.min(840, n.x)); n.y = Math.max(24, Math.min(H - 24, n.y)); }
    }
    const lmax = Math.max(...links.map(l => l.lift)), lmin = Math.min(...links.map(l => l.lift));
    function draw() {
      const dpr = devicePixelRatio || 1; W = cv.clientWidth; cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext('2d'); c.scale(dpr, dpr); const k = W / 900;
      for (const l of links) { const t = (l.lift - lmin) / ((lmax - lmin) || 1); c.strokeStyle = Kit.seqColor(.25 + .75 * t); c.lineWidth = .8 + 2.2 * t; c.beginPath(); c.moveTo(l.s.x * k, l.s.y); c.lineTo(l.t.x * k, l.t.y); c.stroke();
        const ang = Math.atan2(l.t.y - l.s.y, (l.t.x - l.s.x) * k), r = 8 + Math.sqrt(l.t.deg) * 2.5, ax = l.t.x * k - Math.cos(ang) * r, ay = l.t.y - Math.sin(ang) * r; c.fillStyle = c.strokeStyle; c.beginPath(); c.moveTo(ax, ay); c.lineTo(ax - 8 * Math.cos(ang - .4), ay - 8 * Math.sin(ang - .4)); c.lineTo(ax - 8 * Math.cos(ang + .4), ay - 8 * Math.sin(ang + .4)); c.fill(); }
      for (const n of N) { const r = 6 + Math.sqrt(n.deg) * 2.5; c.beginPath(); c.arc(n.x * k, n.y, r, 0, 7); c.fillStyle = Kit.css('--s2'); c.fill(); c.strokeStyle = Kit.css('--surface'); c.lineWidth = 2; c.stroke(); c.fillStyle = Kit.css('--ink'); c.font = '11.5px system-ui'; c.textAlign = 'center'; c.fillText(n.id, n.x * k, n.y - r - 4); }
    }
    new ResizeObserver(draw).observe(cv); draw();
  };
})();
