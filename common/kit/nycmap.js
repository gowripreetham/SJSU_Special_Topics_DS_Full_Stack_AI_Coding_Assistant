// Shared offline NYC map: draws the 263 TLC taxi zones on a canvas (no tile server needed).
// Kit.nycMap(el, {height, points, values:{zoneName:number}, onPick(pickup, dropoff), bbox}) -> {setTrip(p,d)}
Kit.nycMap = async function (el, o = {}) {
  if (!Kit._zones) Kit._zones = await (await fetch('kit/nyc_zones.json')).json();
  const zones = Kit._zones, cv = document.createElement('canvas'), H = o.height || 460;
  cv.style.cssText = `width:100%;height:${H}px;border-radius:8px;cursor:crosshair;display:block`; el.appendChild(cv);
  const bb = o.bbox || [-74.06, 40.60, -73.74, 40.89]; let pick = o.pickup || null, drop = o.dropoff || null, W = 0, hoverZ = null;
  const kx = Math.cos(40.75 * Math.PI / 180);
  const fit = () => { const sx = W / ((bb[2] - bb[0]) * kx), sy = H / (bb[3] - bb[1]), s = Math.min(sx, sy); return { s, ox: (W - (bb[2] - bb[0]) * kx * s) / 2, oy: (H - (bb[3] - bb[1]) * s) / 2 }; };
  const P = (lon, lat) => { const f = fit(); return [f.ox + (lon - bb[0]) * kx * f.s, H - f.oy - (lat - bb[1]) * f.s]; };
  const inv = (x, y) => { const f = fit(); return [bb[0] + (x - f.ox) / (kx * f.s), bb[1] + (H - f.oy - y) / f.s]; };
  const vals = o.values || null; let lo = 0, hi = 1; if (vals) { const v = Object.values(vals).filter(x => x != null); lo = Math.min(...v); hi = Math.max(...v); }
  function draw() {
    const dpr = devicePixelRatio || 1; W = cv.clientWidth; cv.width = W * dpr; cv.height = H * dpr; const c = cv.getContext('2d'); c.scale(dpr, dpr);
    c.fillStyle = Kit.css('--neutral'); c.fillRect(0, 0, W, H);
    for (const z of zones) {
      c.beginPath(); for (const r of z.rings) { r.forEach(([lo2, la], i) => { const [x, y] = P(lo2, la); i ? c.lineTo(x, y) : c.moveTo(x, y); }); c.closePath(); }
      c.fillStyle = vals && vals[z.zone] != null ? Kit.seqColor((vals[z.zone] - lo) / ((hi - lo) || 1)) : Kit.css('--surface'); c.fill();
      c.strokeStyle = z === hoverZ ? Kit.css('--accent') : Kit.css('--axis'); c.lineWidth = z === hoverZ ? 1.6 : .6; c.stroke();
    }
    if (o.points) { c.fillStyle = Kit.css('--s1') + '55'; for (const [lo2, la] of o.points) { const [x, y] = P(lo2, la); c.fillRect(x - 1, y - 1, 2, 2); } }
    if (o.clusters) o.clusters.forEach((k, i) => { const [x, y] = P(k[0], k[1]); c.beginPath(); c.arc(x, y, 7, 0, 7); c.fillStyle = Kit.SERIES()[i % 8]; c.fill(); c.strokeStyle = '#fff'; c.lineWidth = 2; c.stroke(); });
    if (pick && drop) { const a = P(...pick), b = P(...drop); c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.strokeStyle = Kit.css('--ink'); c.setLineDash([6, 5]); c.lineWidth = 1.5; c.stroke(); c.setLineDash([]); }
    for (const [pt, col, lab] of [[pick, Kit.css('--s3'), 'A'], [drop, Kit.css('--s2'), 'B']]) if (pt) { const [x, y] = P(...pt); c.beginPath(); c.arc(x, y, 10, 0, 7); c.fillStyle = col; c.fill(); c.strokeStyle = '#fff'; c.lineWidth = 2; c.stroke(); c.fillStyle = '#fff'; c.font = '700 11px system-ui'; c.textAlign = 'center'; c.textBaseline = 'middle'; c.fillText(lab, x, y + .5); }
  }
  const zoneAt = (lon, lat) => { const c = document.createElement('canvas').getContext('2d'); for (const z of zones) for (const r of z.rings) { c.beginPath(); r.forEach(([a, b], i) => i ? c.lineTo(a, b) : c.moveTo(a, b)); c.closePath(); if (c.isPointInPath(lon, lat)) return z; } return null; };
  let next = 'pick';
  cv.addEventListener('click', e => { if (!o.onPick) return; const r = cv.getBoundingClientRect(), ll = inv(e.clientX - r.left, e.clientY - r.top); if (next === 'pick') { pick = ll; drop = null; next = 'drop'; } else { drop = ll; next = 'pick'; } draw(); o.onPick(pick, drop); });
  cv.addEventListener('mousemove', e => { const r = cv.getBoundingClientRect(), ll = inv(e.clientX - r.left, e.clientY - r.top), z = zoneAt(...ll); if (z !== hoverZ) { hoverZ = z; draw(); } cv.title = z ? `${z.zone} · ${z.borough}` + (vals && vals[z.zone] != null ? ` · ${Kit.fmt(vals[z.zone])}` : '') : ''; });
  new ResizeObserver(draw).observe(cv); draw();
  return { setTrip(p, d) { pick = p; drop = d; next = 'pick'; draw(); }, zoneAt };
};
