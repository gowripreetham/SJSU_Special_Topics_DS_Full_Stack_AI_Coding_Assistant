// Fare estimator (click two zones) and zone choropleth, both on the shared offline NYC map.
(function () {
  const load = () => Kit.nycMap ? Promise.resolve() : new Promise(ok => { const s = document.createElement('script'); s.src = 'kit/nycmap.js'; s.onload = ok; document.head.appendChild(s); });
  const DOW = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const PRE = { 'Midtown → JFK Airport': [[-73.984, 40.755], [-73.7781, 40.6413]], 'Upper East Side → SoHo': [[-73.957, 40.774], [-74.001, 40.723]], 'LaGuardia → Times Sq': [[-73.874, 40.777], [-73.9855, 40.758]], 'Williamsburg → Astoria': [[-73.957, 40.714], [-73.92, 40.765]] };
  Kit.custom.fare = async function (el) {
    await load();
    el.innerHTML = `<div class="grid g2" style="grid-template-columns:minmax(0,1.2fr) minmax(0,1fr)"><div><div class="mapbox"></div><div class="csub" style="margin-top:6px">Click a pickup zone (A), then a drop-off zone (B). Hover to see zone names.</div></div><div><div class="form" style="grid-template-columns:1fr 1fr">
      <div class="field wide"><label>Route preset</label><select class="pre"><option value="">— click the map or choose —</option>${Object.keys(PRE).map(k => `<option>${k}</option>`).join('')}</select></div>
      <div class="field"><label>Day</label><select class="dow">${DOW.map((d, i) => `<option value="${i}" ${i === 4 ? 'selected' : ''}>${d}</option>`).join('')}</select></div><div class="field"><label>Cab</label><select class="col"><option>yellow</option><option>green</option></select></div>
      <div class="field wide"><label>Pickup hour <span class="rv hv">18</span></label><input type="range" class="hr" min="0" max="23" step="1" value="18"></div>
      <div class="field wide"><label>Route distance (miles) <span class="rv dv">auto</span></label><input type="range" class="dist" min="0" max="30" step="0.1" value="0"></div></div><div class="out" style="margin-top:14px"></div></div></div><div class="out2" style="margin-top:16px"></div>`;
    const q = s => el.querySelector(s); let pick = null, drop = null, map;
    const run = async () => {
      if (!pick || !drop) return; const zp = map.zoneAt(...pick), zd = map.zoneAt(...drop);
      if (!zp || !zd) { q('.out').innerHTML = '<div class="callout warn">Both points must be inside a taxi zone.</div>'; return; }
      const r = await Kit.api('api/predict', { pickup_zone: zp.zone, dropoff_zone: zd.zone, hour: +q('.hr').value, dow: +q('.dow').value, color: q('.col').value, distance: +q('.dist').value || null });
      q('.dv').textContent = r.distance.toFixed(1) + (+q('.dist').value ? '' : ' (auto)'); q('.out').innerHTML = ''; Kit.render(q('.out'), r.blocks); q('.out2').innerHTML = ''; Kit.render(q('.out2'), r.detail);
    };
    map = await Kit.nycMap(q('.mapbox'), { height: 470, onPick: (p, d) => { pick = p; drop = d; q('.pre').value = ''; q('.dist').value = 0; run(); } });
    q('.pre').onchange = e => { const p = PRE[e.target.value]; if (!p) return; [pick, drop] = p; map.setTrip(pick, drop); q('.dist').value = 0; run(); };
    q('.hr').oninput = e => q('.hv').textContent = e.target.value;
    for (const s of ['.hr', '.dow', '.col', '.dist']) q(s).onchange = run;
    q('.pre').value = 'Midtown → JFK Airport'; q('.pre').onchange({ target: q('.pre') });
  };
  Kit.custom.choropleth = async function (el, props) { await load(); await Kit.nycMap(el, { height: props.height || 420, values: props.values, clusters: props.clusters }); const d = document.createElement('div'); d.className = 'csub'; d.style.marginTop = '6px'; d.innerHTML = props.note || ''; el.appendChild(d); };
})();
