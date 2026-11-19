// Trip estimator: click the map to set pickup (A) and drop-off (B); the model answers live.
(function () {
  const PRESETS = { 'Times Sq → JFK Airport': [[-73.9855, 40.7580], [-73.7781, 40.6413]], 'Wall St → Central Park': [[-74.0090, 40.7060], [-73.9712, 40.7740]], 'LaGuardia → Midtown': [[-73.8740, 40.7769], [-73.9840, 40.7549]], 'Williamsburg → Union Sq': [[-73.9571, 40.7081], [-73.9903, 40.7359]], 'Upper East → Upper West': [[-73.9566, 40.7736], [-73.9754, 40.7870]] };
  const DOW = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const load = () => Kit.nycMap ? Promise.resolve() : new Promise(ok => { const s = document.createElement('script'); s.src = 'kit/nycmap.js'; s.onload = ok; document.head.appendChild(s); });
  Kit.custom.trip = async function (el, props) {
    await load();
    el.innerHTML = `<div class="grid g2" style="grid-template-columns:minmax(0,1.25fr) minmax(0,1fr)"><div><div class="mapbox"></div><div class="csub" style="margin-top:6px">Click once for pickup <b>A</b>, again for drop-off <b>B</b>. Blue dots = real 2016 pickups from the dataset.</div></div>
      <div><div class="form" style="grid-template-columns:1fr 1fr">
      <div class="field wide"><label>Route preset</label><select class="pre"><option value="">— click the map or choose —</option>${Object.keys(PRESETS).map(k => `<option>${k}</option>`).join('')}</select></div>
      <div class="field"><label>Day</label><select class="dow">${DOW.map((d, i) => `<option value="${i}" ${i === 2 ? 'selected' : ''}>${d}</option>`).join('')}</select></div>
      <div class="field"><label>Passengers</label><select class="pax">${[1, 2, 3, 4, 5, 6].map(i => `<option>${i}</option>`).join('')}</select></div>
      <div class="field wide"><label>Pickup hour <span class="rv hv">17:00</span></label><input type="range" class="hr" min="0" max="23.75" step="0.25" value="17"></div></div>
      <div class="out" style="margin-top:14px"><div class="callout">Pick a route to get a live estimate.</div></div></div></div><div class="out2" style="margin-top:16px"></div>`;
    const q = s => el.querySelector(s); let pick = null, drop = null;
    const run = async () => {
      if (!pick || !drop) return;
      const r = await Kit.api('api/predict', { pickup_lon: pick[0], pickup_lat: pick[1], dropoff_lon: drop[0], dropoff_lat: drop[1], hour: +q('.hr').value, dow: +q('.dow').value, passengers: +q('.pax').value });
      q('.out').innerHTML = ''; Kit.render(q('.out'), r.blocks); q('.out2').innerHTML = ''; Kit.render(q('.out2'), r.detail); document.body.dataset.predicted = '1';
    };
    const map = await Kit.nycMap(q('.mapbox'), { points: props.points, height: 470, onPick: (p, d) => { pick = p; drop = d; q('.pre').value = ''; run(); } });
    q('.pre').onchange = e => { const p = PRESETS[e.target.value]; if (!p) return; [pick, drop] = p; map.setTrip(pick, drop); run(); };
    q('.hr').oninput = e => { const h = +e.target.value; q('.hv').textContent = String(Math.floor(h)).padStart(2, '0') + ':' + String(Math.round((h % 1) * 60)).padStart(2, '0'); };
    q('.hr').onchange = run; q('.dow').onchange = run; q('.pax').onchange = run;
    q('.pre').value = 'Times Sq → JFK Airport'; q('.pre').onchange({ target: q('.pre') });
  };
})();
