// Four live simulators. Everything runs in the browser, so this project also works as a static GitHub Pages site.
(function () {
  const { $, esc, fmt } = Kit;
  const plot = (el, blocks) => { el.querySelectorAll('canvas').forEach(c => c._chart && c._chart.destroy()); el.innerHTML = ''; Kit.render(el, blocks); };
  const slider = (label, id, min, max, step, val) => `<div class="field"><label>${label} <span class="rv" data-v="${id}">${val}</span></label><input type="range" data-id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`;
  const wire = (el, fn) => { el.querySelectorAll('[data-id]').forEach(i => i.addEventListener('input', () => { const v = el.querySelector(`[data-v="${i.dataset.id}"]`); if (v) v.textContent = i.value; fn(); })); };
  const val = (el, id) => { const i = el.querySelector(`[data-id="${id}"]`); return i.type === 'range' || i.type === 'number' ? +i.value : i.value; };
  const N = (x, m, s) => Math.exp(-.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  const erf = x => { const t = 1 / (1 + .3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - .284496736) * t + .254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const cdf = (x, m, s) => .5 * (1 + erf((x - m) / (s * Math.SQRT2)));

  // ---------------------------------------------------------------- 1. Naive Bayes
  const TRAIN = [['spam', 'win free money now'], ['spam', 'free prize claim now'], ['spam', 'urgent win cash prize'], ['spam', 'claim free cash offer'], ['spam', 'limited offer win big'],
  ['ham', 'meeting moved to monday'], ['ham', 'lunch tomorrow at noon'], ['ham', 'project report due monday'], ['ham', 'can we meet tomorrow'], ['ham', 'free for lunch now'], ['ham', 'send the report today'], ['ham', 'call me after the meeting']];
  Kit.custom.nb = function (el) {
    el.innerHTML = `<div class="form">${`<div class="field wide"><label>Type a message</label><input type="text" data-id="msg" value="free cash prize for the meeting"></div>`}${slider('Laplace smoothing α', 'alpha', 0, 3, .1, 1)}${slider('Prior P(spam)', 'prior', .05, .95, .05, .4)}</div><div class="out" style="margin-top:16px"></div>
      <details style="margin-top:10px"><summary style="cursor:pointer;color:var(--ink2)">Show the 12 training messages</summary><table class="t" style="margin-top:8px">${TRAIN.map(([c, t]) => `<tr><td><span class="pill ${c === 'spam' ? 'crit' : 'good'}">${c}</span></td><td>${t}</td></tr>`).join('')}</table></details>`;
    const cnt = { spam: {}, ham: {} }, tot = { spam: 0, ham: 0 }, V = new Set();
    for (const [c, t] of TRAIN) for (const w of t.split(' ')) { cnt[c][w] = (cnt[c][w] || 0) + 1; tot[c]++; V.add(w); }
    const run = () => {
      const a = val(el, 'alpha'), pr = val(el, 'prior'), words = val(el, 'msg').toLowerCase().split(/[^a-z]+/).filter(Boolean);
      let ls = Math.log(pr), lh = Math.log(1 - pr); const rows = [];
      for (const w of words) {
        const known = V.has(w), ps = ((cnt.spam[w] || 0) + a) / (tot.spam + a * V.size), ph = ((cnt.ham[w] || 0) + a) / (tot.ham + a * V.size);
        if (!known) { rows.push([w, '—', '—', 'ignored (never seen)']); continue; }
        ls += Math.log(ps); lh += Math.log(ph); rows.push([w, +ps.toFixed(4), +ph.toFixed(4), ps === 0 || ph === 0 ? '−∞ / +∞  (zero-frequency problem!)' : +Math.log(ps / ph).toFixed(2)]);
      }
      const p = isFinite(ls) || isFinite(lh) ? 1 / (1 + Math.exp(lh - ls)) : NaN;
      plot(el.querySelector('.out'), [
        { t: 'kpis', items: [{ label: 'P(spam | message)', value: isNaN(p) ? 'undefined' : (100 * p).toFixed(1) + '%' }, { label: 'Verdict', value: isNaN(p) ? '0/0' : p > .5 ? 'SPAM' : 'HAM' }, { label: 'log P(msg, spam)', value: isFinite(ls) ? +ls.toFixed(2) : '−∞' }, { label: 'log P(msg, ham)', value: isFinite(lh) ? +lh.toFixed(2) : '−∞' }] },
        { t: 'grid', cols: 2, items: [{ t: 'table', title: 'Per-word evidence', cols: ['word', 'P(w | spam)', 'P(w | ham)', 'log-odds vote'], rows },
        { t: 'chart', kind: 'hbar', title: 'Each word votes (log P(w|spam)/P(w|ham))', height: 240, labels: rows.filter(r => typeof r[3] === 'number').map(r => r[0]), series: [{ name: 'log-odds', data: rows.filter(r => typeof r[3] === 'number').map(r => r[3]), colors: rows.filter(r => typeof r[3] === 'number').map(r => r[3] > 0 ? '8' : '3') }], note: 'Red bars push toward spam, green toward ham. The posterior is just the prior log-odds plus the sum of these votes.' }] },
        a === 0 ? { t: 'callout', kind: 'crit', title: 'α = 0: the zero-frequency problem', text: 'A single word never seen in one class makes that class probability exactly 0 and vetoes all other evidence. Laplace smoothing (α > 0) adds pseudo-counts so no probability is ever zero.' } : null]);
    };
    wire(el, run); run();
  };

  // ---------------------------------------------------------------- 2. Evaluation: threshold, confusion matrix, ROC, PR, cost
  Kit.custom.evalsim = function (el) {
    el.innerHTML = `<div class="form">${slider('Decision threshold', 'thr', 0, 1, .01, .5)}${slider('Class separation (model quality)', 'sep', 0, .6, .02, .3)}${slider('Prevalence of positives', 'prev', .02, .8, .02, .3)}${slider('Cost of a false positive', 'cfp', 1, 20, 1, 1)}${slider('Cost of a false negative', 'cfn', 1, 20, 1, 5)}</div><div class="out" style="margin-top:16px"></div>`;
    const run = () => {
      const thr = val(el, 'thr'), sep = val(el, 'sep'), prev = val(el, 'prev'), cfp = val(el, 'cfp'), cfn = val(el, 'cfn'), s = .15, mn = .5 - sep / 2, mp = .5 + sep / 2, n = 1000;
      const at = t => { const tpr = 1 - cdf(t, mp, s), fpr = 1 - cdf(t, mn, s), tp = tpr * prev * n, fn = (1 - tpr) * prev * n, fp = fpr * (1 - prev) * n, tn = (1 - fpr) * (1 - prev) * n; return { tpr, fpr, tp, fn, fp, tn, prec: tp / Math.max(1e-9, tp + fp), cost: fp * cfp + fn * cfn }; };
      const m = at(thr), xs = Array.from({ length: 101 }, (_, i) => i / 100), all = xs.map(at), best = xs[all.reduce((b, a, i) => a.cost < all[b].cost ? i : b, 0)];
      let auc = 0; for (let i = 1; i < all.length; i++) auc += (all[i - 1].fpr - all[i].fpr) * (all[i].tpr + all[i - 1].tpr) / 2;
      const f1 = 2 * m.prec * m.tpr / Math.max(1e-9, m.prec + m.tpr), cell = (lab, v, sub, kind) => `<div class="kpi" style="border-left:4px solid var(--${kind})"><div class="l">${lab}</div><div class="v">${v.toFixed(0)}</div><div class="h">${sub}</div></div>`;
      plot(el.querySelector('.out'), [
        { t: 'grid', cols: 2, items: [
          { t: 'chart', kind: 'line', title: 'Score distributions and the threshold', height: 250, labels: xs.map(x => x.toFixed(2)), x: 'model score', series: [{ name: 'actual negatives', data: xs.map(x => N(x, mn, s) * (1 - prev)), fill: 'origin', color: '1' }, { name: 'actual positives', data: xs.map(x => N(x, mp, s) * prev), fill: 'origin', color: '2' }, { name: `threshold ${thr.toFixed(2)}`, data: xs.map(x => Math.abs(x - thr) < .005 ? Math.max(N(.5, mn, s), N(.5, mp, s)) * 1.05 : null), color: '--ink', extra: { type: 'bar', barThickness: 2 } }], note: 'Everything right of the line is predicted positive.' },
          { t: 'html', title: 'Confusion matrix (per 1,000 cases)', html: `<div class="kpis" style="grid-template-columns:1fr 1fr">${cell('True positives', m.tp, 'hit', 'good')}${cell('False negatives — Type II error', m.fn, 'miss · costs ' + cfn + ' each', 'crit')}${cell('False positives — Type I error', m.fp, 'false alarm · costs ' + cfp + ' each', 'warn')}${cell('True negatives', m.tn, 'correct rejection', 'good')}</div>` }] },
        { t: 'kpis', items: [{ label: 'Precision', value: m.prec, digits: 3, hint: 'TP / (TP + FP)' }, { label: 'Recall (TPR)', value: m.tpr, digits: 3, hint: 'TP / (TP + FN)' }, { label: 'False-positive rate', value: m.fpr, digits: 3, hint: 'FP / (FP + TN) = α' }, { label: 'F1', value: f1, digits: 3 }, { label: 'Accuracy', value: (m.tp + m.tn) / n, digits: 3 }, { label: 'ROC-AUC', value: auc, digits: 3, hint: 'threshold-free' }] },
        { t: 'grid', cols: 3, items: [
          { t: 'chart', kind: 'scatter', title: 'ROC curve', height: 230, x: 'false-positive rate', y: 'true-positive rate', series: [{ name: 'ROC', data: all.map(a => ({ x: a.fpr, y: a.tpr })), line: true }, { name: 'you are here', data: [{ x: m.fpr, y: m.tpr }], r: 7, color: '2' }, { name: '_chance', data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], line: true, color: '--axis' }] },
          { t: 'chart', kind: 'scatter', title: 'Precision–recall trade-off', height: 230, x: 'recall', y: 'precision', ymin: 0, ymax: 1, series: [{ name: 'PR', data: all.filter(a => a.tp + a.fp > .5).map(a => ({ x: a.tpr, y: a.prec })), line: true, color: '3' }, { name: 'you are here', data: [{ x: m.tpr, y: m.prec }], r: 7, color: '2' }] },
          { t: 'chart', kind: 'line', title: `Total cost by threshold (optimum ${best.toFixed(2)})`, height: 230, labels: xs.map(x => x.toFixed(2)), x: 'threshold', series: [{ name: 'expected cost', data: all.map(a => a.cost), color: '8' }], note: `Your threshold costs **${m.cost.toFixed(0)}**; the best achievable is **${Math.min(...all.map(a => a.cost)).toFixed(0)}**.` }] }]);
    };
    wire(el, run); run();
  };

  // ---------------------------------------------------------------- 3. Derivatives -> gradient descent
  const FNS = { 'x²  (convex bowl)': [x => x * x, x => 2 * x, -3, 3], 'x⁴ − 3x² + x  (two valleys)': [x => x ** 4 - 3 * x * x + x, x => 4 * x ** 3 - 6 * x + 1, -2.2, 2.2], '|x|-like: √(x² + 0.05)': [x => Math.sqrt(x * x + .05), x => x / Math.sqrt(x * x + .05), -3, 3] };
  Kit.custom.gdsim = function (el) {
    el.innerHTML = `<div class="form"><div class="field"><label>Function f(x)</label><select data-id="fn">${Object.keys(FNS).map((k, i) => `<option value="${i}">${k}</option>`).join('')}</select></div>${slider('Start x₀', 'x0', -2, 2, .1, -1.8)}${slider('Learning rate η', 'lr', .01, 1.1, .01, .1)}${slider('Secant gap h (→ 0 gives the derivative)', 'h', .01, 1.5, .01, .8)}
      <div class="field"><button class="btn" data-a="step">Take one step</button></div><div class="field"><button class="btn ghost" data-a="run">Run 25 steps</button></div><div class="field"><button class="btn ghost" data-a="reset">Reset</button></div></div><div class="out" style="margin-top:16px"></div>`;
    let path = [];
    const draw = () => {
      const [f, df, lo, hi] = Object.values(FNS)[+val(el, 'fn')], h = val(el, 'h'), lr = val(el, 'lr'); if (!path.length) path = [val(el, 'x0')];
      const x = path[path.length - 1], xs = Array.from({ length: 121 }, (_, i) => lo + (hi - lo) * i / 120), sec = (f(x + h) - f(x)) / h, ys = xs.map(f), ymax = Math.max(...ys), ymin = Math.min(...ys), clip = v => v > ymax * 1.3 || v < ymin - 1 ? null : v;
      const div = !isFinite(x) || Math.abs(x) > 50;
      plot(el.querySelector('.out'), [
        { t: 'grid', cols: 2, items: [
          { t: 'chart', kind: 'scatter', title: 'f(x), the secant and the tangent', height: 320, x: 'x', y: 'f(x)', xmin: lo, xmax: hi, series: [{ name: 'f(x)', data: xs.map(v => ({ x: v, y: f(v) })), line: true }, { name: `secant slope ${sec.toFixed(3)}`, data: xs.map(v => ({ x: v, y: clip(f(x) + sec * (v - x)) })), line: true, color: '4', extra: { borderDash: [5, 4] } }, { name: `tangent slope f′(x) = ${df(x).toFixed(3)}`, data: xs.map(v => ({ x: v, y: clip(f(x) + df(x) * (v - x)) })), line: true, color: '2' }, { name: 'descent path', data: path.filter(p => p >= lo && p <= hi).map(p => ({ x: p, y: f(p) })), r: 5, color: '8', extra: { showLine: true, borderWidth: 1, borderColor: Kit.css('--s8') } }] },
          { t: 'chart', kind: 'line', title: 'Loss f(xₜ) per step', height: 320, labels: path.map((_, i) => i), x: 'step t', y: 'f(x)', series: [{ name: 'f(xₜ)', data: path.map(p => isFinite(f(p)) && Math.abs(f(p)) < 1e6 ? f(p) : null), points: true, color: '8' }] }] },
        { t: 'kpis', items: [{ label: 'Current x', value: div ? 'diverged' : +x.toFixed(4) }, { label: 'f′(x) — exact', value: div ? '—' : +df(x).toFixed(4) }, { label: `Secant slope (h = ${h})`, value: div ? '—' : +sec.toFixed(4), hint: 'error ' + (div ? '—' : Math.abs(sec - df(x)).toFixed(4)) }, { label: 'Next step', value: div ? '—' : `x − η·f′(x) = ${(x - lr * df(x)).toFixed(4)}` }, { label: 'Steps taken', value: path.length - 1 }] },
        div ? { t: 'callout', kind: 'crit', title: 'Diverged!', text: 'The learning rate is so large that each step overshoots the minimum by more than it gained. For f(x) = x², any η > 1 diverges because xₜ₊₁ = (1 − 2η)xₜ.' } : null]);
    };
    const step = () => { const [, df] = Object.values(FNS)[+val(el, 'fn')], x = path[path.length - 1]; if (isFinite(x) && Math.abs(x) < 1e6) path.push(x - val(el, 'lr') * df(x)); };
    el.querySelectorAll('[data-a]').forEach(b => b.onclick = () => { if (b.dataset.a === 'reset') path = []; else for (let i = 0; i < (b.dataset.a === 'run' ? 25 : 1); i++) step(); draw(); });
    el.querySelectorAll('[data-id]').forEach(i => i.addEventListener('input', () => { const v = el.querySelector(`[data-v="${i.dataset.id}"]`); if (v) v.textContent = i.value; if (i.dataset.id === 'x0' || i.dataset.id === 'fn') path = []; draw(); }));
    draw();
  };

  // ---------------------------------------------------------------- 4. Chain rule -> backpropagation (one neuron, drawn as a graph)
  Kit.custom.backprop = function (el) {
    el.innerHTML = `<div class="form">${slider('input x', 'x', -3, 3, .1, 1.5)}${slider('weight w', 'w', -3, 3, .05, .8)}${slider('bias b', 'b', -3, 3, .05, -.5)}${slider('target t', 't', 0, 1, .05, 1)}${slider('learning rate η', 'lr', .1, 5, .1, 1)}<div class="field"><button class="btn" data-a="step">Gradient step on w, b</button></div></div>
      <div class="gr" style="margin-top:16px;overflow:auto"></div><div class="out" style="margin-top:14px"></div>`;
    let hist = [];
    const set = (id, v) => { const i = el.querySelector(`[data-id="${id}"]`); i.value = v; el.querySelector(`[data-v="${id}"]`).textContent = (+i.value).toFixed(2); };
    const run = () => {
      const x = val(el, 'x'), w = val(el, 'w'), b = val(el, 'b'), t = val(el, 't');
      const z = w * x + b, a = 1 / (1 + Math.exp(-z)), L = (a - t) ** 2, dLda = 2 * (a - t), dadz = a * (1 - a), dLdz = dLda * dadz, dLdw = dLdz * x, dLdb = dLdz, dLdx = dLdz * w;
      const node = (X, Y, lab, v, g, col) => `<g><rect x="${X - 62}" y="${Y - 30}" width="124" height="60" rx="10" fill="var(--surface)" stroke="var(--${col || 'axis'})" stroke-width="1.5"/><text x="${X}" y="${Y - 10}" text-anchor="middle" font-size="12" fill="var(--ink2)">${lab}</text><text x="${X}" y="${Y + 6}" text-anchor="middle" font-size="13" font-weight="700" fill="var(--ink)">${v}</text>${g != null ? `<text x="${X}" y="${Y + 22}" text-anchor="middle" font-size="11.5" fill="var(--s8)">∂L/∂ = ${g}</text>` : ''}</g>`;
      const edge = (x1, y1, x2, y2, lab, back) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="var(--axis)" stroke-width="1.5" marker-end="url(#ar)"/><text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 7}" text-anchor="middle" font-size="11" fill="var(--s1)">${lab || ''}</text>${back ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 + 15}" text-anchor="middle" font-size="11" fill="var(--s8)">← × ${back}</text>` : ''}`;
      const f = v => (+v).toFixed(3);
      el.querySelector('.gr').innerHTML = `<svg viewBox="0 0 980 250" style="width:100%;min-width:760px;font-family:system-ui"><defs><marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="var(--axis)"/></marker></defs>
        ${edge(132, 55, 268, 110, '', 'w')}${edge(132, 125, 268, 125, '', 'x')}${edge(132, 195, 268, 140, '', '1')}${edge(394, 125, 478, 125, 'σ(z)', f(dadz))}${edge(604, 125, 688, 125, '(a − t)²', f(dLda))}${edge(814, 195, 760, 158, '')}
        ${node(70, 55, 'input x', f(x), f(dLdx))}${node(70, 125, 'weight w', f(w), f(dLdw), 's2')}${node(70, 195, 'bias b', f(b), f(dLdb), 's2')}${node(331, 125, 'z = w·x + b', f(z), f(dLdz))}${node(541, 125, 'a = σ(z)', f(a), f(dLda))}${node(751, 125, 'L = (a − t)²', f(L), '1', 's8')}${node(880, 210, 'target t', f(t))}</svg>`;
      plot(el.querySelector('.out'), [
        { t: 'md', title: 'The chain rule, multiplied out with the live numbers', text: `**Forward pass** (blue, left → right): z = ${f(w)}·${f(x)} + ${f(b)} = **${f(z)}**, a = σ(z) = **${f(a)}**, L = (a − t)² = **${f(L)}**\n\n**Backward pass** (red, right → left) — each node multiplies the gradient arriving from the right by its own *local* derivative:\n\n- ∂L/∂a = 2(a − t) = **${f(dLda)}**\n- ∂L/∂z = ∂L/∂a · σ′(z) = ${f(dLda)} × a(1 − a) = ${f(dLda)} × ${f(dadz)} = **${f(dLdz)}**\n- ∂L/∂w = ∂L/∂z · x = ${f(dLdz)} × ${f(x)} = **${f(dLdw)}**\n- ∂L/∂b = ∂L/∂z · 1 = **${f(dLdb)}**\n\nBackpropagation is nothing more than this bookkeeping, done once per node so shared factors (like ∂L/∂z) are never recomputed.` },
        Math.abs(dadz) < .02 ? { t: 'callout', kind: 'warn', title: 'Vanishing gradient', text: `σ′(z) = ${f(dadz)} — the sigmoid is saturated, so almost no gradient reaches w and b even though the loss is ${f(L)}. This is why deep nets moved to ReLU-family activations and careful initialisation.` } : null,
        hist.length > 1 ? { t: 'chart', kind: 'line', title: 'Loss after each gradient step', height: 200, labels: hist.map((_, i) => i), series: [{ name: 'L', data: hist, points: true, color: '8' }] } : null]);
      return { L, dLdw, dLdb };
    };
    el.querySelector('[data-a]').onclick = () => { const r = run(); if (!hist.length) hist.push(r.L); set('w', val(el, 'w') - val(el, 'lr') * r.dLdw); set('b', val(el, 'b') - val(el, 'lr') * r.dLdb); hist.push(run().L); run(); };
    wire(el, () => { hist = []; run(); }); run();
  };
})();
