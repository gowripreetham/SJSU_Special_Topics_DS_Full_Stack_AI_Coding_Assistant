// NanoLlama chat studio: bubbles, sampling controls, per-reply confidence and speed.
(function () {
  const { $, esc } = Kit;
  const st = document.createElement('style'); st.textContent = `
  .chat{display:grid;grid-template-columns:minmax(0,1fr) 250px;gap:18px}.clog{height:430px;overflow:auto;background:var(--neutral);border-radius:12px;padding:14px;display:flex;flex-direction:column;gap:10px}
  .bub{max-width:78%;padding:9px 13px;border-radius:14px;line-height:1.45}.bub.u{align-self:flex-end;background:var(--accent);color:#fff;border-bottom-right-radius:4px}
  .bub.a{align-self:flex-start;background:var(--surface);border:1px solid var(--grid);border-bottom-left-radius:4px}.bub .m{font:11px var(--mono);color:var(--muted);margin-top:5px}
  .cin{display:flex;gap:8px;margin-top:10px}.cin input{flex:1;padding:10px 12px;border:1px solid var(--axis);border-radius:9px;background:var(--surface);color:var(--ink);font:inherit}
  .sug{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.sug span{cursor:pointer}@media(max-width:900px){.chat{grid-template-columns:1fr}}`; document.head.appendChild(st);
  Kit.custom.chat = function (el, props) {
    el.innerHTML = `<div class="chat"><div><div class="clog"><div class="bub a">Hi — I'm NanoLlama, an ${props.params.toLocaleString()}-parameter transformer trained from scratch on a CPU. I was fine-tuned on ${props.n} data-science concepts. Ask me to explain one!</div></div>
      <div class="cin"><input placeholder="Ask about a data-science concept…" maxlength="60"><button class="btn">Send</button></div><div class="sug"></div></div>
      <div><div class="field"><label>Temperature <span class="rv tv">0</span></label><input type="range" class="temp" min="0" max="1.5" step="0.1" value="0"></div>
      <div class="field" style="margin-top:12px"><label>Top-k <span class="rv kv">20</span></label><input type="range" class="topk" min="1" max="60" step="1" value="20"></div>
      <div class="callout" style="margin-top:14px;font-size:12.5px"><b class="t">How to read replies</b>Temperature 0 = greedy decoding (most reliable). <b>Confidence</b> is the joint probability of the first 12 characters — how sure the model was about <i>which</i> answer to give. It knows 56 concepts and nothing else: ask about anything else and it still replies with one of them, usually (not always) with a low-confidence flag.</div></div></div>`;
    const log = el.querySelector('.clog'), inp = el.querySelector('.cin input'), q = s => el.querySelector(s);
    q('.temp').oninput = e => q('.tv').textContent = e.target.value; q('.topk').oninput = e => q('.kv').textContent = e.target.value;
    const send = async (text) => {
      text = (text || inp.value).trim(); if (!text) return; inp.value = '';
      log.appendChild($('div', 'bub u', esc(text))); const b = $('div', 'bub a', '…'); log.appendChild(b); log.scrollTop = 1e9;
      try { const r = await Kit.api('api/chat', { message: text, temperature: +q('.temp').value, top_k: +q('.topk').value });
        b.innerHTML = esc(r.reply) + `<div class="m"><span class="pill ${r.confidence > .9 ? 'good' : r.confidence > .6 ? 'warn' : 'crit'}">${r.confidence > .9 ? '✓ confident' : r.confidence > .6 ? '~ unsure' : '⚠ likely out of scope'} ${(100 * r.confidence).toFixed(0)}%</span> ${r.tokens} tokens · ${r.tok_per_s.toFixed(0)} tok/s</div>`; }
      catch (e) { b.textContent = 'Error: ' + e.message; }
      log.scrollTop = 1e9; document.body.dataset.chatted = '1';
    };
    q('.cin button').onclick = () => send(); inp.onkeydown = e => { if (e.key === 'Enter') send(); };
    for (const s of props.suggestions) { const c = $('span', 'pill info', esc(s)); c.onclick = () => send(s); q('.sug').appendChild(c); }
    send(props.suggestions[0]);
  };
})();
