"""人工精校台

产出一个**单文件、零依赖、可离线打开**的 HTML。设计目标只有一个：
让人工只处理机器不敢定的那部分，并且每条都能在 15 秒内做完决定。

界面约定：
  · 默认只显示悬置条目（机器已定的折叠在"已决"页签）
  · 键盘流：J/K 移动，1 取底本，2 取今本，3 两存，Enter 存档
  · 决策写入 localStorage 并可导出 decisions.json，回灌流水线重出定本
"""

from __future__ import annotations

import html
import json


def build_review_app(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    title = html.escape(payload.get("work", "古籍校勘"))
    return """<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ · 人工精校台</title>
<style>
:root{--ink:#17150f;--paper:#efe9dc;--card:#fdfbf5;--rule:#cdc2a8;
 --red:#9c3b2e;--blue:#274b63;--green:#3f6b4a;--mut:#6f6555}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
 font-family:"Noto Sans CJK SC",system-ui,sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--paper);
 border-bottom:1px solid var(--rule);padding:14px 20px;display:flex;
 gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;letter-spacing:.24em;font-weight:700}
.tabs{display:flex;gap:4px}
.tab{border:1px solid var(--rule);background:transparent;padding:5px 12px;
 font-size:12px;cursor:pointer;border-radius:2px;color:var(--mut)}
.tab.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.spacer{flex:1}
.stat{font-size:12px;color:var(--mut);letter-spacing:.05em}
button.act{border:1px solid var(--ink);background:var(--ink);color:var(--paper);
 padding:6px 14px;font-size:12px;cursor:pointer;border-radius:2px}
main{max-width:920px;margin:0 auto;padding:24px 20px 120px}
.case{background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--red);
 padding:18px 20px;margin-bottom:14px;border-radius:3px}
.case.done{border-left-color:var(--green);opacity:.62}
.case.cur{outline:2px solid var(--blue);outline-offset:2px}
.meta{display:flex;gap:10px;font-size:11px;color:var(--mut);letter-spacing:.08em;
 margin-bottom:10px;flex-wrap:wrap}
.badge{border:1px solid var(--rule);padding:1px 7px;border-radius:99px}
.ctx{font-family:"Noto Serif CJK SC",serif;font-size:19px;line-height:2.1;margin:6px 0 14px}
.ctx b{color:var(--red);font-weight:700}
.ops{border-top:1px dashed var(--rule);margin-top:10px;padding-top:10px}
.op{font-size:13px;line-height:1.85;margin-bottom:5px;
 font-family:"Noto Serif CJK SC",serif}
.op .who{display:inline-block;min-width:74px;font-weight:700;color:var(--blue)}
.op .g{font-size:11px;color:var(--mut);border:1px solid var(--rule);
 padding:0 5px;border-radius:99px;margin:0 6px}
.choices{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}
.ch{border:1px solid var(--rule);background:#fff;padding:8px 14px;cursor:pointer;
 font-size:14px;border-radius:2px;font-family:"Noto Serif CJK SC",serif}
.ch:hover{border-color:var(--ink)}
.ch.sel{background:var(--blue);color:#fff;border-color:var(--blue)}
.ch kbd{font-size:10px;opacity:.6;margin-right:6px}
textarea{width:100%;margin-top:10px;border:1px solid var(--rule);padding:8px;
 font-size:13px;border-radius:2px;background:#fff;resize:vertical;min-height:44px;
 font-family:"Noto Serif CJK SC",serif}
.risk{font-size:12px;color:var(--red);margin-top:8px}
.hint{font-size:11px;color:var(--mut);margin-top:20px;line-height:1.9}
.empty{text-align:center;color:var(--mut);padding:60px 0;font-size:14px}
</style>
<header>
 <h1>人 工 精 校 台</h1>
 <div class="tabs">
  <button class="tab on" data-f="pending">待覆核 <span id="n1"></span></button>
  <button class="tab" data-f="done">已決 <span id="n2"></span></button>
  <button class="tab" data-f="all">全部 <span id="n3"></span></button>
 </div>
 <div class="spacer"></div>
 <div class="stat" id="prog"></div>
 <button class="act" id="exp">導出 decisions.json</button>
</header>
<main id="list"></main>
<script>
const DATA = __DATA__;
const KEY = 'guji:' + DATA.work + ':' + DATA.run_id;
let dec = JSON.parse(localStorage.getItem(KEY) || '{}');
let filter = 'pending', cur = 0;

const save = () => localStorage.setItem(KEY, JSON.stringify(dec));
const esc = s => (s ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

function visible(){
  return DATA.cases.filter(c => filter === 'all' ? true
    : filter === 'done' ? (dec[c.id] || !c.suspended)
    : (c.suspended && !dec[c.id]));
}

function render(){
  const list = document.getElementById('list');
  const items = visible();
  document.getElementById('n1').textContent = DATA.cases.filter(c=>c.suspended&&!dec[c.id]).length;
  document.getElementById('n2').textContent = DATA.cases.filter(c=>dec[c.id]||!c.suspended).length;
  document.getElementById('n3').textContent = DATA.cases.length;
  const solved = Object.keys(dec).length;
  const need = DATA.cases.filter(c=>c.suspended).length;
  document.getElementById('prog').textContent =
    `懸置 ${need} 條，已處理 ${solved} 條`;

  if(!items.length){ list.innerHTML = '<div class="empty">此頁無條目。懸置條目已全部處理。</div>'; return; }
  list.innerHTML = items.map((c,i) => {
    const d = dec[c.id];
    const chs = c.choices.map((x,k) =>
      `<button class="ch ${d&&d.choice===x?'sel':''}" data-id="${c.id}" data-v="${esc(x)}">
        <kbd>${k+1}</kbd>${esc(x)||'〇（刪）'}</button>`).join('');
    const ops = c.opinions.map(o =>
      `<div class="op"><span class="who">${esc(o.name)}</span>
       <span class="g">${esc(o.method)}·${esc(o.grade)}·${o.confidence.toFixed(2)}</span>
       主「${esc(o.choice)}」：${esc(o.argument)}</div>`).join('');
    return `<div class="case ${d?'done':''} ${i===cur?'cur':''}" data-id="${c.id}">
      <div class="meta"><span class="badge">${esc(c.id)}</span>
        <span class="badge">${esc(c.type)}</span>
        <span class="badge">合議一致度 ${c.confidence.toFixed(2)}</span>
        <span class="badge">OCR ${c.ocr_conf.toFixed(2)}</span>
        ${c.suspended?'<span class="badge" style="color:#9c3b2e">懸置</span>':''}</div>
      <div class="ctx">…${esc(c.left)}<b>〔${esc(c.base)||'〇'}〕</b>${esc(c.right)}…
        <br><span style="font-size:14px;color:#6f6555">今本此處作：〔${esc(c.ref)||'〇'}〕</span></div>
      <div class="ops">${ops}</div>
      <div class="risk">機器裁斷：${esc(c.chosen)}　｜　${esc(c.risk)}</div>
      <div class="choices">${chs}
        <button class="ch ${d&&d.choice==='兩存'?'sel':''}" data-id="${c.id}" data-v="兩存"><kbd>3</kbd>兩存</button>
      </div>
      <textarea placeholder="人工按語（將寫入校勘記）" data-note="${c.id}">${esc(d?d.note:'')}</textarea>
    </div>`;
  }).join('') + `<div class="hint">鍵盤：J／K 移動　1‑9 選讀法　3 兩存　E 導出<br>
   決策存於本機 localStorage，導出 decisions.json 後回灌流水線即可重出定本。</div>`;
}

document.addEventListener('click', e => {
  const t = e.target.closest('.ch');
  if(t){ const id = t.dataset.id;
    dec[id] = {choice: t.dataset.v, note: (dec[id]||{}).note || '', at: new Date().toISOString()};
    save(); render(); return; }
  const tab = e.target.closest('.tab');
  if(tab){ filter = tab.dataset.f; cur = 0;
    document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('on', x===tab));
    render(); }
});
document.addEventListener('input', e => {
  const id = e.target.dataset.note;
  if(id){ dec[id] = Object.assign({choice:'', note:''}, dec[id], {note:e.target.value}); save(); }
});
document.getElementById('exp').onclick = () => {
  const out = {work: DATA.work, run_id: DATA.run_id,
    exported_at: new Date().toISOString(), decisions: dec};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download = 'decisions.json'; a.click();
};
document.addEventListener('keydown', e => {
  if(e.target.tagName === 'TEXTAREA') return;
  const items = visible();
  if(e.key === 'j'){ cur = Math.min(cur+1, items.length-1); render(); }
  if(e.key === 'k'){ cur = Math.max(cur-1, 0); render(); }
  if(e.key === 'e'){ document.getElementById('exp').click(); }
  if(/^[1-9]$/.test(e.key) && items[cur]){
    const c = items[cur]; const opts = c.choices.concat(['兩存']);
    const v = opts[+e.key-1]; if(v===undefined) return;
    dec[c.id] = {choice:v, note:(dec[c.id]||{}).note||'', at:new Date().toISOString()};
    save(); render();
  }
});
render();
</script></html>""".replace("__DATA__", data).replace("__TITLE__", title)


def build_payload(work: str, run_id: str, verdicts, divergences) -> dict:
    from .normalize import normalize
    dmap = {d.id: d for d in divergences}
    cases = []
    for v in verdicts:
        d = dmap.get(v.div_id)
        if d is None:
            continue
        # 今本多为简体，直接作为候选会让人工"选出一个简体字"落进公开本，
        # 故候选一律先投影为正体，只有底本原字保持原样。
        raw_choices = [d.base_text] + [normalize(c).cnf for c in
                                       ([d.ref_text] + d.candidates) if c is not None]
        choices = [c for c in dict.fromkeys(raw_choices)]
        cases.append({
            "id": v.div_id, "type": v.dtype, "left": d.left_ctx, "right": d.right_ctx,
            "base": d.base_text, "ref": d.ref_text, "choices": choices,
            "chosen": v.chosen, "confidence": v.confidence, "ocr_conf": d.ocr_conf,
            "suspended": v.suspended, "risk": v.risk, "note": v.apparatus_note,
            "opinions": [{"name": o["name"], "method": o["method"],
                          "choice": o["choice"], "grade": o["evidence_grade"],
                          "confidence": o["confidence"], "argument": o["argument"]}
                         for o in v.opinions],
        })
    cases.sort(key=lambda c: (not c["suspended"], c["confidence"]))
    return {"work": work, "run_id": run_id, "cases": cases}
