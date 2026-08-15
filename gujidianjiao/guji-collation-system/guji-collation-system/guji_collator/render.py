"""断句与成品渲染

版权隔离是本模块的第一约束：

  · 底本（公有领域善本）的**文字**不受著作权保护 —— 可公开。
  · 现代点校本的**标点、分段、校勘记文字**是整理者的独创性成果，受保护 —— 不可照搬。
  · 单字异文的**事实**（某本作某字）本身是事实信息，可记录；但成套照抄他人校记不可。

因此系统的做法是：
  1. 公开本的句读**必须独立生成**（规则断句器 + 戴震智能体），
     现代本的标点只作为"提示"参与比对，绝不直接落盘为公开本标点。
  2. 生成后计算与现代本的句读重合率，过高则告警并要求人工复核（防止"洗稿式"抄标点）。
  3. 仅以现代本为唯一来源的校记条目，在公开本中标记为 restricted，
     只保留异文事实，不保留他人按语措辞。
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field

from .normalize import NormalizedText, split_sentences
from .adjudicate import Verdict
from .align import DivType

# --------------------------------------------------------------------------
# 规则断句器（独立于现代点校本）
# --------------------------------------------------------------------------

Q_END = set("乎邪耶")                  # 疑问语气词
STRONG_END = set("矣焉耳爾哉")            # 陈述／感叹收句
SOFT_END = set("也")                      # 语气词，句中句末两可
FINAL_PARTICLE = set("與歟夫")            # 仅在句末位置成立
CLAUSE_HEAD = set("故然雖蓋若凡且至由是今")   # 句首连词／发语词（不含「而」「則」）
QUOTE_VERB = set("曰云謂對")              # 引语动词
HEADS_STRONG = set("子曾有夫蓋故然雖凡若君")  # 足以开启新句的字


@dataclass
class Mark:
    pos: int              # 底本 CNF 下标：标点加在该字之后
    punct: str
    source: str           # rule | agent | modern-hint | human
    rationale: str = ""


class RuleSegmenter:
    """浅层文言断句器：语气词收句、引语动词起引号、发语词前顿开。

    定位是**独立初稿**，准确率不追求超越现代点校本，只求不依赖它——
    这是版权隔离的技术前提。定稿由戴震智能体与人工完成。

    刻意保守的两处：
      · 「而」「則」不触发顿开。现代点校本在此处的取舍差异极大，
        跟着它走等于抄它的判断。
      · 「也」「與」按位置定性，不按词性猜测，宁可漏断不可错断。
    """

    def segment(self, cnf: str, breaks: set[int] | None = None) -> list[Mark]:
        breaks = set(breaks or ())
        n = len(cnf)
        marks: list[Mark] = []
        in_quote = False
        last = -1
        for i, ch in enumerate(cnf):
            nxt = cnf[i + 1] if i + 1 < n else ""
            nx2 = cnf[i + 1:i + 3]
            at_break = (i in breaks) or (i == n - 1)
            tail_room = min((b for b in breaks if b >= i), default=n - 1) - i

            def put(punct: str, why: str):
                nonlocal last, in_quote
                if at_break and in_quote:
                    punct += "」"
                    in_quote = False
                marks.append(Mark(i, punct, "rule", why))
                last = i

            if ch in QUOTE_VERB and nxt and not in_quote and not at_break:
                marks.append(Mark(i, "：「", "rule", "引語動詞後起引號"))
                in_quote, last = True, i
                continue
            if nx2 == "不亦":
                put("，", "「不亦……乎」句式前頓開")
                continue
            if ch in Q_END:
                put("？", "疑問語氣詞收句")
                continue
            if ch in STRONG_END:
                put("，" if tail_room <= 2 and not at_break else "。", "語氣詞收句")
                continue
            if ch in SOFT_END:
                if nx2.startswith("者"):
                    continue
                put("。" if (at_break or nxt in HEADS_STRONG) else "，", "「也」按位置定性")
                continue
            if ch in FINAL_PARTICLE:
                if at_break or nxt in HEADS_STRONG:
                    put("？" if ch in "與歟" else "。", "句末語氣詞")
                continue
            if nxt in CLAUSE_HEAD and (i - last) >= 4:
                put("，", "發語詞前頓開")
                continue
            if at_break:
                put("。", "章末補句號")
        return _dedupe(marks)


def _dedupe(marks: list[Mark]) -> list[Mark]:
    seen: dict[int, Mark] = {}
    for m in marks:
        if m.pos not in seen or len(m.punct) > len(seen[m.pos].punct):
            seen[m.pos] = m
    return [seen[k] for k in sorted(seen)]


def punctuation_overlap(ours: list[Mark], modern: dict[int, str]) -> dict:
    """句读重合率：过高说明公开本的标点实质上复制了现代本，须人工干预。"""
    ours_pos = {m.pos for m in ours}
    mod_pos = set(modern)
    inter = ours_pos & mod_pos
    union = ours_pos | mod_pos
    jaccard = len(inter) / len(union) if union else 0.0
    same_punct = sum(1 for p in inter if modern[p][:1] == next(m.punct for m in ours if m.pos == p)[:1])
    return {
        "ours": len(ours_pos), "modern": len(mod_pos), "shared_positions": len(inter),
        "jaccard": round(jaccard, 4),
        "identical_marks": same_punct,
        "warn": jaccard > 0.92,
    }


# --------------------------------------------------------------------------
# 正文生成
# --------------------------------------------------------------------------

@dataclass
class Edition:
    title: str
    body: str
    apparatus: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    notice: str = ""


def build_base_edition(base: NormalizedText, verdicts: list[Verdict],
                       marks: list[Mark], title: str,
                       witness_label: str = "今本") -> Edition:
    """生成善本点校本：底本原字形 + 独立句读 + 校勘记序号。"""
    repl: dict[int, tuple[int, str]] = {}
    insert: dict[int, str] = {}
    marker_at: dict[int, list[int]] = {}

    numbered = [v for v in verdicts if _in_apparatus(v)]
    for idx, v in enumerate(numbered, start=1):
        d = _span(v)
        if d is None:
            continue
        s, e = d
        marker_at.setdefault(max(s, e - 1) if e > s else s, []).append(idx)
        if v.amend_base_text:
            # 补脱之字以〔〕标出，使读者一望而知何字为底本所无
            new = f"〔{v.chosen}〕" if (v.dtype == DivType.LACUNA.value and v.chosen) else v.chosen
            if e > s:
                repl[s] = (e, new)
            else:
                insert[s] = v.chosen

    mark_map = {m.pos: m.punct for m in marks}
    out: list[str] = []
    i, n = 0, len(base.cnf)
    while i < n:
        if i in insert:
            out.append(f"〔{insert[i]}〕")
        if i in repl:
            e, new = repl[i]
            out.append(new)
            for k in range(i, e):
                if k in marker_at:
                    out.append(_sup(marker_at[k]))
                if k in mark_map:
                    out.append(mark_map[k])
            i = e
            continue
        out.append(base.raw[base.index_map[i]])
        if i in marker_at:
            out.append(_sup(marker_at[i]))
        if i in mark_map:
            out.append(mark_map[i])
        if i in base.line_break_after:
            out.append("\n")
        i += 1
    if n in insert:
        out.append(f"〔{insert[n]}〕")

    body = "".join(out)
    apparatus = [_apparatus_entry(idx, v) for idx, v in enumerate(numbered, start=1)]
    return Edition(title=title, body=body, apparatus=apparatus,
                   notice="底本為公有領域善本，本點校本之句讀與校記由本系統獨立生成，可自由傳播。")


def _span(v: Verdict):
    return getattr(v, "_span", None)


def _in_apparatus(v: Verdict) -> bool:
    return v.dtype != DivType.ORTHOGRAPHY.value


def _sup(nums: list[int]) -> str:
    trans = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "".join(f"[{n}]" for n in nums) if max(nums) > 99 else \
        "".join(str(n).translate(trans) for n in nums)


def _apparatus_entry(idx: int, v: Verdict) -> dict:
    return {
        "no": idx, "id": v.div_id, "type": v.dtype,
        "lemma": v.base_text or "〇", "witness": v.ref_text or "〇",
        "note": v.apparatus_note, "suspended": v.suspended,
        "amended": v.amend_base_text, "confidence": v.confidence,
        "risk": v.risk,
        "opinions": [{"name": o["name"], "method": o["method"],
                      "choice": o["choice"], "grade": o["evidence_grade"],
                      "confidence": o["confidence"], "argument": o["argument"]}
                     for o in v.opinions],
    }


def build_modern_internal(ref_raw: str, verdicts: list[Verdict], title: str) -> Edition:
    """内部参校本：仅供个人研读，带强制声明，禁止发布。"""
    notice = (
        "⚠ 內部研讀本｜禁止公開傳播\n"
        "本文件基於現代點校本掃描件之 OCR 結果生成。現代點校本之標點、分段、"
        "校勘記屬整理者獨創性成果，受著作權法保護。本文件僅供本人比對研讀，"
        "不得複製、發布、上傳或用於任何公開用途。"
    )
    return Edition(title=title + "（內部研讀本）", body=ref_raw,
                   apparatus=[_apparatus_entry(i, v) for i, v in enumerate(verdicts, 1)],
                   notice=notice)


# --------------------------------------------------------------------------
# Markdown / HTML 输出
# --------------------------------------------------------------------------

def edition_to_markdown(ed: Edition, public: bool = True) -> str:
    lines = [f"# {ed.title}", "", f"> {ed.notice}", "", "## 正文", ""]
    for para in ed.body.split("\n"):
        if para.strip():
            lines.append(para.strip())
            lines.append("")
    lines += ["## 校勘記", ""]
    for e in ed.apparatus:
        if public and e.get("restricted"):
            continue
        flag = "〔待覆核〕" if e["suspended"] else ("〔已改〕" if e["amended"] else "")
        lines.append(f"[{e['no']}] {flag}{e['note']}")
    if ed.stats:
        lines += ["", "## 統計", "", "```json",
                  json.dumps(ed.stats, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines)


def edition_to_html(ed: Edition, public: bool = True, subtitle: str = "") -> str:
    esc = html.escape
    rows = []
    for e in ed.apparatus:
        if public and e.get("restricted"):
            continue
        cls = "susp" if e["suspended"] else ("amend" if e["amended"] else "keep")
        ops = "".join(
            f"<li><b>{esc(o['name'])}</b>（{esc(o['method'])}·{esc(o['grade'])}·"
            f"{o['confidence']:.2f}）主「{esc(o['choice'])}」：{esc(o['argument'])}</li>"
            for o in e["opinions"])
        rows.append(
            f"<li class='{cls}' id='ap{e['no']}'><span class='no'>{e['no']}</span>"
            f"<div><p class='note'>{esc(e['note'])}</p>"
            f"<details><summary>合議記錄（{esc(e['type'])}，信度 {e['confidence']:.2f}）</summary>"
            f"<ul class='ops'>{ops}</ul><p class='risk'>風險：{esc(e['risk'])}</p>"
            f"</details></div></li>")
    body_html = "".join(f"<p>{esc(p)}</p>" for p in ed.body.split("\n") if p.strip())
    return f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>{esc(ed.title)}</title>
<style>
:root{{--ink:#1c1a17;--paper:#f7f4ec;--rule:#c9bfa8;--red:#9c3b2e;--blue:#2d4a63}}
body{{margin:0;background:var(--paper);color:var(--ink);
 font-family:"Noto Serif CJK SC","Songti SC",serif;line-height:2}}
.wrap{{max-width:880px;margin:0 auto;padding:48px 24px 96px}}
h1{{font-size:28px;letter-spacing:.08em;margin:0 0 4px}}
.sub{{color:#6b6154;font-size:13px;letter-spacing:.1em;margin-bottom:28px}}
.notice{{border-left:3px solid var(--red);background:#fff8f2;padding:12px 16px;
 font-size:13px;white-space:pre-wrap;margin-bottom:32px}}
.text p{{text-indent:2em;font-size:18px;margin:0 0 .6em}}
h2{{font-size:15px;letter-spacing:.3em;border-bottom:1px solid var(--rule);
 padding-bottom:8px;margin:48px 0 20px;font-weight:600}}
ol,ul{{list-style:none;padding:0;margin:0}}
.ap li{{display:flex;gap:12px;padding:10px 0;border-bottom:1px dotted var(--rule);font-size:14px;line-height:1.9}}
.no{{flex:0 0 28px;height:28px;border-radius:50%;display:grid;place-items:center;
 font-size:12px;background:#e8e0cd;color:#4a4238}}
.susp .no{{background:var(--red);color:#fff}}
.amend .no{{background:var(--blue);color:#fff}}
.note{{margin:0}}
details{{margin-top:6px;font-size:13px;color:#514a3f}}
summary{{cursor:pointer;color:var(--blue)}}
.ops li{{display:block;border:none;padding:3px 0;font-size:13px}}
.risk{{color:var(--red);font-size:12px;margin:6px 0 0}}
</style><div class="wrap">
<h1>{esc(ed.title)}</h1><div class="sub">{esc(subtitle)}</div>
<div class="notice">{esc(ed.notice)}</div>
<div class="text">{body_html}</div>
<h2>校 勘 記</h2><ul class="ap">{''.join(rows)}</ul>
</div></html>"""
