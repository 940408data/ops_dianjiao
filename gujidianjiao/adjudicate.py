r"""裁决层

把三家意见聚合为一个决议。聚合不是简单多数——校勘学里证据是**分等级**的，
一条唐写本的实证胜过三条"文义较顺"的推测。故采用双因子加权：

    \[
    S(h)=\sum_{k}w_k(t)\cdot g_k \cdot c_{k,h}
    \]

其中 \(w_k(t)\) 为角色 \(k\) 在异文类型 \(t\) 上的话语权，\(g_k\) 为该次发言的
证据等级系数，\(c_{k,h}\) 为角色对读法 \(h\) 的置信度。归一化后：

    \[
    P(h)=\frac{S(h)}{\sum_{h'}S(h')}
    \]

**底本优先原则**（不改字之偏置）：若 \(h\) 即底本原文，得分再乘以 \(1+\beta\)，
\(\beta=0.15\)。这把清代校勘学"底本可通則不改"的成规写进了目标函数。

悬置条件（满足其一即挂起，交人工）：

    \[
    P_{(1)}<\tau \quad\text{或}\quad P_{(1)}-P_{(2)}<\delta
    \quad\text{或}\quad \max_k g_k \le g_{\text{推測}}\ \text{且需改動正文}
    \]

最后一条最要紧：**没有硬证据就要改底本正文的，一律悬置**。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .align import Divergence, DivType
from .agents import Opinion, Persona, COUNCIL, CHAIR

GRADE_COEF = {"確證": 1.0, "旁證": 0.7, "推測": 0.4, "無證": 0.15}
GRADE_RANK = {"確證": 3, "旁證": 2, "推測": 1, "無證": 0}

BASE_PREFERENCE_BETA = 0.15   # 底本优先偏置
TAU = 0.55                    # 首选读法最低占比
DELTA = 0.18                  # 首选与次选的最小差距


@dataclass
class Verdict:
    div_id: str
    dtype: str
    base_text: str
    ref_text: str
    context: str
    chosen: str
    amend_base_text: bool
    suspended: bool
    confidence: float
    margin: float
    scores: dict[str, float]
    apparatus_note: str
    risk: str
    opinions: list[dict] = field(default_factory=list)
    decided_by: str = "council"      # council | chair-llm | human
    human_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _weight(p: Persona, dtype: DivType) -> float:
    return p.weights.get(dtype.value, p.default_weight)


def tally(d: Divergence, opinions: list[Opinion]) -> dict[str, float]:
    scores: dict[str, float] = {}
    by_key = {p.key: p for p in COUNCIL}
    for o in opinions:
        p = by_key.get(o.persona)
        if p is None:
            continue
        w = _weight(p, d.dtype)
        g = GRADE_COEF.get(o.evidence_grade, 0.4)
        scores[o.choice] = scores.get(o.choice, 0.0) + w * g * o.confidence
    if d.base_text and d.base_text in scores:
        scores[d.base_text] *= (1 + BASE_PREFERENCE_BETA)
    return {k: round(v, 4) for k, v in scores.items()}


def adjudicate(d: Divergence, opinions: list[Opinion],
               chair_json: dict | None = None,
               tau: float = TAU, delta: float = DELTA) -> Verdict:
    scores = tally(d, opinions)
    total = sum(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    p1 = ranked[0][1] / total if ranked else 0.0
    p2 = ranked[1][1] / total if len(ranked) > 1 else 0.0
    top = ranked[0][0] if ranked else "兩存"
    margin = round(p1 - p2, 4)

    best_grade = max((GRADE_RANK.get(o.evidence_grade, 0) for o in opinions), default=0)
    would_amend = (top != d.base_text) and top != "兩存"

    suspend = False
    reasons = []
    if p1 < tau:
        suspend, _ = True, reasons.append(f"首選讀法得票率 {p1:.2f} 低於閾值 {tau}")
    if margin < delta:
        suspend, _ = True, reasons.append(f"首選與次選差距 {margin:.2f} 小於閾值 {delta}")
    if would_amend and best_grade <= GRADE_RANK["推測"]:
        suspend, _ = True, reasons.append("擬改動底本正文，然三家皆無確證或旁證")
    if top == "兩存":
        suspend, _ = True, reasons.append("三家主兩存")
    if d.ocr_conf < 0.75:
        suspend, _ = True, reasons.append(f"底本該處 OCR 置信度僅 {d.ocr_conf:.2f}，須覆核書影")

    chosen = top
    amend = would_amend and not suspend
    decided_by = "council"
    note = _compose_note(d, opinions, chosen, amend, suspend)
    risk = "；".join(reasons) if suspend else ("改動底本正文" if amend else "僅出校記，不改正文")

    if chair_json:
        decided_by = "chair-llm"
        chosen = chair_json.get("verdict", chosen)
        amend = bool(chair_json.get("amend_base_text", amend))
        suspend = bool(chair_json.get("suspend", suspend))
        note = chair_json.get("apparatus_note", note)
        risk = chair_json.get("risk", risk)
        p1 = float(chair_json.get("confidence", p1))

    if suspend:
        amend = False   # 悬置一律不动正文

    return Verdict(
        div_id=d.id, dtype=d.dtype.value, base_text=d.base_text, ref_text=d.ref_text,
        context=d.display, chosen=chosen, amend_base_text=amend, suspended=suspend,
        confidence=round(p1, 4), margin=margin, scores=scores,
        apparatus_note=note, risk=risk,
        opinions=[o.to_dict() for o in opinions], decided_by=decided_by)


def _compose_note(d: Divergence, opinions: list[Opinion],
                  chosen: str, amend: bool, suspend: bool) -> str:
    """按四库体例生成校勘记文字。"""
    b = d.base_text or "無此字"
    r = d.ref_text or "無此字"
    head = f"「{b}」，今本作「{r}」。"
    if d.dtype is DivType.PHONETIC_LOAN:
        head = f"「{b}」，今本作「{r}」，二字通。"
    elif d.dtype is DivType.TABOO:
        head = f"「{b}」，今本作「{r}」，避諱改字。"
    elif d.dtype is DivType.LACUNA:
        head = f"底本無「{r}」字，今本有之。"
    elif d.dtype is DivType.DITTOGRAPHY:
        head = f"底本「{b}」重出，今本不重。"
    elif d.dtype is DivType.TRANSPOSITION:
        head = f"底本作「{b}」，今本作「{r}」，次第互異。"

    strongest = max(opinions, key=lambda o: (GRADE_RANK.get(o.evidence_grade, 0), o.confidence))
    body = f"{strongest.name}案：{strongest.argument}"
    if suspend:
        tail = "諸說未一，姑兩存之，俟考。"
    elif amend:
        tail = "今據刪。" if chosen == "" else f"今據改作「{chosen}」。"
    else:
        tail = "今仍底本之舊，不改。"
    return head + body + tail


def summarize(verdicts: list[Verdict]) -> dict:
    by_type: dict[str, int] = {}
    for v in verdicts:
        by_type[v.dtype] = by_type.get(v.dtype, 0) + 1
    return {
        "total": len(verdicts),
        "amended": sum(1 for v in verdicts if v.amend_base_text),
        "kept": sum(1 for v in verdicts if not v.amend_base_text and not v.suspended),
        "suspended": sum(1 for v in verdicts if v.suspended),
        "by_type": by_type,
        "auto_resolution_rate": round(
            sum(1 for v in verdicts if not v.suspended) / max(1, len(verdicts)), 4),
    }
