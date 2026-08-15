r"""对齐与异文检测

两级对齐：
  一级——锚点对齐：以长公共子串为锚，把长文切成互相对应的段块，
        避免单次全局比对在万字以上文本中把远处相同字错配。
  二级——字符级对齐：段块内用带古籍代价函数的 Needleman–Wunsch，
        代价函数把"通假/异体/形近"视为低代价替换，把"实质异文"视为高代价。

替换代价定义：

    \[
    d(a,b)=
    \begin{cases}
      0,      & \text{CNF 相同（简繁、异体）}\\
      0.25,   & \text{通假或古今字}\\
      0.35,   & \text{避讳改字}\\
      0.45,   & \text{形近易混（疑 OCR 誤）}\\
      1.0,    & \text{其他}
    \end{cases}
    \]

脱字/衍字代价 \(g=0.8\)，故形近误识优先被判为"替换"而非"脱+衍"。
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field, asdict
from enum import Enum

from .normalize import (
    NormalizedText, normalize, loan_note, taboo_note, is_confusable,
    LACUNA_MARKS, DITTO_MARKS,
)


class DivType(str, Enum):
    ORTHOGRAPHY = "字形差異"       # 简繁/异体，自动消解，不入校勘记正文
    PHONETIC_LOAN = "通假古今字"
    TABOO = "避諱改字"
    OCR_SHAPE = "形近疑誤"
    LACUNA = "脫文缺字"
    DITTOGRAPHY = "衍文重出"
    TRANSPOSITION = "倒文"
    PUNCTUATION = "句讀"
    SUBSTANTIVE = "實質異文"


@dataclass
class Divergence:
    """一处异文。base = 善本（底本），ref = 现代点校本（参校本）。"""
    id: str
    base_start: int
    base_end: int
    ref_start: int
    ref_end: int
    base_text: str
    ref_text: str
    left_ctx: str
    right_ctx: str
    dtype: DivType
    ocr_conf: float = 1.0
    auto_note: str = ""
    candidates: list[str] = field(default_factory=list)
    resolved: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dtype"] = self.dtype.value
        return d

    @property
    def display(self) -> str:
        b = self.base_text or "〇"
        r = self.ref_text or "〇"
        return f"{self.left_ctx}【底本：{b} / 今本：{r}】{self.right_ctx}"


SUB_COST = {"same": 0.0, "loan": 0.25, "taboo": 0.35, "shape": 0.45, "other": 1.0}
GAP_COST = 0.8


def sub_kind(a: str, b: str) -> str:
    if a == b:
        return "same"
    if loan_note(a, b):
        return "loan"
    if taboo_note(a, b):
        return "taboo"
    if is_confusable(a, b):
        return "shape"
    return "other"


def nw_align(a: str, b: str) -> list[tuple[str, int, int, int, int]]:
    """带古籍代价的 Needleman–Wunsch，返回 opcodes：(tag, i1, i2, j1, j2)。"""
    n, m = len(a), len(b)
    if n * m > 400_000:      # 超长块退回 difflib，保证可伸缩
        return list(difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes())
    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = i * GAP_COST
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j * GAP_COST
        bt[0][j] = "I"
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            c = SUB_COST[sub_kind(ai, b[j - 1])]
            best, tag = dp[i - 1][j - 1] + c, ("M" if c == 0 else "S")
            if dp[i - 1][j] + GAP_COST < best:
                best, tag = dp[i - 1][j] + GAP_COST, "D"
            if dp[i][j - 1] + GAP_COST < best:
                best, tag = dp[i][j - 1] + GAP_COST, "I"
            dp[i][j], bt[i][j] = best, tag

    path, i, j = [], n, m
    while i > 0 or j > 0:
        t = bt[i][j]
        if t in ("M", "S"):
            path.append((t, i - 1, j - 1)); i -= 1; j -= 1
        elif t == "D":
            path.append(("D", i - 1, j)); i -= 1
        else:
            path.append(("I", i, j - 1)); j -= 1
    path.reverse()

    ops: list[tuple[str, int, int, int, int]] = []
    for t, i0, j0 in path:
        tag = {"M": "equal", "S": "replace", "D": "delete", "I": "insert"}[t]
        i1, i2 = (i0, i0 + 1) if t in ("M", "S", "D") else (i0, i0)
        j1, j2 = (j0, j0 + 1) if t in ("M", "S", "I") else (j0, j0)
        if ops and ops[-1][0] == tag and ops[-1][2] == i1 and ops[-1][4] == j1:
            p = ops.pop()
            ops.append((tag, p[1], i2, p[3], j2))
        else:
            ops.append((tag, i1, i2, j1, j2))
    return ops


def anchor_blocks(a: str, b: str, min_anchor: int = 12) -> list[tuple[int, int, int, int]]:
    """用长公共块做锚，切分为可独立比对的段块。"""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    anchors = [m for m in sm.get_matching_blocks() if m.size >= min_anchor]
    blocks, pa, pb = [], 0, 0
    for m in anchors:
        if m.a > pa or m.b > pb:
            blocks.append((pa, m.a, pb, m.b))
        blocks.append((m.a, m.a + m.size, m.b, m.b + m.size))
        pa, pb = m.a + m.size, m.b + m.size
    if pa < len(a) or pb < len(b):
        blocks.append((pa, len(a), pb, len(b)))
    return blocks or [(0, len(a), 0, len(b))]


def classify(base_txt: str, ref_txt: str, base_cnf: str, ref_cnf: str,
             left: str, right: str) -> tuple[DivType, str, list[str]]:
    """异文类型判定 + 自动按语 + 候选读法。

    判定一律在 CNF 上进行（消解简繁异体），但按语与候选读法用原字形，
    因为最终落盘的必须是底本的原字。
    """
    cands = [c for c in dict.fromkeys([base_txt, ref_txt])]

    if base_txt and any(ch in LACUNA_MARKS for ch in base_txt):
        return DivType.LACUNA, "底本此處漫漶／蟲蛀，據今本擬補，當覆核書影", cands
    if not base_txt:
        return DivType.LACUNA, "底本無此字，今本有之，疑底本脫", cands
    if not ref_txt:
        if len(base_txt) == 1 and (right.startswith(base_txt) or left.endswith(base_txt)):
            return DivType.DITTOGRAPHY, "底本重出，疑重文號「々」被誤展，或涉上下文而衍", cands
        return DivType.DITTOGRAPHY, "底本有而今本無，疑衍文，或今本脫", cands

    if len(base_cnf) == len(ref_cnf) >= 2 and sorted(base_cnf) == sorted(ref_cnf):
        return DivType.TRANSPOSITION, "二本字同而次序異，疑有倒文，當以文義及他書徵引定之", cands

    if len(base_cnf) == len(ref_cnf) == 1:
        a, b = base_cnf, ref_cnf
        if (n := loan_note(a, b)):
            return DivType.PHONETIC_LOAN, n, cands
        if (t := taboo_note(a, b)):
            return (DivType.TABOO,
                    f"{t['note']}（避{t['dynasty']}{t['taboo_for']}諱）；"
                    f"底本作「{base_txt}」可為版本斷代之證", cands)
        if is_confusable(a, b):
            return DivType.OCR_SHAPE, f"「{base_txt}」「{ref_txt}」形近，須回書影覆核字口", cands
        return DivType.SUBSTANTIVE, "二本用字不同，非形近亦非通假，屬實質異文", cands

    return DivType.SUBSTANTIVE, "二本文字互異，須詳考", cands


def _fix_transpositions(ops, a: str, b: str, max_gap: int = 4):
    """识别「脫—同—衍」形态的隔字倒文。

    NW 在 \\(2g<2d\\)（两个空位代价低于两次替换）时，会把「文學→學文」
    拆成「刪文 + 學 + 補文」。这是代价函数的必然结果，不是错误，
    但校勘上应作一条倒文处理，故在此做形态归并。
    """
    out, k = [], 0
    while k < len(ops):
        if k + 2 < len(ops):
            o1, o2, o3 = ops[k], ops[k + 1], ops[k + 2]
            if (o1[0] in ("delete", "insert") and o2[0] == "equal"
                    and o3[0] in ("delete", "insert") and o1[0] != o3[0]
                    and (o2[2] - o2[1]) <= max_gap):
                bs, be = o1[1], o3[2] if o3[0] == "delete" else o3[1]
                rs, re_ = o1[3], o3[4]
                base_seg, ref_seg = a[bs:be], b[rs:re_]
                if len(base_seg) == len(ref_seg) and sorted(base_seg) == sorted(ref_seg):
                    out.append(("replace", bs, be, rs, re_))
                    k += 3
                    continue
        out.append(ops[k])
        k += 1
    return out


def _merge_ops(ops):
    """把相邻的非 equal 操作合并为一个跨度，避免「倒文」被拆成脫＋衍。"""
    merged, buf = [], None
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            if buf:
                merged.append(buf)
                buf = None
            continue
        if buf and buf[2] == i1 and buf[4] == j1:
            buf = ("replace", buf[1], i2, buf[3], j2)
        else:
            if buf:
                merged.append(buf)
            buf = ("replace", i1, i2, j1, j2)
    if buf:
        merged.append(buf)
    return merged


def collate(base: NormalizedText, ref: NormalizedText,
            conf_lookup=None, ctx_width: int = 8) -> list[Divergence]:
    """主对校函数：底本 CNF × 今本 CNF -> 异文列表。

    切分策略：先把相邻差异合并成跨度，再按下列规则拆条——
      · 等长且互为易位  -> 整跨度作一条「倒文」
      · 等长非易位      -> 逐字拆条（校勘記以字為單位）
      · 不等长          -> 整跨度作一条（脫／衍／實質異文）
    """
    divs: list[Divergence] = []
    counter = 0

    def emit(bs, be, rs, re_):
        nonlocal counter
        base_cnf, ref_cnf = base.cnf[bs:be], ref.cnf[rs:re_]
        if base_cnf == ref_cnf:
            return
        base_txt = base.raw_slice(bs, be)
        ref_txt = ref.raw_slice(rs, re_)
        left, right = base.context(bs, be, ctx_width)
        dtype, note, cands = classify(base_txt, ref_txt, base_cnf, ref_cnf, left, right)
        counter += 1
        conf = 1.0
        if conf_lookup:
            cs = [conf_lookup(k) for k in range(bs, be)]
            conf = min(cs) if cs else 1.0
        divs.append(Divergence(
            id=f"D{counter:04d}", base_start=bs, base_end=be,
            ref_start=rs, ref_end=re_, base_text=base_txt, ref_text=ref_txt,
            left_ctx=left, right_ctx=right, dtype=dtype,
            ocr_conf=round(conf, 4), auto_note=note, candidates=cands))

    for (a0, a1, b0, b1) in anchor_blocks(base.cnf, ref.cnf):
        seg_a, seg_b = base.cnf[a0:a1], ref.cnf[b0:b1]
        if seg_a == seg_b:
            continue
        for tag, i1, i2, j1, j2 in _merge_ops(
                _fix_transpositions(nw_align(seg_a, seg_b), seg_a, seg_b)):
            bs, be, rs, re_ = a0 + i1, a0 + i2, b0 + j1, b0 + j2
            la, lb = be - bs, re_ - rs
            if la == lb >= 2 and sorted(base.cnf[bs:be]) == sorted(ref.cnf[rs:re_]):
                emit(bs, be, rs, re_)
            elif la == lb >= 2:
                for k in range(la):
                    emit(bs + k, bs + k + 1, rs + k, rs + k + 1)
            else:
                emit(bs, be, rs, re_)
    return divs


# --------------------------------------------------------------------------
# 句读投影：把今本标点搬到底本上（仅作为"待审提案"，非最终定稿）
# --------------------------------------------------------------------------

def project_punctuation(base: NormalizedText, ref: NormalizedText) -> dict[int, str]:
    """返回 {底本 CNF 下标: 该字之后应加的标点}。"""
    sm = difflib.SequenceMatcher(None, base.cnf, ref.cnf, autojunk=False)
    mapping: dict[int, int] = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            mapping[blk.b + k] = blk.a + k
    out: dict[int, str] = {}
    for rj, punct in ref.punct_after.items():
        if rj in mapping:
            out[mapping[rj]] = punct
    return out


def punctuation_disputes(base: NormalizedText, proposals: dict[int, str],
                         alt_proposals: dict[int, str] | None = None) -> list[Divergence]:
    """当存在两套句读方案（如今本 A、今本 B）时，逐点生成句读异议。"""
    if not alt_proposals:
        return []
    divs, n = [], 0
    for i in sorted(set(proposals) | set(alt_proposals)):
        p1, p2 = proposals.get(i, ""), alt_proposals.get(i, "")
        if p1 == p2:
            continue
        n += 1
        left, right = base.context(i, i + 1, 10)
        divs.append(Divergence(
            id=f"P{n:04d}", base_start=i, base_end=i + 1, ref_start=i, ref_end=i + 1,
            base_text=base.raw_slice(i, i + 1), ref_text=base.raw_slice(i, i + 1),
            left_ctx=left, right_ctx=right, dtype=DivType.PUNCTUATION,
            auto_note=f"句讀分歧：甲本作「{p1 or '不斷'}」，乙本作「{p2 or '不斷'}」",
            candidates=[p1 or "不斷", p2 or "不斷"]))
    return divs
