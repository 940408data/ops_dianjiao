"""校书官智能体（Agent 合议庭）

设计的核心不是"让模型扮演古人说话"，而是把**陈垣《校勘学释例》的校法四则**
拆成四个互相制衡的角色，每个角色只被允许使用自己那一路证据：

    对校法 —— 劉向：只比版本异同，不据文义改字（"不主一本，不改一字"）
    他校法 —— 解縉：只查他书徵引、类书、注疏所引本文
    本校法 —— 戴震（兼理校）：只用本书前后互证 + 音韵训诂
    理校法 —— 紀昀：不单独立论，只做终审裁断与风险控制

之所以要"限权"，是因为单一模型最大的风险是**理校泛滥**——凭文义顺畅去改字，
这是清代校勘学公认的大忌。角色隔离 + 权重分型，是对这一风险的工程化约束。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .align import Divergence, DivType


# --------------------------------------------------------------------------
# 角色定义
# --------------------------------------------------------------------------

@dataclass
class Persona:
    key: str
    name: str
    dynasty: str
    method: str
    charter: str          # 系统提示词
    allowed_evidence: list[str]
    # 分型权重：该角色在不同异文类型上的话语权
    weights: dict[str, float] = field(default_factory=dict)
    default_weight: float = 0.6


LIU_XIANG = Persona(
    key="liuxiang", name="劉向", dynasty="西漢", method="對校法",
    allowed_evidence=["底本書影", "他本異文", "篇卷體例", "字數行款"],
    charter="""你是劉向，漢成帝時領校中秘書，《別錄》之作者。校書之法，你首重「對校」。

你的鐵律：
1. **不改一字**。你只陳述「某本作某，某本作某」，並判斷哪一本更可能存古。
2. 你只根據**版本形態**立論：行款、字數、避諱、刻工、篇第、章次、重文號。
3. 你**不得**以「文義較順」為據主張改字。文義是戴震與紀昀的職分，不是你的。
4. 遇底本不誤而今本臆改者，你必須明確指出今本之失。
5. 你熟知：凡兩本互異，晚出之本往往經人整理，反而失真；「不誤而改」多於「誤而未改」。

輸出時給出：你認為當從何本、理由（限版本層面）、把握度。""",
    weights={DivType.LACUNA.value: 1.0, DivType.DITTOGRAPHY.value: 1.0,
             DivType.TABOO.value: 1.0, DivType.OCR_SHAPE.value: 0.9,
             DivType.SUBSTANTIVE.value: 0.85, DivType.PHONETIC_LOAN.value: 0.6,
             DivType.TRANSPOSITION.value: 0.8, DivType.PUNCTUATION.value: 0.3},
)

XIE_JIN = Persona(
    key="xiejin", name="解縉", dynasty="明", method="他校法",
    allowed_evidence=["類書徵引", "他書引文", "注疏所引", "石刻碑版", "敦煌寫本"],
    charter="""你是解縉，《永樂大典》總裁官，平生所閱典籍浩如煙海。校書之法，你長於「他校」。

你的鐵律：
1. 你只以**本書以外的證據**立論：類書所引（《北堂書鈔》《藝文類聚》《太平御覽》）、
   古注所引、他書稱引、碑版石經、寫本殘卷。
2. 引證必須**指名出處**。若你不能確指某書某卷，就必須說「未得確證」，
   **絕不可杜撰書名、卷次或引文**——偽造書證是校勘之罪，甚於不校。
3. 你須留意：類書轉引多有節略改字，故他書所引與本書不同時，未必本書為誤。
4. 唐以前古注所引，其價值高於宋以後類書。

輸出時給出：所見他書異同、出處可靠性評級（確證／旁證／未得確證）、把握度。""",
    weights={DivType.SUBSTANTIVE.value: 1.0, DivType.TRANSPOSITION.value: 0.95,
             DivType.LACUNA.value: 0.9, DivType.PHONETIC_LOAN.value: 0.75,
             DivType.TABOO.value: 0.7, DivType.DITTOGRAPHY.value: 0.8,
             DivType.OCR_SHAPE.value: 0.5, DivType.PUNCTUATION.value: 0.6},
)

DAI_ZHEN = Persona(
    key="daizhen", name="戴震", dynasty="清", method="本校法・理校法（音韻訓詁）",
    allowed_evidence=["本書前後互證", "古音通轉", "小學訓詁", "文例句法"],
    charter="""你是戴震，皖派樸學宗師，由聲音以通訓詁，由訓詁以明義理。

你的鐵律：
1. 你以**本書內部證據**與**小學**立論：同書他處文例、古音通轉、字義本訓。
2. 通假必須合乎**古音**（聲紐、韻部）。你不得僅因二字今音相近便言通假。
3. 你深知理校最險：「凡無左證而以意改者，雖通亦不可從。」
   故你主張改字時，必須同時申明「若不改，於文義有何窒礙」。
4. 若二字皆可通，你當主張**存異不改**，出校記而已。
5. 遇通假字，你的定見是：**古書用本字者少，用借字者多；當存借字，不當回改本字。**

輸出時給出：訓詁依據、古音關係（若涉通假）、當改／當存、把握度。""",
    weights={DivType.PHONETIC_LOAN.value: 1.0, DivType.PUNCTUATION.value: 0.95,
             DivType.SUBSTANTIVE.value: 0.9, DivType.OCR_SHAPE.value: 0.7,
             DivType.TRANSPOSITION.value: 0.85, DivType.LACUNA.value: 0.7,
             DivType.TABOO.value: 0.6, DivType.DITTOGRAPHY.value: 0.7},
)

JI_YUN = Persona(
    key="jiyun", name="紀昀", dynasty="清", method="總纂・裁斷",
    allowed_evidence=["三家之議", "全書體例", "風險權衡"],
    charter="""你是紀昀，《四庫全書》總纂官，《總目提要》多出你手。你不親校一字，只做終審。

你的職分：
1. 你**不得引入三家未提出的新證據**。你只在劉向（對校）、解縉（他校）、
   戴震（本校理校）三家所陳之上作裁斷。
2. 你的裁斷準則，按證據等級由高至低：
   底本可通 ＞ 版本實證 ＞ 他書確證 ＞ 本書文例 ＞ 音韻通轉 ＞ 文義順適。
3. **底本可通則不改**。凡底本文義可通者，縱他本較優，亦只出校記，不改正文。
4. 三家意見分歧而無一方有硬證據者，你必須**判為「兩存」**，不可強斷。
   懸置不是失職，妄斷才是。
5. 你須指出此條若判錯，將造成何種後果（改動正文／僅出校記／影響句讀）。

輸出時給出：裁斷、所據等級、是否改動底本正文、是否懸置待人工覆核。""",
    weights={},
    default_weight=0.0,     # 主席不参与加权投票，只做终审
)

COUNCIL: list[Persona] = [LIU_XIANG, XIE_JIN, DAI_ZHEN]
CHAIR: Persona = JI_YUN
ALL_PERSONAS = {p.key: p for p in COUNCIL + [CHAIR]}


# --------------------------------------------------------------------------
# 议题提示词
# --------------------------------------------------------------------------

CASE_PROMPT = """【校勘議題 {did}】類型：{dtype}

底本（{base_name}）：……{left}〔{base_text}〕{right}……
今本（{ref_name}）此處作：〔{ref_text}〕
機器初判：{auto_note}
底本該處 OCR 置信度：{conf:.2f}

候選讀法：{cands}

請依你的職分與鐵律作出判斷。僅輸出 JSON，不要任何前言後語：
{{
  "choice": "<候選讀法之一，或 \\"兩存\\">",
  "confidence": 0.0-1.0,
  "evidence_grade": "確證|旁證|推測|無證",
  "argument": "不超過80字，須為文言或淺近文言",
  "amend_base_text": true|false
}}"""

CHAIR_PROMPT = """【終審 {did}】類型：{dtype}
底本作〔{base_text}〕，今本作〔{ref_text}〕。上下文：……{left}〔{base_text}〕{right}……

三家之議：
{opinions}

請依《四庫》體例裁斷。僅輸出 JSON：
{{
  "verdict": "<採用之讀法，或 \\"兩存\\">",
  "amend_base_text": true|false,
  "suspend": true|false,
  "confidence": 0.0-1.0,
  "apparatus_note": "校勘記文字，文言，不超過60字",
  "risk": "不超過30字"
}}"""


@dataclass
class Opinion:
    persona: str
    name: str
    method: str
    choice: str
    confidence: float
    evidence_grade: str
    argument: str
    amend_base_text: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# --------------------------------------------------------------------------
# LLM 客户端
# --------------------------------------------------------------------------

class AnthropicClient:
    """接入 Claude。需环境变量 ANTHROPIC_API_KEY。"""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 1024):
        self.model, self.max_tokens = model, max_tokens
        self._key = os.environ.get("ANTHROPIC_API_KEY")

    def complete(self, system: str, user: str) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens,
            "system": system, "messages": [{"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"content-type": "application/json",
                     "x-api-key": self._key or "",
                     "anthropic-version": "2023-06-01"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return "".join(b.get("text", "") for b in data.get("content", []))

    def complete_vision(self, prompt: str, image_ref=None) -> str:
        raise NotImplementedError("接入时改为 base64 image block + text block")


class OpenAICompatClient:
    """兼容 OpenAI /v1/chat/completions 的自建或第三方端点。"""

    def __init__(self, base_url: str, model: str, api_key_env: str = "OPENAI_API_KEY"):
        self.base_url, self.model = base_url.rstrip("/"), model
        self._key = os.environ.get(api_key_env, "")

    def complete(self, system: str, user: str) -> str:
        import urllib.request
        body = json.dumps({"model": self.model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {self._key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> dict | None:
    raw = re.sub(r"```(json)?", "", raw).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------
# 离线启发式推理（无网络时的确定性回退，也用作回归测试基线）
# --------------------------------------------------------------------------

class HeuristicReasoner:
    """把三家鐵律编码为可执行规则。规则来源见各 Persona.charter。"""

    def opine(self, p: Persona, d: Divergence) -> Opinion:
        fn = getattr(self, f"_{p.key}")
        choice, conf, grade, arg, amend = fn(d)
        return Opinion(persona=p.key, name=p.name, method=p.method, choice=choice,
                       confidence=conf, evidence_grade=grade, argument=arg,
                       amend_base_text=amend)

    # ---- 劉向：对校，唯版本是问 ----
    def _liuxiang(self, d: Divergence):
        t = d.dtype
        if t is DivType.TABOO:
            return (d.base_text, 0.92, "確證",
                    f"底本作「{d.base_text}」，今本作「{d.ref_text}」，此避諱改字之跡，"
                    f"正可據以定版本時代，當存底本之舊，不可回改。", False)
        if t is DivType.LACUNA:
            if "□" in d.base_text:
                return (d.ref_text, 0.6, "旁證",
                        "底本此處版面漫漶，非本無其字。當據書影覆核，覈實乃可補。", True)
            return (d.ref_text, 0.7, "旁證",
                    f"底本無「{d.ref_text}」字，今本有之。以行款字數推之，疑底本脫。", True)
        if t is DivType.DITTOGRAPHY:
            return (d.ref_text, 0.82, "確證",
                    f"底本「{d.base_text}」重出。刻本每以重文號「々」示疊字，"
                    f"傳錄者誤展，遂成衍文，當刪。", True)
        if t is DivType.OCR_SHAPE:
            return ("兩存", 0.5, "無證",
                    f"「{d.base_text}」「{d.ref_text}」形近，此當覆核書影字口，"
                    f"非版本之爭，非吾所能斷。", False)
        if t is DivType.PHONETIC_LOAN:
            return (d.base_text, 0.8, "確證",
                    f"底本作「{d.base_text}」，古本多如此；今本作「{d.ref_text}」，"
                    f"乃後人以今字易古字，非舊觀也。", False)
        if t is DivType.TRANSPOSITION:
            return (d.base_text, 0.55, "旁證",
                    "二本次第互異。無他本可證之前，姑仍底本之舊，出校記可也。", False)
        if t is DivType.PUNCTUATION:
            return ("兩存", 0.3, "無證", "句讀非版本之事，吾不與焉。", False)
        return (d.base_text, 0.65, "旁證",
                f"底本作「{d.base_text}」。不改一字，存異而已。", False)

    # ---- 解縉：他校，唯书证是问 ----
    def _xiejin(self, d: Divergence):
        cites = _lookup_citations(d)
        if cites:
            best = cites[0]
            return (best["reading"], best["weight"], best["grade"],
                    f"{best['source']}引此文作「{best['reading']}」，"
                    f"{best.get('remark', '可為佐證')}。", best["reading"] != d.base_text)
        t = d.dtype
        if t is DivType.OCR_SHAPE:
            return ("兩存", 0.35, "無證", "他書無引此句者，未得確證，不敢懸揣。", False)
        if t is DivType.TABOO:
            return (d.base_text, 0.7, "旁證",
                    "唐宋類書所引，避諱字往往隨時改易，故當以底本原字為斷。", False)
        return ("兩存", 0.3, "無證",
                "遍檢類書注疏，未見徵引此句者。未得確證，闕疑可也。", False)

    # ---- 戴震：本校 + 小学 ----
    def _daizhen(self, d: Divergence):
        t = d.dtype
        if t is DivType.PHONETIC_LOAN:
            return (d.base_text, 0.95, "確證",
                    f"{d.auto_note}。古書用借字者多，用本字者少。"
                    f"當存「{d.base_text}」而於校記著其讀，不當回改。", False)
        if t is DivType.OCR_SHAPE:
            good = _semantic_fit(d)
            if good is None:
                return ("兩存", 0.5, "推測",
                        "二字形近而皆可成文，非訓詁所能決。當覆核書影字口，"
                        "不可以意定之。", False)
            return (good, 0.78, "旁證",
                    f"以文義訓詁繩之，作「{good}」乃通；作「"
                    f"{d.ref_text if good == d.base_text else d.base_text}」則於義有窒。", good != d.base_text)
        if t is DivType.TABOO:
            return (d.base_text, 0.75, "旁證",
                    "避諱改字，於訓詁無涉，然改字則害文義，仍當復其本字之舊。", False)
        if t is DivType.LACUNA:
            return (d.ref_text, 0.72, "旁證",
                    f"此句文例當有「{d.ref_text}」字，無之則句法不完。以本書他處例之，可補。", True)
        if t is DivType.DITTOGRAPHY:
            return (d.ref_text, 0.7, "旁證",
                    "重出之字於文義無所繫屬，衍無疑也。", True)
        if t is DivType.TRANSPOSITION:
            return (d.ref_text, 0.6, "推測",
                    "以文義先後推之，今本次第較順。然無左證而以意乙正，不可從，姑志之。", False)
        if t is DivType.PUNCTUATION:
            return (d.candidates[0] if d.candidates else "兩存", 0.7, "旁證",
                    "以語氣詞與文例斷句，當如此讀。", False)
        return ("兩存", 0.45, "推測",
                "無左證而以意改者，雖通不可從。當存異待考。", False)


def _semantic_fit(d: Divergence) -> str | None:
    """形近字取舍：只有当一方明显是罕用字（几乎不可能出现在此类文献）时才表态；
    二字皆可成文时返回 None，由戴震主「兩存」——这是对理校泛滥的刻意抑制。"""
    RARE = set("眚巳戌戍剌冑祟簿詎")
    b, r = d.base_text, d.ref_text
    if b and r and all(c in RARE for c in b) and not any(c in RARE for c in r):
        return r
    if b and r and all(c in RARE for c in r) and not any(c in RARE for c in b):
        return b
    return None


_CITATIONS: dict[str, list[dict]] = {}


def load_citations(path: str | None) -> None:
    """加载他校证据库（类书/古注引文索引）。"""
    global _CITATIONS
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _CITATIONS = json.load(f)


def _lookup_citations(d: Divergence) -> list[dict]:
    """检索他校证据。

    两个必要条件缺一不可：
      1. 证据键出现在本条异文的上下文窗口中（定位对）
      2. 证据键包含本条异文的底本字（指向对）
    只满足条件 1 会造成"张冠李戴"——同一段落里的另一处异文被误配证据，
    这是他校最常见的事故，故必须两条同时成立。
    """
    window = d.left_ctx + d.base_text + d.right_ctx
    hits = [v for k, v in _CITATIONS.items()
            if k and k in window and d.base_text and d.base_text in k]
    return sorted([c for v in hits for c in v], key=lambda x: -x["weight"])


class Council:
    """合议庭执行器：三家发议 -> 主席终审。"""

    def __init__(self, client=None, reasoner: HeuristicReasoner | None = None,
                 base_name: str = "底本", ref_name: str = "今本"):
        self.client = client
        self.reasoner = reasoner or HeuristicReasoner()
        self.base_name, self.ref_name = base_name, ref_name

    def deliberate(self, d: Divergence) -> list[Opinion]:
        return [self._one(p, d) for p in COUNCIL]

    def _one(self, p: Persona, d: Divergence) -> Opinion:
        if self.client is None:
            return self.reasoner.opine(p, d)
        user = CASE_PROMPT.format(
            did=d.id, dtype=d.dtype.value, base_name=self.base_name,
            ref_name=self.ref_name, left=d.left_ctx, right=d.right_ctx,
            base_text=d.base_text or "〇", ref_text=d.ref_text or "〇",
            auto_note=d.auto_note, conf=d.ocr_conf,
            cands="、".join(d.candidates) or "（無）")
        parsed = _parse_json(self.client.complete(p.charter, user))
        if not parsed:
            return self.reasoner.opine(p, d)
        return Opinion(persona=p.key, name=p.name, method=p.method,
                       choice=parsed.get("choice", "兩存"),
                       confidence=float(parsed.get("confidence", 0.5)),
                       evidence_grade=parsed.get("evidence_grade", "推測"),
                       argument=parsed.get("argument", ""),
                       amend_base_text=bool(parsed.get("amend_base_text", False)))

    def chair_review(self, d: Divergence, opinions: list[Opinion]) -> dict | None:
        if self.client is None:
            return None
        text = "\n".join(f"- {o.name}（{o.method}，{o.evidence_grade}，"
                         f"信度{o.confidence:.2f}）主「{o.choice}」：{o.argument}"
                         for o in opinions)
        user = CHAIR_PROMPT.format(did=d.id, dtype=d.dtype.value,
                                   base_text=d.base_text or "〇",
                                   ref_text=d.ref_text or "〇",
                                   left=d.left_ctx, right=d.right_ctx, opinions=text)
        return _parse_json(self.client.complete(CHAIR.charter, user))
