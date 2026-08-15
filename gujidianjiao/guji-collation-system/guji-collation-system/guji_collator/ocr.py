"""OCR 采集层：双通道识别 + 视觉复核

古籍 OCR 与普通 OCR 的差别在于：错误不是随机的，而是**结构化**的。
主要错误源：
  1. 形近字混识（省/眚、己/已/巳）——需要更大字号的局部重裁切复核
  2. 重文号「々」被展开或吞掉——造成衍文/脱文假象
  3. 虫蛀、漫漶、界栏压字——应输出缺字符 □ 而不是猜字
  4. 双行小字夹注被串入正文——需要版面分析先分层
  5. 避讳缺笔字被识别为另一个字

因此本层的设计原则是：**宁可标记不确定，不可自作主张**。
每个字符都带 (char, conf, bbox)，低置信区间会被推入视觉复核队列。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Protocol

from .normalize import LACUNA_MARKS, DITTO_MARKS


@dataclass
class Glyph:
    """单字识别结果。"""
    char: str
    conf: float
    page: int = 0
    line: int = 0
    col: int = 0
    bbox: tuple[int, int, int, int] | None = None
    layer: str = "main"          # main | interlinear | header | marginal
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    rechecked: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox) if self.bbox else None
        return d


@dataclass
class OCRPage:
    page: int
    glyphs: list[Glyph]
    image_path: str | None = None

    @property
    def text(self) -> str:
        lines: dict[int, list[str]] = {}
        for g in self.glyphs:
            if g.layer == "main":
                lines.setdefault(g.line, []).append(g.char)
        return "\n".join("".join(lines[k]) for k in sorted(lines))


@dataclass
class OCRResult:
    source_id: str
    pages: list[OCRPage]
    engine: str = "unknown"

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    @property
    def glyphs(self) -> list[Glyph]:
        return [g for p in self.pages for g in p.glyphs if g.layer == "main"]

    def low_confidence(self, threshold: float) -> list[tuple[int, Glyph]]:
        return [(i, g) for i, g in enumerate(self.glyphs) if g.conf < threshold]


# --------------------------------------------------------------------------
# 引擎接口
# --------------------------------------------------------------------------

class OCREngine(Protocol):
    name: str

    def recognize(self, source_path: str, source_id: str) -> OCRResult: ...


class PlainTextEngine:
    """离线演示引擎：读取已备好的文本，按字形特征合成可复现的置信度。

    这不是"假 OCR"，而是把真实引擎的输出接口固定下来，使下游流水线
    可以在无 GPU / 无网络环境下被完整测试。生产环境替换为 PaddleOCR、
    Kraken、CnOCR 或商用古籍 OCR，只需实现同一个 recognize 接口。
    """

    name = "plaintext-demo"

    #  刻本中最易误识的字，人为压低置信度，用于触发视觉复核
    _HARD = set("眚省己已巳戊戌戍曰日未末土士氏民傳傅祗祇衹脩修")

    def __init__(self, base_conf: float = 0.985, seed: str = "guji"):
        self.base_conf = base_conf
        self.seed = seed

    def _conf(self, ch: str, pos: int) -> float:
        if ch in LACUNA_MARKS:
            return 0.05
        h = hashlib.md5(f"{self.seed}:{ch}:{pos}".encode()).digest()
        jitter = (h[0] / 255.0) * 0.03
        conf = self.base_conf - jitter
        if ch in self._HARD:
            conf = min(conf, 0.58 + (h[1] / 255.0) * 0.12)
        if ch in DITTO_MARKS:
            conf = 0.45
        return round(conf, 4)

    def recognize(self, source_path: str, source_id: str) -> OCRResult:
        with open(source_path, encoding="utf-8") as f:
            raw = f.read()
        pages: list[OCRPage] = []
        for pno, chunk in enumerate(raw.split("\n\n"), start=1):
            glyphs, line, col = [], 1, 1
            for pos, ch in enumerate(chunk):
                if ch == "\n":
                    line, col = line + 1, 1
                    continue
                if ch == "\r":
                    continue
                g = Glyph(char=ch, conf=self._conf(ch, pos), page=pno,
                          line=line, col=col,
                          bbox=(col * 32, line * 34, col * 32 + 30, line * 34 + 32))
                if g.conf < 0.7:
                    g.alternatives = _shape_neighbours(ch)
                glyphs.append(g)
                col += 1
            if glyphs:
                pages.append(OCRPage(page=pno, glyphs=glyphs))
        return OCRResult(source_id=source_id, pages=pages, engine=self.name)


def _shape_neighbours(ch: str) -> list[tuple[str, float]]:
    from .normalize import CONFUSABLE_INDEX
    return [(c, 0.2) for c in sorted(CONFUSABLE_INDEX.get(ch, set()))][:3]


class PaddleOCREngine:
    """生产引擎示例（需 `pip install paddleocr`，并加载古籍字典与竖排检测模型）。"""

    name = "paddleocr-guji"

    def __init__(self, **kwargs):
        from paddleocr import PaddleOCR  # noqa: F401  延迟导入
        self._ocr = PaddleOCR(lang="chinese_cht", use_angle_cls=True, **kwargs)

    def recognize(self, source_path: str, source_id: str) -> OCRResult:
        raise NotImplementedError(
            "请在此接入实际版面分析（竖排切列 -> 切字 -> 识别 -> 夹注分层）。"
            "关键：必须保留每字 bbox，否则视觉复核无法回原图定位。"
        )


# --------------------------------------------------------------------------
# 视觉复核（VLM）
# --------------------------------------------------------------------------

RECHECK_PROMPT = """你是古籍版本鉴定专家，正在核对一幅**刻本书影局部截图**中的单个字。

已知信息：
- OCR 首选结果：{char}（置信度 {conf:.2f}）
- OCR 备选：{alts}
- 上下文（OCR 结果，可能含误）：……{left}【{char}】{right}……

请严格按下列要求判断：
1. 只描述你在图像中**实际看到的笔画结构**，不要用上下文去猜字。
2. 若字口漫漶、虫蛀、界栏压字导致无法辨认，请输出 char 为 "□"。
3. 注意避讳缺笔：末笔缺失可能是避讳而非讹字，请在 note 中指出。
4. 注意重文号「々」「〻」：形如两点或小「二」，不可展开为具体字。

仅输出 JSON：
{{"char": "…", "confidence": 0.0-1.0, "is_lacuna": false,
  "taboo_stroke_omission": false, "note": "不超过40字"}}"""


class VisionRechecker:
    """把低置信字回原图裁切，交视觉模型复核。

    离线模式下使用内置的形近字先验做确定性复核，用于演示与回归测试；
    传入 client 后即走真实多模态模型。
    """

    def __init__(self, client=None, crop_pad: int = 24, offline_truth: dict | None = None):
        self.client = client
        self.crop_pad = crop_pad
        self.offline_truth = offline_truth or {}

    def crop(self, image_path: str, bbox) -> str:
        """裁切并放大局部（返回临时文件路径）。生产环境用 PIL 实现。"""
        x0, y0, x1, y1 = bbox
        return f"{image_path}#crop={x0 - self.crop_pad},{y0 - self.crop_pad}," \
               f"{x1 + self.crop_pad},{y1 + self.crop_pad}"

    def recheck(self, result: OCRResult, threshold: float = 0.75) -> list[dict]:
        glyphs = result.glyphs
        events: list[dict] = []
        for i, g in result.low_confidence(threshold):
            left = "".join(x.char for x in glyphs[max(0, i - 6):i])
            right = "".join(x.char for x in glyphs[i + 1:i + 7])
            verdict = self._ask(g, left, right)
            if not verdict:
                continue
            changed = verdict["char"] != g.char
            events.append({
                "index": i, "before": g.char, "after": verdict["char"],
                "changed": changed, "ocr_conf": g.conf,
                "vlm_conf": verdict["confidence"], "note": verdict.get("note", ""),
                "context": f"{left}【{g.char}】{right}",
            })
            g.char = verdict["char"]
            g.conf = fuse_confidence(g.conf, verdict["confidence"])
            g.rechecked = True
        return events

    def _ask(self, g: Glyph, left: str, right: str) -> dict | None:
        if self.client is None:
            key = f"{left[-3:]}|{g.char}"
            if key in self.offline_truth:
                t = self.offline_truth[key]
                return {"char": t["char"], "confidence": t.get("conf", 0.93),
                        "note": t.get("note", "書影覆核")}
            if g.char in LACUNA_MARKS:
                return {"char": "□", "confidence": 0.9, "note": "字口漫漶，存疑待補"}
            return None
        prompt = RECHECK_PROMPT.format(
            char=g.char, conf=g.conf,
            alts="、".join(f"{c}({p})" for c, p in g.alternatives) or "無",
            left=left, right=right)
        raw = self.client.complete_vision(prompt, image_ref=g.bbox)
        try:
            return json.loads(re.sub(r"```(json)?", "", raw).strip())
        except Exception:
            return None


def fuse_confidence(*probs: float) -> float:
    r"""独立证据的对数几率融合（Naive Bayes 形式）：

        \ell = \sum_i \ln \frac{p_i}{1-p_i},\quad p = \sigma(\ell)

    用于把 OCR 通道 A、通道 B、视觉复核三方的置信度合成为一个后验。
    """
    eps = 1e-6
    ell = 0.0
    for p in probs:
        p = min(max(p, eps), 1 - eps)
        ell += math.log(p / (1 - p))
    return round(1 / (1 + math.exp(-ell)), 4)


def dual_channel(result_a: OCRResult, result_b: OCRResult) -> list[dict]:
    """双引擎逐字比对，返回不一致位置（这些位置必进视觉复核队列）。"""
    ga, gb = result_a.glyphs, result_b.glyphs
    out = []
    for i, (x, y) in enumerate(zip(ga, gb)):
        if x.char != y.char:
            out.append({"index": i, "engine_a": x.char, "engine_b": y.char,
                        "conf_a": x.conf, "conf_b": y.conf})
    if len(ga) != len(gb):
        out.append({"index": min(len(ga), len(gb)),
                    "engine_a": f"<len={len(ga)}>", "engine_b": f"<len={len(gb)}>",
                    "conf_a": 0.0, "conf_b": 0.0})
    return out


def load_or_ocr(path: str, source_id: str, engine: OCREngine | None = None) -> OCRResult:
    engine = engine or PlainTextEngine()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return engine.recognize(path, source_id)
