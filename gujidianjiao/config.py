"""运行配置。

一份配置对应一次"校书任务"，可版本化入库，保证结果可复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


@dataclass
class SourceSpec:
    """一个文献见证（witness）。"""
    id: str
    label: str
    path: str
    kind: str = "base"          # base（底本善本） | modern（现代点校本） | witness（他校本）
    rights: str = "public-domain"   # public-domain | copyrighted | unknown
    repository: str = ""
    edition_note: str = ""
    image_dir: str = ""

    @property
    def publishable(self) -> bool:
        return self.rights == "public-domain"


@dataclass
class Config:
    work: str
    title: str = ""
    base: SourceSpec = None  # type: ignore
    modern: SourceSpec | None = None
    witnesses: list[SourceSpec] = field(default_factory=list)

    out_dir: str = "outputs"
    citations_db: str = ""

    # OCR
    ocr_engine: str = "plaintext"
    recheck_threshold: float = 0.78
    dual_channel: bool = True

    # 裁决
    tau: float = 0.55
    delta: float = 0.18
    base_preference_beta: float = 0.15

    # 智能体
    llm: str = "offline"        # offline | anthropic | openai-compat
    model: str = "claude-sonnet-4-6"
    base_url: str = ""
    council: list[str] = field(default_factory=lambda: ["liuxiang", "xiejin", "daizhen"])
    chair: str = "jiyun"

    # 合规
    publish_public_edition: bool = True
    punct_independent: bool = True      # 公开本句读必须独立生成
    punct_overlap_alarm: float = 0.92

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        base = SourceSpec(**raw.pop("base"))
        modern = raw.pop("modern", None)
        wit = raw.pop("witnesses", [])
        return cls(base=base,
                   modern=SourceSpec(**modern) if modern else None,
                   witnesses=[SourceSpec(**w) for w in wit], **raw)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
