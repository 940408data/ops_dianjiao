"""guji_collator —— AI 古籍校勘流水线。

用法：
    from guji_collator import Config, run
    cfg = Config.load("task.json")
    run(cfg)
"""
from .config import Config, SourceSpec          # noqa: F401
from .pipeline import run, apply_decisions      # noqa: F401

__version__ = "0.3.0"
