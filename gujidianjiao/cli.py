"""命令行入口

    python -m guji_collator run     --task data/demo/task.json
    python -m guji_collator apply   --task data/demo/task.json --decisions decisions.json
    python -m guji_collator inspect --task data/demo/task.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Config
from .pipeline import run, apply_decisions

DEMO_RECHECK_TRUTH = {
    # 视觉复核的离线基准：key = "前三字|OCR首选字"
    "吾日三|眚": {"char": "省", "conf": 0.94,
                "note": "書影字口作「省」，上部從少不從屮，OCR 誤識為「眚」"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("guji_collator", description="AI 古籍校勘流水線")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="执行完整校勘流程")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--llm", choices=["offline", "anthropic", "openai-compat"])
    p_run.add_argument("--tau", type=float)
    p_run.add_argument("--delta", type=float)

    p_ap = sub.add_parser("apply", help="回灌人工决策，重出定本")
    p_ap.add_argument("--task", required=True)
    p_ap.add_argument("--decisions", required=True)

    p_in = sub.add_parser("inspect", help="只做对齐与异文检测，不合议")
    p_in.add_argument("--task", required=True)

    a = ap.parse_args(argv)
    cfg = Config.load(a.task)
    root = os.path.dirname(os.path.abspath(a.task))
    # 任务文件中的相对路径以任务文件所在的工程根为基准
    cwd = os.getcwd()

    if a.cmd == "run":
        if a.llm:
            cfg.llm = a.llm
        if a.tau is not None:
            cfg.tau = a.tau
        if a.delta is not None:
            cfg.delta = a.delta
        res = run(cfg, offline_recheck_truth=DEMO_RECHECK_TRUTH)
        print(json.dumps({"run_id": res.run_id, "stats": res.stats,
                          "files": res.files, "alerts": res.alerts},
                         ensure_ascii=False, indent=2))
        return 0

    if a.cmd == "apply":
        res = apply_decisions(cfg, a.decisions,
                              offline_recheck_truth=DEMO_RECHECK_TRUTH)
        print(json.dumps({"run_id": res.run_id, "stats": res.stats,
                          "files": res.files}, ensure_ascii=False, indent=2))
        return 0

    if a.cmd == "inspect":
        from .normalize import normalize
        from .align import collate
        b = normalize(open(cfg.base.path, encoding="utf-8").read())
        m = normalize(open(cfg.modern.path, encoding="utf-8").read()) if cfg.modern else normalize("")
        divs = collate(b, m)
        for d in divs:
            print(f"{d.id}  {d.dtype.value:<8} {d.display}")
        print(f"\n共 {len(divs)} 處異文")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
