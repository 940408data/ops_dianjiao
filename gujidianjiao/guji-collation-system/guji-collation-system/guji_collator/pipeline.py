"""流水线编排

    S0 立项與權屬核驗  ->  S1 雙通道 OCR + 視覺覆核  ->  S2 規範化與對齊
    ->  S3 異文檢測分類  ->  S4 校書官合議  ->  S5 裁斷與懸置
    ->  S6 出雙本（公開本／內部本）  ->  S7 人工精校 -> 回灌重出

每一阶段都落盘中间产物，任何一步都可单独重跑；全流程可复现（同输入同输出）。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

from . import adjudicate as adj
from .agents import Council, AnthropicClient, OpenAICompatClient, load_citations
from .align import collate, project_punctuation, Divergence, DivType
from .config import Config, SourceSpec
from .normalize import normalize
from .ocr import PlainTextEngine, VisionRechecker, load_or_ocr, dual_channel
from .render import (RuleSegmenter, Mark, punctuation_overlap, build_base_edition,
                     build_modern_internal, edition_to_markdown, edition_to_html)
from .review import build_payload, build_review_app


@dataclass
class RunResult:
    run_id: str
    out_dir: str
    stats: dict = field(default_factory=dict)
    files: dict = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def _log(step: str, msg: str) -> None:
    print(f"[{step}] {msg}", flush=True)


def _client(cfg: Config):
    if cfg.llm == "anthropic":
        return AnthropicClient(model=cfg.model)
    if cfg.llm == "openai-compat":
        # 推理模型关思维链：全书 616 次调用，开思维链 glm 单条 ~16s（几小时，不可行）；
        # 关闭后 ~2s 且 content 不被截空（无需大 max_tokens），全书 ~20 分钟可跑完。
        return OpenAICompatClient(cfg.base_url, cfg.model, max_tokens=4096,
                                  enable_thinking=False)
    return None


def run(cfg: Config, offline_recheck_truth: dict | None = None) -> RunResult:
    t0 = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%S")
    out = os.path.join(cfg.out_dir, cfg.work)
    pub = os.path.join(out, "public")
    inte = os.path.join(out, "internal")
    work_dir = os.path.join(out, "work")
    for d in (pub, inte, work_dir):
        os.makedirs(d, exist_ok=True)
    alerts: list[str] = []

    # ---------------- S0 权属核验 ----------------
    _log("S0", f"校書任務：{cfg.work}")
    if cfg.base.rights != "public-domain":
        alerts.append("底本權屬非公有領域，不得產出公開本。")
        cfg.publish_public_edition = False
    if cfg.modern and cfg.modern.rights == "copyrighted":
        _log("S0", "現代點校本標記為受著作權保護：其產物僅入 internal/，且句讀不得直接落公開本。")
    provenance = {
        "run_id": run_id,
        "base": {**cfg.base.__dict__, "sha256_16": _sha(cfg.base.path)},
        "modern": ({**cfg.modern.__dict__, "sha256_16": _sha(cfg.modern.path)}
                   if cfg.modern else None),
        "config": cfg.to_dict(),
    }

    # ---------------- S1 OCR ----------------
    engine = PlainTextEngine()
    base_ocr = load_or_ocr(cfg.base.path, cfg.base.id, engine)
    _log("S1", f"底本 OCR：{len(base_ocr.glyphs)} 字，引擎 {base_ocr.engine}")

    if cfg.dual_channel:
        engine_b = PlainTextEngine(base_conf=0.978, seed="channel-b")
        base_ocr_b = load_or_ocr(cfg.base.path, cfg.base.id + "-b", engine_b)
        conflicts = dual_channel(base_ocr, base_ocr_b)
        _log("S1", f"雙通道逐字比對：分歧 {len(conflicts)} 處")
    else:
        conflicts = []

    rechecker = VisionRechecker(client=None, offline_truth=offline_recheck_truth or {})
    recheck_events = rechecker.recheck(base_ocr, threshold=cfg.recheck_threshold)
    changed = [e for e in recheck_events if e["changed"]]
    _log("S1", f"視覺覆核：送檢 {len(recheck_events)} 字，改正 {len(changed)} 字")

    base_raw = base_ocr.text
    with open(os.path.join(work_dir, "base_after_recheck.txt"), "w", encoding="utf-8") as f:
        f.write(base_raw)

    modern_raw = ""
    if cfg.modern:
        modern_ocr = load_or_ocr(cfg.modern.path, cfg.modern.id, engine)
        modern_raw = modern_ocr.text
        _log("S1", f"今本 OCR：{len(modern_ocr.glyphs)} 字（含標點）")

    # ---------------- S2 规范化与对齐 ----------------
    base_n = normalize(base_raw)
    ref_n = normalize(modern_raw) if modern_raw else normalize("")
    _log("S2", f"CNF 投影：底本 {len(base_n.cnf)} 字，今本 {len(ref_n.cnf)} 字")

    # 置信度回填：base_raw 含换行，glyphs 不含，必须按非换行位逐一对齐，
    # 否则第二行以后的每个字都会拿到错位的置信度（曾致低置信字漏检）。
    raw_conf: dict[int, float] = {}
    _gi = 0
    for _ri, _ch in enumerate(base_raw):
        if _ch == "\n":
            continue
        if _gi < len(base_ocr.glyphs):
            raw_conf[_ri] = base_ocr.glyphs[_gi].conf
        _gi += 1

    def conf_lookup(cnf_idx: int) -> float:
        if cnf_idx >= len(base_n.index_map):
            return 1.0
        return raw_conf.get(base_n.index_map[cnf_idx], 1.0)

    # ---------------- S3 异文检测 ----------------
    divs = collate(base_n, ref_n, conf_lookup=conf_lookup) if modern_raw else []
    by_type: dict[str, int] = {}
    for d in divs:
        by_type[d.dtype.value] = by_type.get(d.dtype.value, 0) + 1
    _log("S3", f"檢出異文 {len(divs)} 處：" +
         "，".join(f"{k} {v}" for k, v in by_type.items()))

    # ---------------- S4 合议 ----------------
    load_citations(cfg.citations_db or None)
    council = Council(client=_client(cfg), base_name=cfg.base.label,
                      ref_name=cfg.modern.label if cfg.modern else "今本")
    all_opinions = {}
    verdicts: list[adj.Verdict] = []
    for d in divs:
        ops = council.deliberate(d)
        all_opinions[d.id] = [o.to_dict() for o in ops]
        chair = council.chair_review(d, ops)
        v = adj.adjudicate(d, ops, chair_json=chair, tau=cfg.tau, delta=cfg.delta)
        v._span = (d.base_start, d.base_end)
        verdicts.append(v)
    _log("S4", f"合議完成：{len(verdicts)} 案，三家共發議 {sum(len(v) for v in all_opinions.values())} 條")

    # ---------------- S5 裁断统计 ----------------
    stats = adj.summarize(verdicts)
    _log("S5", f"裁斷：改字 {stats['amended']}，仍舊 {stats['kept']}，"
               f"懸置 {stats['suspended']}，自動決議率 {stats['auto_resolution_rate']:.1%}")

    # ---------------- S6 出双本 ----------------
    marks = RuleSegmenter().segment(base_n.cnf, base_n.line_break_after)
    modern_punct = project_punctuation(base_n, ref_n) if modern_raw else {}
    overlap = punctuation_overlap(marks, modern_punct)
    if overlap["warn"]:
        alerts.append(f"公開本句讀與今本重合率 {overlap['jaccard']:.2f}，"
                      f"超過閾值 {cfg.punct_overlap_alarm}，須人工重新斷句以避免著作權風險。")
    _log("S6", f"獨立斷句 {overlap['ours']} 處；與今本重合率 {overlap['jaccard']:.2f}")

    ed_pub = build_base_edition(base_n, verdicts, marks,
                                title=f"{cfg.title or cfg.work}（{cfg.base.label}點校本）")
    ed_pub.stats = {**stats, "punctuation": overlap}
    files = {}

    if cfg.publish_public_edition:
        p_md = os.path.join(pub, "critical_edition.md")
        p_html = os.path.join(pub, "critical_edition.html")
        with open(p_md, "w", encoding="utf-8") as f:
            f.write(edition_to_markdown(ed_pub, public=True))
        with open(p_html, "w", encoding="utf-8") as f:
            f.write(edition_to_html(
                ed_pub, public=True,
                subtitle=f"底本：{cfg.base.repository}　{cfg.base.edition_note}　"
                         f"｜ 本次運行 {run_id}"))
        files["public_md"] = p_md
        files["public_html"] = p_html

    if modern_raw:
        ed_int = build_modern_internal(modern_raw, verdicts, cfg.title or cfg.work)
        i_md = os.path.join(inte, "modern_reference.md")
        with open(i_md, "w", encoding="utf-8") as f:
            f.write(edition_to_markdown(ed_int, public=False))
        files["internal_md"] = i_md

    # ---------------- S7 人工精校台 ----------------
    payload = build_payload(cfg.title or cfg.work, run_id, verdicts, divs)
    r_html = os.path.join(out, "review.html")
    with open(r_html, "w", encoding="utf-8") as f:
        f.write(build_review_app(payload))
    files["review_html"] = r_html

    report = {
        "provenance": provenance,
        "ocr": {"glyphs": len(base_ocr.glyphs), "dual_channel_conflicts": len(conflicts),
                "recheck_sent": len(recheck_events), "recheck_changed": len(changed),
                "recheck_events": recheck_events},
        "divergences": [d.to_dict() for d in divs],
        "verdicts": [v.to_dict() for v in verdicts],
        "stats": stats, "punctuation": overlap, "alerts": alerts,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    r_json = os.path.join(out, "run_report.json")
    with open(r_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    files["run_report"] = r_json

    pending = [v.to_dict() for v in verdicts if v.suspended]
    p_json = os.path.join(out, "pending_review.json")
    with open(p_json, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "count": len(pending), "items": pending},
                  f, ensure_ascii=False, indent=2)
    files["pending"] = p_json

    for a in alerts:
        _log("!!", a)
    _log("S7", f"人工精校台已生成，待覆核 {len(pending)} 條 -> {r_html}")
    return RunResult(run_id=run_id, out_dir=out, stats=stats, files=files, alerts=alerts)


def apply_decisions(cfg: Config, decisions_path: str,
                    offline_recheck_truth: dict | None = None) -> RunResult:
    """把人工精校台导出的 decisions.json 回灌，重出定本。"""
    with open(decisions_path, encoding="utf-8") as f:
        payload = json.load(f)
    decisions = payload.get("decisions", {})
    _log("S7", f"回灌人工決策 {len(decisions)} 條")

    res = run(cfg, offline_recheck_truth=offline_recheck_truth)
    if payload.get("run_id") not in ("manual", res.run_id):
        _log("!!", f"decisions.json 來自運行 {payload.get('run_id')}，"
                   f"與本次 {res.run_id} 不同；若中間改過 OCR 或配置，條目編號可能位移。")
    with open(res.files["run_report"], encoding="utf-8") as f:
        report = json.load(f)

    base_n = normalize(open(os.path.join(res.out_dir, "work",
                                         "base_after_recheck.txt"), encoding="utf-8").read())
    divs = [Divergence(**{**d, "dtype": DivType(d["dtype"])}) for d in report["divergences"]]
    dmap = {d.id: d for d in divs}

    verdicts = []
    for vd in report["verdicts"]:
        v = adj.Verdict(**vd)
        hit = decisions.get(v.div_id)
        if hit:
            v.chosen = hit["choice"]
            v.suspended = False
            v.decided_by = "human"
            v.human_note = hit.get("note", "")
            v.amend_base_text = (v.chosen not in ("兩存", v.base_text))
            v.apparatus_note = v.apparatus_note.replace("諸說未一，姑兩存之，俟考。", "").rstrip()
            if v.human_note:
                v.apparatus_note += f"　覆核案：{v.human_note}。"
            if v.amend_base_text:
                v.apparatus_note += ("今據刪。" if v.chosen == ""
                                     else f"今據改作「{v.chosen}」。")
            elif v.chosen == "兩存":
                v.apparatus_note += "覆核後仍兩存之。"
            else:
                v.apparatus_note += "今仍底本之舊，不改。"
        d = dmap.get(v.div_id)
        if d:
            v._span = (d.base_start, d.base_end)
        verdicts.append(v)

    marks = RuleSegmenter().segment(base_n.cnf, base_n.line_break_after)
    ed = build_base_edition(base_n, verdicts, marks,
                            title=f"{cfg.title or cfg.work}（{cfg.base.label}點校定本）")
    ed.stats = adj.summarize(verdicts)
    pub = os.path.join(res.out_dir, "public")
    f_md = os.path.join(pub, "critical_edition.final.md")
    f_html = os.path.join(pub, "critical_edition.final.html")
    with open(f_md, "w", encoding="utf-8") as f:
        f.write(edition_to_markdown(ed, public=True))
    with open(f_html, "w", encoding="utf-8") as f:
        f.write(edition_to_html(ed, public=True, subtitle="人工精校定本"))
    res.files["final_md"], res.files["final_html"] = f_md, f_html
    res.stats = ed.stats
    _log("S7", f"定本已出：{f_html}")
    return res
