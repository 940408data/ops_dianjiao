"""测速：deepseek/glm × 开/关思维链，看 content 是否非空、单条耗时。"""
import sys, time
sys.path.insert(0, ".")
from guji_collator.agents import OpenAICompatClient, LIU_XIANG, CASE_PROMPT, _parse_json
from guji_collator.config import Config
from guji_collator.normalize import normalize
from guji_collator.align import collate

cfg = Config.load("data/daxue/task-ds.json")
base = normalize(open(cfg.base.path, encoding="utf-8").read())
modern = normalize(open(cfg.modern.path, encoding="utf-8").read())
d = collate(base, modern)[0]
user = CASE_PROMPT.format(
    did=d.id, dtype=d.dtype.value, base_name="底本", ref_name="今本",
    left=d.left_ctx, right=d.right_ctx, base_text=d.base_text or "〇",
    ref_text=d.ref_text or "〇", auto_note=d.auto_note, conf=d.ocr_conf,
    cands="、".join(d.candidates) or "（無）")


def trial(model, mt, extra, label):
    c = OpenAICompatClient(cfg.base_url, model, max_tokens=mt)
    t = time.time()
    try:
        raw = c.complete(LIU_XIANG.charter, user, **extra)
    except Exception as e:
        print(f"{label}: 调用异常 {e}")
        return
    dt = time.time() - t
    p = _parse_json(raw) if raw else None
    ok = "Y" if p else "N"
    ch = p.get("choice") if p else "-"
    print(f"{label}: {dt:.1f}s  content={len(raw)}  JSON={ok}  choice={ch!r}")
    if p:
        print(f"    arg: {p.get('argument','')[:60]}")


trial("deepseek-v4-flash", 4096, {}, "deepseek 开思维链 4096")
trial("deepseek-v4-flash", 4096, {"enable_thinking": False}, "deepseek 关思维链 4096")
trial("glm-5.2", 16384, {}, "glm 开思维链 16384")
trial("glm-5.2", 4096, {"enable_thinking": False}, "glm 关思维链 4096")
