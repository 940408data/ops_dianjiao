"""冒烟测试：对大学章句第一条异文，分别让 deepseek-v4-flash / glm-5.2
以劉向 charter 发议，看 content 是否非空、能否解出 JSON。"""
import sys
sys.path.insert(0, ".")
from guji_collator.agents import OpenAICompatClient, LIU_XIANG, CASE_PROMPT, _parse_json
from guji_collator.config import Config
from guji_collator.normalize import normalize
from guji_collator.align import collate

cfg = Config.load("data/daxue/task-ds.json")
base = normalize(open(cfg.base.path, encoding="utf-8").read())
modern = normalize(open(cfg.modern.path, encoding="utf-8").read())
divs = collate(base, modern)
d = divs[0]
user = CASE_PROMPT.format(
    did=d.id, dtype=d.dtype.value, base_name="底本", ref_name="今本",
    left=d.left_ctx, right=d.right_ctx,
    base_text=d.base_text or "〇", ref_text=d.ref_text or "〇",
    auto_note=d.auto_note, conf=d.ocr_conf,
    cands="、".join(d.candidates) or "（無）")
print(f"异文 {d.id} 类型={d.dtype.value} 底本={d.base_text!r} 今本={d.ref_text!r}\n")

MT = {"deepseek-v4-flash": 4096, "glm-5.2": 16384}
for model in ["deepseek-v4-flash", "glm-5.2"]:
    c = OpenAICompatClient(cfg.base_url, model, max_tokens=MT[model])
    raw = c.complete(LIU_XIANG.charter, user)
    parsed = _parse_json(raw) if raw else None
    print(f"=== {model} | content长度={len(raw)} | JSON解析={'成功' if parsed else '失败'} ===")
    if parsed:
        print(f"  choice={parsed.get('choice')!r} conf={parsed.get('confidence')} "
              f"grade={parsed.get('evidence_grade')!r} amend={parsed.get('amend_base_text')}")
        print(f"  arg: {parsed.get('argument','')[:120]}")
    else:
        print(f"  raw前200字: {raw[:200]!r}")
    print()
