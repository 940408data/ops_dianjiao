"""两模型合议对比：逐条比对 deepseek / glm 的 chosen 分歧与悬置差异。"""
import sys, json
from collections import Counter
sys.path.insert(0, ".")

ROOT = "outputs"
ds = json.load(open(f"{ROOT}/daxue-ds/run_report.json", encoding="utf-8"))
glm = json.load(open(f"{ROOT}/daxue-glm/run_report.json", encoding="utf-8"))
dsv = {v["div_id"]: v for v in ds["verdicts"]}
glv = {v["div_id"]: v for v in glm["verdicts"]}
ids = sorted(set(dsv) & set(glv))

agree = 0
rows = []
for i in ids:
    a, b = dsv[i]["chosen"], glv[i]["chosen"]
    if a == b:
        agree += 1
    else:
        rows.append({"id": i, "type": dsv[i]["dtype"],
                     "base": dsv[i]["base_text"], "ref": dsv[i]["ref_text"],
                     "deepseek": a, "glm": b,
                     "ds_susp": dsv[i]["suspended"], "glm_susp": glv[i]["suspended"]})

disagree = len(rows)
print(f"两模型共议 {len(ids)} 条：chosen 一致 {agree}（{agree/len(ids):.1%}），"
      f"分歧 {disagree}（{disagree/len(ids):.1%}）\n")

tc = Counter(r["type"] for r in rows)
print("分歧条目类型分布：", dict(tc), "\n")

ds_susp = sum(1 for i in ids if dsv[i]["suspended"])
glm_susp = sum(1 for i in ids if glv[i]["suspended"])
both_susp = sum(1 for i in ids if dsv[i]["suspended"] and glv[i]["suspended"])
print(f"悬置：deepseek {ds_susp} 条，glm {glm_susp} 条，共同悬置 {both_susp} 条\n")

print("=== 全部分歧条目（div_id 类型 | 底本/今本 | deepseek / glm）===")
for r in rows:
    flag = ""
    if r["ds_susp"] and not r["glm_susp"]:
        flag = " [ds悬/glm定]"
    if r["glm_susp"] and not r["ds_susp"]:
        flag = " [ds定/glm悬]"
    print(f"{r['id']} [{r['type']}] 底={r['base'][:10]!r} 今={r['ref'][:10]!r} | "
          f"ds={r['deepseek']!r} glm={r['glm']!r}{flag}")

json.dump({"共议": len(ids), "一致": agree, "分歧": disagree,
           "分歧率": round(disagree / len(ids), 4),
           "deepseek悬置": ds_susp, "glm悬置": glm_susp, "共同悬置": both_susp,
           "分歧类型分布": dict(tc), "分歧明细": rows},
          open(f"{ROOT}/compare_report.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n报告已存 {ROOT}/compare_report.json")
