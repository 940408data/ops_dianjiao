"""形近字三方仲裁：deepseek / glm / qwen3.7 书影 对齐。

形近疑误是 OCR 误识问题，qwen3.7 看书影最准，作 ground truth。
qwen 书影结果取离线复核 outputs/qwen_visual_recheck.json（覆盖全部形近字）。
"""
import json

ds = json.load(open('outputs/daxue-ds/run_report.json', encoding='utf-8'))
glm = json.load(open('outputs/daxue-glm/run_report.json', encoding='utf-8'))
q = json.load(open('outputs/qwen_visual_recheck.json', encoding='utf-8'))
dsv = {v['div_id']: v for v in ds['verdicts']}
glv = {v['div_id']: v for v in glm['verdicts']}

shape_ids = sorted(v['div_id'] for v in ds['verdicts'] if v['dtype'] == '形近疑誤')
print(f"形近疑誤共 {len(shape_ids)} 条\n")
print(f"{'id':6}{'底':3}{'今':3} | {'deepseek':12}{'glm':12}{'qwen书影':5} | ds对 glm对")

ds_ok = glm_ok = n = 0
for i in shape_ids:
    d, g = dsv[i], glv[i]
    base, ref = d['base_text'], d['ref_text']
    dsc, glc = d['chosen'], g['chosen']
    qw = q.get(i, {}).get('qwen_char', '-')
    # 仲裁：qwen 书影为准；模型 chosen 含该字即算对
    d_ok = (qw != '-' and qw in dsc)
    g_ok = (qw != '-' and qw in glc)
    if qw != '-':
        n += 1
        ds_ok += d_ok
        glm_ok += g_ok
    print(f"{i:6}{base:3}{ref:3} | {dsc:12}{glc:12}{qw:5} | {'✓' if d_ok else '✗'}    {'✓' if g_ok else '✗'}")

print(f"\n以 qwen3.7 书影为准（{n} 条有效）：deepseek 判对 {ds_ok}，glm 判对 {glm_ok}")
