"""真·视觉复核：对悬置的形近字，定位其在当涂郡本的书影页，
pdftoppm 转成 PNG，送 qwen3.7-plus 看图断字（不是文本近似）。

用法：
  OPENAI_API_KEY=sk-... python3 qwen_visual_recheck.py [run_report.json]
默认读 outputs/daxue-zhangju/run_report.json（离线版，含悬置形近字）。
"""
import os, json, base64, urllib.request, glob, subprocess, re, sys, time

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = os.environ.get("OPENAI_API_KEY", "")
QWEN = "qwen3.7-plus"

REPORT = sys.argv[1] if len(sys.argv) > 1 else "outputs/daxue-zhangju/run_report.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/qwen_visual_recheck.json"
BOOK_PDF_DIR = "/root/ops_dianjiao/大学章句/当涂郡本_pdf"      # 真书影（主仓库）
BOOK_OCR_GLOB = "/root/ops_dianjiao/大学章句/当涂郡本_ocr/page_*.md"

rep = json.load(open(REPORT, encoding="utf-8"))
dmap = {d["id"]: d for d in rep["divergences"]}
targets = [v for v in rep["verdicts"] if v["suspended"] and v["dtype"] == "形近疑誤"]
print(f"报告 {REPORT}：悬置形近字 {len(targets)} 条\n")

pages = {p: open(p, encoding="utf-8").read().replace("\n", "")
         for p in glob.glob(BOOK_OCR_GLOB)}


def locate(d):
    for needle in (d["left_ctx"][-4:] + d["base_text"] + d["right_ctx"][:4],
                   d["left_ctx"][-3:] + d["base_text"],
                   d["base_text"]):
        if not needle:
            continue
        for p, txt in pages.items():
            if needle in txt:
                return p
    return None


_pdf_cache = {}


def page_png(ocr_md_path):
    pno = re.search(r"page_(\d+)", ocr_md_path).group(1)
    if pno in _pdf_cache:
        return _pdf_cache[pno]
    pdf = os.path.join(BOOK_PDF_DIR, f"page_{pno}.pdf")
    out = f"/tmp/recheck_{pno}"
    subprocess.run(["pdftoppm", "-png", "-r", "110", pdf, out], check=True)
    png = sorted(glob.glob(out + "*.png"))[0]
    _pdf_cache[pno] = png
    return png


PROMPT = ("這是一頁古籍書影（豎排繁體）。其中某處 OCR 識作「{base}」（置信度 {conf:.2f}），"
          "今本對應處作「{ref}」。該字上下文（OCR 文本，可能含誤）：……{left}【{base}】{right}……\n"
          "請你在書影中找到該字，細看其字形筆畫，判斷 OCR 識得對不對、該字應作何字。"
          "若書影該處漫漶難辨，答「□」。\n"
          "僅輸出 JSON：{{\"char\": \"…\", \"confidence\": 0.0-1.0, \"note\": \"不超過40字，描述書影實際所見字形\"}}")


def ask(png, prompt):
    b64 = base64.b64encode(open(png, "rb").read()).decode()
    body = json.dumps({"model": QWEN, "max_tokens": 1500, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}).encode()
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body,
        headers={"content-type": "application/json", "authorization": f"Bearer {KEY}"})
    for att in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"].get("content") or ""
        except Exception:
            time.sleep(2 * (att + 1))
    return ""


results = {}
for v in targets:
    d = dmap.get(v["div_id"])
    if not d:
        continue
    page = locate(d)
    if not page:
        results[v["div_id"]] = {"base": d["base_text"], "error": "定位不到书影页"}
        print(f"{v['div_id']} 底={d['base_text']!r}: 定位不到书影页")
        continue
    png = page_png(page)
    prompt = PROMPT.format(base=d["base_text"] or "〇", ref=d["ref_text"] or "〇",
                           conf=d["ocr_conf"], left=d["left_ctx"], right=d["right_ctx"])
    raw = ask(png, prompt)
    m = re.search(r"\{.*\}", re.sub(r"```(json)?", "", raw), re.S)
    parsed = None
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            pass
    results[v["div_id"]] = {
        "base": d["base_text"], "ref": d["ref_text"], "ocr_conf": d["ocr_conf"],
        "page": os.path.basename(page),
        "qwen_char": parsed.get("char") if parsed else None,
        "qwen_conf": parsed.get("confidence") if parsed else None,
        "qwen_note": (parsed.get("note", "") if parsed else raw[:60]),
    }
    qc = parsed.get("char") if parsed else "?"
    qn = (parsed.get("note", "") if parsed else raw[:40])
    print(f"{v['div_id']} 底={d['base_text']!r} 今={d['ref_text']!r} [{os.path.basename(page)}] "
          f"qwen→{qc!r} ({qn})")

json.dump(results, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n已存 {OUT}")
