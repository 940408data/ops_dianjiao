"""实测各 qwen 模型的视觉能力：发一页大学章句书影，看能否识别古籍内容。"""
import os, json, base64, urllib.request, glob

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = os.environ.get("OPENAI_API_KEY", "")

img_path = sorted(glob.glob("/tmp/zy_page0002*.png"))[0]
raw = open(img_path, "rb").read()
b64 = base64.b64encode(raw).decode()
print(f"图片 {img_path}: {len(raw)} bytes, base64 {len(b64)} chars\n")

PROMPT = ("这是一页古籍书影（竖排繁体）。请简明回答："
          "1. 这是什么书的什么篇章？2. 尽量多地识别其中文字（按列从右至左）。"
          "3. 有无漫漶、缺字、避讳缺笔之处？")


def ask(model):
    body = json.dumps({
        "model": model, "max_tokens": 1500,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"content-type": "application/json", "authorization": f"Bearer {KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        msg = data["choices"][0]["message"]
        return msg.get("content") or f"(content空; reasoning前100: {str(msg.get('reasoning_content',''))[:100]})"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return f"ERR: {e}"


for m in ["qwen3.7-plus", "qwen3.8-max", "qwen3.6-plus", "qwen-vl-plus"]:
    print(f"===== {m} =====")
    print(ask(m))
    print()
