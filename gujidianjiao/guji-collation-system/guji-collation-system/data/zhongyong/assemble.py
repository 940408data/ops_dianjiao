"""把中庸章句两版本逐页 OCR 拼成系统输入文本。

页与页之间用空行（\\n\\n）分隔——这是 PlainTextEngine 的分页符；
页内换行保留（同一页内按行切分 glyph）。

清洗：仅去掉儒藏本的脚注版面符号 ` $ ^{①} $`（纯排版噪声，非校勘对象）。
标点、异体字、避讳字、夹注校记一律保留——后者正是校勘与合议的对象。
"""
import glob
import pathlib
import re

CORPUS = pathlib.Path("/root/ops_dianjiao/中庸章句")
OUT = pathlib.Path("/root/ops_dianjiao/gujidianjiao/guji-collation-system/guji-collation-system/data/zhongyong")
OUT.mkdir(parents=True, exist_ok=True)

FOOTNOTE = re.compile(r"\s*\$\s*\^\{[^}]*\}\s*\$\s*")  # ①② 脚注版面标记


def assemble(pattern: str, clean_footnotes: bool = False) -> str:
    pages = sorted(glob.glob(str(CORPUS / pattern / "page_*.md")))
    chunks = []
    for p in pages:
        text = pathlib.Path(p).read_text(encoding="utf-8").strip("\r\n")
        if clean_footnotes:
            text = FOOTNOTE.sub("", text)
        chunks.append(text)
    return "\n\n".join(chunks)


base = assemble("当涂郡本_ocr")
modern = assemble("儒藏本_ocr", clean_footnotes=True)

(OUT / "base.txt").write_text(base, encoding="utf-8")
(OUT / "modern.txt").write_text(modern, encoding="utf-8")

print(f"当涂郡本(底本)  页={base.count(chr(10)+chr(10))+1:3d}  字符={len(base):5d}  -> {OUT/'base.txt'}")
print(f"儒藏本  (今本)  页={modern.count(chr(10)+chr(10))+1:3d}  字符={len(modern):5d}  -> {OUT/'modern.txt'}")
