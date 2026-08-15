# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库概况

`ops_dianjiao`（古籍点校）包含两个**互不相关、尚未打通**的部分：

1. **数据语料库**（根目录下中文命名的子目录）——四书章句集注（朱熹）的逐页 OCR 成果，每部书各有两个版本：
   - `当涂郡本`（南宋当涂郡斋本，公有领域）与 `儒藏本`（儒藏精华编第110册）。每本以 `<书名>/当涂郡本_{pdf,ocr}/` 与 `<书名>/儒藏本_{pdf,ocr}/` 存放。
   - OCR 页为 `page_NNNN.md`（文本）配 `page_NNNN.pdf`（图）。`目录总览.md` 是总清单（页数、儒藏原页码范围、切分理据）。已收：`大学章句`、`中庸章句`；论语、孟子规划中但暂缺。
2. **软件**（`gujidianjiao/`，包名 `guji_collator`）——一条 AI 校勘流水线。两个可跑示例：精心制作的论语示范（`data/demo/*.txt`）与**已接入真实语料**的大学章句（`data/daxue/`——当涂郡本底本×儒藏本今本，`assemble.py` 负责逐页 OCR 拼接与版面符号清洗）。中庸/论语/孟子语料待同样接入。

### 源码有两份完全相同的副本，但只有一份能跑

`gujidianjiao/*.py`（平铺）与 `gujidianjiao/guji-collation-system/guji-collation-system/guji_collator/*.py` 是**逐字节相同**的（已核对）。只有嵌套那份才是真正的包——它有 `__init__.py`、`__main__.py`、`requirements.txt`、`tests/`、`data/demo/`；平铺那份没有 `__init__.py`，无法以 `python -m guji_collator` 运行。**请把 `gujidianjiao/guji-collation-system/guji-collation-system/` 视为正本。** 若改动源码，要么两份都改、要么合并——任其静默分叉是坑。另有冗余：重复的 `大学章句/大学章句/`、`中庸章句/中庸章句/` 目录树，以及一个因 brace 展开失败而留下的字面目录 `{guji_collator/data,data,outputs,docs}`——均忽略即可。

## 常用命令

以下命令均在正本包根 `gujidianjiao/guji-collation-system/guji-collation-system/` 下执行：

```bash
python -m guji_collator inspect --task data/demo/task.json   # 只做对齐与异文检测，不合议
python -m guji_collator run     --task data/demo/task.json   # 完整流水线：OCR→对校→合议→裁断→双本→精校台
python -m guji_collator apply   --task data/demo/task.json --decisions data/demo/decisions.sample.json  # 回灌人工决策，重出定本
python -m pytest tests/ -q        # 或：python tests/test_pipeline.py
```

LLM 后端（默认 offline；offline 是确定性的，也是回归基线）：
```bash
export ANTHROPIC_API_KEY=sk-...   && python -m guji_collator run --task task.json --llm anthropic
# 或：--llm openai-compat  （需 cfg.base_url + OPENAI_API_KEY）
```

核心流水线**零依赖**（仅标准库，Python 3.10+）。`requirements.txt` 全是注释掉的可选生产依赖（`paddleocr`、`opencc-python-reimplemented`、`pillow`、`pytest`）。

## 架构：七阶段流水线

`pipeline.py` 编排 S0→S7。每阶段都在 `outputs/<work>/{work,internal,public}/` 下落盘中间产物，故任一阶段可单独重跑，全流程可复现（同输入同输出）。各模块及其不可破坏的不变式：

- **`config.py`** — 由 `task.json` 加载的 `Config`；一个 task = 一个可版本化、可复现的校勘任务。`SourceSpec` 承载每个见证（witness）的 `rights`（`public-domain` | `copyrighted` | `unknown`）与 `kind`（`base` 底本善本 | `modern` 现代点校本 | `witness` 他校本）。`base.publishable` 决定是否产出公开本。
- **`normalize.py`** — **CNF（Collation Normal Form，校勘中间层）**，正确性的枢纽。把两侧文本投影到同一正字层：简繁/异体/俗字差异在此**消解**（不算异文），通假/避讳/形近/脱衍倒则**保留**（成为异文，交合议庭）。关键在于规范化是**可逆索引**的——CNF 的每个字都记录其在原文中的下标，故最终裁断必落回底本善本的**原字形**，绝不把底本"洗"成现代字。内含驱动一切下游判断的数据表：`S2T`、`VARIANT_TO_ORTHODOX`、`PHONETIC_LOANS`、`TABOO_RULES`（避讳是版本断代证据，须显式标注、不得静默还原）、`CONFUSABLE_GROUPS`。
- **`ocr.py`** — `PlainTextEngine` 是**确定性离线演示引擎**，不是"假 OCR"：它固定了真实引擎的输出接口，并按字形特征合成可复现的逐字置信度，使整条流水线在无 GPU/无网络下可测。生产替换件 = `PaddleOCREngine`（桩）或任何实现 `OCREngine` 协议的实现——但**必须保留每字 `bbox`**，否则视觉复核无法回原图定位。`VisionRechecker` 把低置信字送 VLM 复核（离线回退用形近先验 + `offline_truth`）。`fuse_confidence` = OCR-A / OCR-B / 视觉三方的对数几率（Naive Bayes）融合。硬规则：漫漶输出 `□`，**绝不猜字**。
- **`align.py`** — 两级对齐：`anchor_blocks`（以最长公共子串为锚切分段块，避免万字长文中远处相同字被错配）+ `nw_align`（带古籍代价函数的 Needleman–Wunsch：通假 0.25、避讳 0.35、形近 0.45、其他 1.0；gap 0.8——故形近误识判为"替换"而非"脱+衍"）。`classify()` 给每处异文贴 9 类 `DivType` 标签，附自动按语与候选读法。`project_punctuation()` 仅把今本标点**提案**性地投到底本上，绝非最终公开句读。
- **`agents.py`** — **校书官合议庭**。四个 persona 把陈垣《校勘学释例》的校法四则各拆成一个角色，每个角色**只许用自己那一类证据**（限权是全部设计的用意所在——以此防止理校泛滥，即凭"文义较顺"改字这一清代校勘学大忌）：劉向（对校，唯版本、不改字）、解縉（他校，类书+古注+引文，必指出处、**绝不杜撰书证**）、戴震（本校+理校，音韵训诂）、紀昀（主席/终审，不引新证、只裁断）。`HeuristicReasoner` 是各 charter 的确定性编码，兼作回归基线；`AnthropicClient`/`OpenAICompatClient` 为生产替换件。`Council.deliberate()` → `chair_review()`。
- **`adjudicate.py`** — 把三家意见聚合成 `Verdict`。双因子加权投票 `S(h)=Σ w_k(t)·g_k·c_{k,h}`（角色权重 × 证据等级系数 × 置信度），归一为 `P(h)`，并以**底本优先偏置 β=0.15** 编码"底本可通则不改"。**三条悬置规则**（满足其一即挂起人工、不得自动改字）：首选得票 `P(1)<τ(0.55)`、首选次选差距 `P(1)−P(2)<δ(0.18)`、或**无硬证据却要改正文**。第三条最要紧。`_compose_note()` 按四库体例写校勘记。
- **`render.py`** — `RuleSegmenter` 是一个刻意保守的文言断句器，**独立于现代点校本**——这是版权隔离的技术前提（底本文字公有领域，现代标点则否）。`punctuation_overlap()`（与今本的 Jaccard）**在 >0.92 时告警**——重合过高意味着公开本实质上抄了今本标点。产出公开善本点校本与仅供自用的内部参校本。
- **`review.py`** — 产出**单文件、零依赖、可离线**的 HTML 人工精校台（键盘流：J/K、1–9、E）。决策存 `localStorage`，导出 `decisions.json`，由 `apply` 回灌以重出定本。

## 测试锁死的不变式（勿破坏）

`tests/test_pipeline.py` 针对精心准备的示范数据断言：`stats.total == 11`；至少一条悬置；每条 `形近疑誤`（OCR 形近）**必须**悬置（绝不自动改字）；句读 Jaccard `< 0.92`；公开本不含简体字且带公有领域声明；若干特定字的分类精确无误（`說`→通假、`□`→脱文、`樂`→衍文、`眚`→形近、`謹`→实质、`文學`→倒文、`人`→避讳）；且任何纯字形差异（汎/逺/衆/爲）都不得作为异文出现。改动 `normalize.py`/`align.py`/`adjudicate.py` 时，这些数字与分类即是契约。

## 三条硬约束（出自 README，统摄一切改动）

1. **底本可通则不改** — 底本文义可通即不改正文。由 β=0.15 与"无硬证据则悬置"规则共同保证。"文义较顺"绝不能作为改字理由。
2. **公开本句读必须独立生成** — 公开本标点须来自 `RuleSegmenter`，不得照搬今本（版权）。Jaccard 告警即此约束的执行。
3. **宁可标记不确定，不可自作主张** — 漫漶→`□`，无共识→`兩存`，每条结论都留三家原始意见，可追溯、可推翻。

## 术语表（代码中高频出现的专名）

底本=善本底本；今本/参校本=现代点校参考本；对校/他校/本校/理校=校法四则；通假=通假字；避讳=避讳改字；衍文=重出衍字；脱文=缺字；倒文=次第互异；校勘记=异文考订记录；悬置=挂起待人工；句读=标点断句；两存=两读并存、不作取舍。
