# AI 古籍校勘系统 · guji-collator

以公有领域善本为底本，用 AI 做初校、用"校书官合议庭"做理校顺通、用人工做终审，
最终产出**两个本子**：可自由传播的**善本点校本**，和仅供自用的**现代参校本**。

```
善本书影 ──OCR──┐
                ├─ 规范化 ─ 对校 ─ 异文检测 ─ 校书官合议 ─ 裁断/悬置 ─┬─ 公开：善本点校本 + 校勘记
现代点校本扫描 ──┘                                                  └─ 内部：现代参校本
                                                       悬置项 → 人工精校台 → 回灌 → 定本
```

## 快速开始

无需网络、无需 GPU、无需 API Key，开箱即跑：

```bash
cd guji-collation-system

# 1. 只看异文检测结果
python -m guji_collator inspect --task data/demo/task.json

# 2. 跑完整流程（OCR → 对校 → 合议 → 出双本 → 生成人工精校台）
python -m guji_collator run --task data/demo/task.json

# 3. 打开人工精校台，处理悬置条目，点「导出 decisions.json」
open outputs/lunyu-xueer/review.html

# 4. 回灌人工决策，出定本
python -m guji_collator apply --task data/demo/task.json \
       --decisions data/demo/decisions.sample.json
```

产物：

| 路径 | 内容 | 可否公开 |
|---|---|---|
| `outputs/<work>/public/critical_edition.html` | 善本点校本（AI 初校） | ✅ |
| `outputs/<work>/public/critical_edition.final.html` | 善本点校**定本**（人工精校后） | ✅ |
| `outputs/<work>/internal/modern_reference.md` | 现代参校本 | ❌ 仅自用 |
| `outputs/<work>/review.html` | 人工精校台（单文件、可离线） | 内部 |
| `outputs/<work>/pending_review.json` | 悬置待审清单 | 内部 |
| `outputs/<work>/run_report.json` | 全量审计记录（含每条异文的三家意见） | 内部 |

## 接入真实模型

演示态用的是确定性的离线组件（`PlainTextEngine` + `HeuristicReasoner`），
接口与生产件完全一致，替换即可：

```bash
export ANTHROPIC_API_KEY=sk-...
python -m guji_collator run --task task.json --llm anthropic
```

| 演示件 | 生产替换 | 位置 |
|---|---|---|
| `PlainTextEngine` | PaddleOCR / Kraken / 商用古籍 OCR | `ocr.py` |
| `VisionRechecker(client=None)` | 多模态模型 + 原图裁切 | `ocr.py` |
| `HeuristicReasoner` | `AnthropicClient` / `OpenAICompatClient` | `agents.py` |
| `_CITATIONS`（示例 JSON） | 类书／古注引文全文索引库 | `agents.py` |

## 模块

```
guji_collator/
├── normalize.py    字形规范化层（简繁、异体、通假、避讳、形近表）→ CNF 中间层
├── ocr.py          双通道 OCR、字符级置信度、低置信回原图视觉复核、对数几率融合
├── align.py        锚点对齐 + 古籍代价 Needleman–Wunsch + 异文分类（9 类）
├── agents.py       四位校书官（刘向/解缙/戴震/纪昀）的职权、铁律与提示词
├── adjudicate.py   分型加权投票、证据分级、底本优先偏置、三条悬置规则
├── render.py       独立断句器、善本点校本、校勘记、内部参校本、版权隔离
├── review.py       单文件人工精校台（键盘流、localStorage、导出 JSON）
├── pipeline.py     七阶段编排、审计留痕、决策回灌
└── cli.py          run / apply / inspect
```

## 设计上的三个硬约束

1. **底本可通则不改。** 裁决目标函数里显式带底本偏置 \(\beta=0.15\)；
   没有确证或旁证而要动底本正文的，一律悬置，不允许"文义较顺"改字。
2. **公开本的句读必须独立生成。** 底本文字是公有领域，现代点校本的标点不是。
   系统用独立断句器出稿，再统计与今本的句读重合率，超阈值即告警。
3. **宁可标记不确定，不可自作主张。** OCR 遇漫漶输出 `□` 而非猜字；
   合议无共识就"两存"；每条结论都留三家原始意见，可追溯、可推翻。

## 测试

```bash
python -m pytest tests/ -q     # 或 python tests/test_pipeline.py
```

## 授权与合规

- 底本须为公有领域（`rights: public-domain`），否则系统自动关闭公开本产出。
- 标记为 `copyrighted` 的现代点校本，其派生物只落 `internal/`，并强制加不可传播声明。
- 校勘记只记录"某本作某字"这一事实，不复制他人按语措辞。
- 实际发布前请自行确认所用书影的具体授权条款（部分馆藏对**扫描件**另有使用限制）。
