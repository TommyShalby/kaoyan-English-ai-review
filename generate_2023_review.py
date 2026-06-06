#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2023 考研英语（二）客观题黑笔复盘 v8.2 —— 全题复习 + 词汇表版（Bug修复版）

v8.2 修复（相对 v8.1）：
  - 修复 SECTIONS 元组长度不一致导致的 ValueError（统一改为 5 元素，use_json 默认 False）
  - 修复词汇段 JSON 模式缺失转换逻辑（json_to_md_vocab 函数将 JSON 渲染为 Markdown 表格）
  - 修复 done_count 解包崩溃（同步 SECTIONS 改动）
  - 词汇段改为 use_json=False，用 Markdown prompt 直接输出（更稳定，避免 JSON→MD 转换复杂度）
"""

import os
import sys
import json
import time
import threading
import requests
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ==================== ANSI 颜色 ====================
if os.name == "nt":
    os.system("")
C = "\033[96m"
G = "\033[92m"
Y = "\033[93m"
B = "\033[1m"
D = "\033[2m"
R = "\033[0m"

# ==================== 配置 ====================
API_KEY        = os.getenv("GEMINI_API_KEY")
BUCKET_NAME    = os.getenv("GCS_BUCKET_NAME", "")
PDF_OBJECT_NAME = os.getenv("PDF_OBJECT_NAME", "23English.pdf")

OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR", "outputs"))
CHECKPOINT_DIR  = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
CHECKPOINT_PATH = CHECKPOINT_DIR / "2023英语_全题复盘_checkpoint.json"
MD_PATH         = OUTPUT_DIR / "2023英语_全题复习手册_v8.2.md"

MODEL_NAME  = os.getenv("MODEL_NAME", "gemini-3.1-pro-preview")
GCS_URI     = os.getenv("GCS_URI") or (("gs://" + BUCKET_NAME + "/" + PDF_OBJECT_NAME) if BUCKET_NAME else "")
API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "https://aiplatform.googleapis.com/v1/publishers/google/models")

BUDGET_LIMIT_USD  = 15.0
MAX_OUTPUT_TOKENS = 32768
TEMPERATURE       = 0.7

# ==================== 成本追踪 ====================
class BudgetExceeded(Exception):
    pass

class CostTracker:
    def __init__(self, budget):
        self.budget = budget
        self.total_input_chars  = 0
        self.total_output_chars = 0
        self.call_count = 0

    def record(self, input_len, output_len):
        self.total_input_chars  += input_len
        self.total_output_chars += output_len
        self.call_count += 1

    def estimated_cost(self):
        return (self.total_input_chars  / 1_000_000 * 1.25 +
                self.total_output_chars / 1_000_000 * 5.0)

    def check_budget(self):
        cost = self.estimated_cost()
        if cost > self.budget:
            raise BudgetExceeded("预算超限！已用 $%.2f / $%.2f" % (cost, self.budget))
        return cost

    def status(self):
        return "$%.3f/$%.0f" % (self.estimated_cost(), self.budget)

cost_tracker = CostTracker(BUDGET_LIMIT_USD)

# ==================== Spinner ====================
_spinner_stop = threading.Event()

def _spin(msg):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not _spinner_stop.is_set():
        sys.stdout.write("\r" + C + frames[i % len(frames)] + R + " " + msg)
        sys.stdout.flush()
        time.sleep(0.1)
        i += 1
    sys.stdout.write("\r" + " " * (len(msg) + 4) + "\r")
    sys.stdout.flush()

def start_spinner(msg):
    _spinner_stop.clear()
    t = threading.Thread(target=_spin, args=(msg,), daemon=True)
    t.start()
    return t

def stop_spinner(t):
    _spinner_stop.set()
    t.join()

def progress_bar(current, total, prefix="", width=30):
    pct   = current / total if total > 0 else 0
    filled = int(width * pct)
    bar   = "█" * filled + "░" * (width - filled)
    return "%s%s [%s] %d/%d (%d%%)%s" % (C, prefix, bar, current, total, pct * 100, R)

# ==================== API 调用 ====================
def call_gemini(prompt_text, step="", max_retries=3, timeout=360,
                use_pdf=True, use_json=False):
    if not API_KEY:
        raise RuntimeError("请先设置环境变量 GEMINI_API_KEY，或在 .env 中填写。")
    if use_pdf and not GCS_URI:
        raise RuntimeError("当前任务需要 PDF。请设置 GCS_URI，或设置 GCS_BUCKET_NAME + PDF_OBJECT_NAME。")

    url = API_BASE_URL.rstrip("/") + "/" + MODEL_NAME + ":generateContent?key=" + API_KEY

    if use_pdf:
        parts = [
            {"text": prompt_text},
            {"file_data": {"mime_type": "application/pdf", "file_uri": GCS_URI}},
        ]
    else:
        parts = [{"text": prompt_text}]

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "temperature":     TEMPERATURE,
            "top_p":  0.95,
            "top_k":  64,
        },
    }

    if use_json:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseSchema"] = {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "section_name":       {"type": "string"},
                            "one_word_multi_meaning": {
                                "type": "array",
                                "items": {"type": "object"}
                            },
                            "logic_connectors": {
                                "type": "array",
                                "items": {"type": "object"}
                            },
                        },
                    },
                },
                "final_paragraph": {
                    "type": "string",
                    "description": "放在整个词汇表最末尾的一个完整大段总结，80-120字",
                },
            },
            "required": ["sections", "final_paragraph"],
        }

    input_len = len(prompt_text)

    for attempt in range(max_retries):
        status_msg = "%s (尝试 %d/%d) [%s]" % (step, attempt + 1, max_retries, cost_tracker.status())
        spinner    = start_spinner(status_msg)
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            stop_spinner(spinner)

            if resp.status_code == 200:
                data       = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    print("  " + Y + "⚠ 无候选结果，重试..." + R)
                    time.sleep(8 * (attempt + 1))
                    continue

                content   = candidates[0].get("content", {})
                parts_out = content.get("parts", [])
                if not parts_out or "text" not in parts_out[0]:
                    finish = candidates[0].get("finishReason", "unknown")
                    print("  " + Y + "⚠ 返回无 text（finishReason=" + str(finish) + "），重试..." + R)
                    time.sleep(8 * (attempt + 1))
                    continue

                text = parts_out[0]["text"]
                cost_tracker.record(input_len, len(text))
                cost_tracker.check_budget()

                finish_reason = candidates[0].get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    print("  " + Y + "⚠ 输出被截断（MAX_TOKENS），建议增大 MAX_OUTPUT_TOKENS" + R)

                print("  " + G + "✓ " + step + " 完成" + R +
                      " " + D + "(" + str(len(text)) + "字符)" + R)
                return text

            elif resp.status_code == 429:
                wait = 15 * (attempt + 1)
                print("  " + Y + "⚠ 429 限速，等待 " + str(wait) + "s..." + R)
                time.sleep(wait)
            else:
                print("  " + Y + "⚠ HTTP " + str(resp.status_code) + R)
                print("  " + D + resp.text[:300] + R)
                if attempt == max_retries - 1:
                    raise Exception("API 失败 HTTP " + str(resp.status_code))
                time.sleep(8 * (attempt + 1))

        except BudgetExceeded:
            raise
        except requests.exceptions.Timeout:
            try:
                stop_spinner(spinner)
            except Exception:
                pass
            print("  " + Y + "⚠ 请求超时 (" + str(timeout) + "s)，重试..." + R)
            if attempt == max_retries - 1:
                raise
            time.sleep(10)
        except Exception as exc:
            try:
                stop_spinner(spinner)
            except Exception:
                pass
            if isinstance(exc, BudgetExceeded):
                raise
            print("  " + Y + "⚠ " + str(exc) + R)
            if attempt == max_retries - 1:
                raise
            time.sleep(8 * (attempt + 1))

    raise Exception("API 调用失败（" + str(max_retries) + " 次重试均失败）")

# ==================== 答案序列 ====================
CLOZE_USER = ["C","D","C","D","D","D","C","C","A","C",
              "D","D","D","A","C","B","D","A","C","D"]
CLOZE_ANS  = ["B","C","A","D","D","D","A","B","B","C",
              "C","D","C","A","A","B","D","B","C","A"]

READ_USER  = ["A","B","B","B","A","C","D","C","B","B",
              "B","D","D","A","A","A","C","A","B","D"]
READ_ANS   = ["A","B","B","C","D","D","A","C","B","D",
              "C","D","A","A","B","A","C","D","B","A"]

PARTB_USER = ["A","F","B","D","G"]
PARTB_ANS  = ["D","F","B","A","G"]

CLOZE_WORDS = {
    1: "profit",   2: "prioritize", 3: "exclusively", 7: "despite",
    8: "however",  9: "created",   11: "lack",        13: "promoting",
   15: "unite",   18: "responsible", 20: "serve",
}
PARTB_SPEAKERS = {
    41: "Brian Berry (Federation of Master Builders CEO)",
    42: "Gareth Belsham (Naismiths surveyors)",
    43: "Marcus Jefford (Build Aviator)",
    44: "John Kelly (Freeths law firm, construction lawyer)",
    45: "Andrew Mellor (PRP architects)",
}

READING_TEXTS = {
    "Text 1": (21, 25, "RHS 草坪塑料草"),
    "Text 2": (26, 30, "美国国家公园"),
    "Text 3": (31, 35, "Sparrow 与互联网对人脑记忆的影响"),
    "Text 4": (36, 40, "青少年 prosocial 行为"),
}

# ==================== 数据构建 ====================
def build_cloze_data():
    rows = []
    for i in range(20):
        q       = i + 1
        user    = CLOZE_USER[i]
        correct = CLOZE_ANS[i]
        rows.append({
            "q": q, "user": user, "correct": correct,
            "is_wrong": (user != correct),
            "correct_word": CLOZE_WORDS.get(q, ""),
        })
    return rows

def build_reading_data(text_label):
    start, end, _ = READING_TEXTS[text_label]
    rows = []
    for q in range(start, end + 1):
        idx  = q - 21
        user = READ_USER[idx]
        correct = READ_ANS[idx]
        rows.append({
            "q": q, "user": user, "correct": correct,
            "is_wrong": (user != correct),
        })
    return rows

def build_partb_data():
    rows = []
    for i in range(5):
        q       = 41 + i
        user    = PARTB_USER[i]
        correct = PARTB_ANS[i]
        rows.append({
            "q": q, "user": user, "correct": correct,
            "is_wrong": (user != correct),
            "speaker": PARTB_SPEAKERS[q],
        })
    return rows

# ==================== Prompt 模板 ====================
COMMON_RULES = """**硬性写作规则（必须遵守）：**
- 所有分析必须引用 PDF 原文的英文句子（用反引号 `...` 包起来）。
- 不写"首先/其次/总之""语境理解不到位"这类空话。
- 每题的诊断独立自包含 —— 复习时不必翻原卷。
- 防御口诀/要点 ≤ 20 字。
- 格式：整段只用一个 `##` 二级标题起头，内部用 `###` / `####` 层级。
- **对题与错题的深度差异必须严格区分**（见下方题目级格式要求）。
"""

def prompt_cloze():
    data    = build_cloze_data()
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    n_wrong = sum(1 for r in data if r["is_wrong"])
    return COMMON_RULES + """

# 任务：2023 考研英语（二）完形填空 · 全局 + 全题分析

PDF 里 Section I 是原文（创业公司 growth team 故事）。
学生 20 题作答情况（is_wrong=true 表示错题，共 """ + str(n_wrong) + """ 道错）：
""" + data_json + """

## 一、完形填空复盘

### 1. 文章全局地图
- **主题句**（引用原文）：
- **行文逻辑链**（用 → 串起 5 段核心论点）：
  段1：... → 段2：... → 段3：... → 段4：... → 段5：...
- **考点分布统计**：逻辑连词 X 题 / 近义词辨析 X 题 / 词组搭配 X 题 / 语境复现 X 题

### 2. 学生思维漏洞诊断（基于错题提炼）
写 3–4 条漏洞，每条必须：
- 漏洞名 ≤ 10 字（例："忽略转折信号"）
- 涉及至少 2 道题号佐证
- 一句话本质 + 一句话矫正法

### 3. 逐题分析（**必须写完 20 题**）

**错题格式**（标题旁加 ❌ 号）：
Q[号] ❌ 你选 [错项] → 正答 [正项] ([correct_word])

原题空格前后句（引用 PDF 原文 10-20 词）
错项陷阱：为什么 [错项] 看起来合理但错？（用原文证据驳倒）
正项逻辑：为什么 [正项] 才是对的？（引用原文信号词）
防御口诀：... （≤ 20 字）

**对题格式**（标题旁加 ✅ 号）：
Q[号] ✅ 选 [正项] ([单词])
要点：... （一行 ≤ 30 字，说明这题为什么应该选这个 / 关键信号词）

必须严格按此格式写，**不要遗漏任何一题**（Q1 到 Q20 每题都要出现）。

### 4. 完形 SOP（3 步标准流程）
基于本次教训：以后做完形按哪 3 步？每步 ≤ 15 字。
"""

def prompt_reading(text_label):
    data    = build_reading_data(text_label)
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    start, end, theme = READING_TEXTS[text_label]
    n_wrong = sum(1 for r in data if r["is_wrong"])
    section_title = text_label + " (" + theme + ")"
    return COMMON_RULES + """

# 任务：2023 考研英语（二）阅读 """ + section_title + """ · 全局 + 全题分析

PDF 里这篇阅读（Q""" + str(start) + """–Q""" + str(end) + """）是原文。
学生 5 题作答情况（共 """ + str(n_wrong) + """ 道错）：
""" + data_json + """

## 阅读 """ + section_title + """

### 1. 文章全局地图
- **主题句**（引用原文 thesis sentence，英文）：
- **段落功能图**（每段 1 句话）：
  段1：... → 段2：... → 段3：... → 段4：... → 段5：...
- **作者立场**：支持/反对/中立 + 理由
- **本文高频陷阱信号词**：however / but / unlike / despite / such as 等在本文具体出现位置

### 2. 本篇命题套路
错题用了什么陷阱？从以下 7 类勾选 + 一句话说明：
- [ ] 偷换范围（part vs whole）
- [ ] 绝对化表述（always/never/all）
- [ ] 因果倒置
- [ ] 张冠李戴
- [ ] 混淆作者 vs 他人观点
- [ ] 超纲推断
- [ ] 字面陷阱（字眼相同但意思偏移）

### 3. 逐题分析（**必须写完 Q""" + str(start) + """–Q""" + str(end) + """ 全部 5 题**）

**错题格式**（❌）：
Q[号] ❌ 你选 [错项] → 正答 [正项]

题干翻译（中文）
四选项翻译（A/B/C/D 各一行简洁译文）
定位：正答对应原文哪句？（引用英文原句 10-30 词）
陷阱剖析：错项为什么诱人？（从原文找到看似支持它的那句话）+ 陷阱类型
矫正口诀：... （≤ 20 字）

**对题格式**（✅）：
Q[号] ✅ 选 [正项]
要点：... + 原文关键句 ... （一行 ≤ 40 字）

### 4. 本篇共性教训
1–2 句话，只针对这篇：错题暴露了什么共同问题？
"""

def prompt_partb():
    data    = build_partb_data()
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    n_wrong = sum(1 for r in data if r["is_wrong"])
    return COMMON_RULES + """

# 任务：2023 考研英语（二）Part B（房屋能效新规）· 全局 + 全题分析

PDF 里 Part B 是原文（5 位发言人就新建房/扩建节能新规表态）。
学生 5 题作答情况（共 """ + str(n_wrong) + """ 道错）：
""" + data_json + """

## 三、Part B 复盘

### 1. 5 位发言人立场速览表

| 题号 | 发言人 | 身份 | 核心观点（≤20字）| 原文 key phrase |
|------|--------|------|------------------|-----------------|
| 41 | Brian Berry | Federation of Master Builders CEO | ... | `...` |
| 42 | Gareth Belsham | Naismiths surveyors | ... | `...` |
| 43 | Marcus Jefford | Build Aviator | ... | `...` |
| 44 | John Kelly | Freeths law firm (construction) | ... | `...` |
| 45 | Andrew Mellor | PRP architects | ... | `...` |

### 2. 7 个选项（A–G）题眼关键词

| 选项 | 原文 | 题眼（≤15字）|
|------|------|--------------|
| A | The rise of home prices is a temporary matter. | ... |
| B | Builders possibly need to submit new estimates of their projects. | ... |
| C | There will be specific limits on home extensions to prevent heat loss. | ... |
| D | The new rules will take plans to an even higher level. | ... |
| E | Many people feel that home prices are already beyond what they can afford. | ... |
| F | The new rules will affect people whose home extensions include new windows or doors. | ... |
| G | The rule changes will benefit homeowners eventually. | ... |

### 3. 逐题分析（**必须写完 Q41–Q45 全部 5 题**）

**错题格式**（❌）：
Q[号] ❌ 你选 [错项] → 正答 [正项]（发言人：...）

发言人原话（引用原文关键句 20–40 词）
错项 [X] 为什么骗到你：错项 [X] 对应哪位发言人？那位说了什么？你为何混淆？
正项 [Y] 的不可替代性：Y 精准命中哪个核心论点？原文有什么独特信号只指向 Y？
防御口诀：... （≤ 20 字）

**对题格式**（✅）：
Q[号] ✅ 选 [正项]（发言人：...）
要点：... + 原文关键句 ... （一行 ≤ 40 字）

### 4. Part B 通用 SOP（4 步）
基于本次教训总结做 Part B 的 4 步流程。
"""

def prompt_summary(results):
    chunks = []
    for key, text in results.items():
        if text and not (key.startswith("99") or key.startswith("08")):
            chunks.append("=== " + key + " ===\n" + text[:2500])
    sections_text = "\n\n".join(chunks)
    return """你是考研英语顶级阅卷老师。下面是一位学生 2023 英语（二）客观题 45 题的详细诊断（23 错 + 22 对）。

请基于这些诊断，写一份"跨题型漏洞总结 + 考场 SOP"。严格遵守：
- 不重复前面已有的内容，只做"拔高总结"。
- 每条漏洞至少涉及 3 道具体题号（跨题型最好）。
- 格式严格按下方骨架。

## 四、跨题型思维漏洞总览

### 漏洞 1：[名字 ≤ 10 字]
- **涉及题号**（至少 3 个跨题型例子）：...
- **本质**：一句话说清楚
- **矫正训练**：考前一周每天做什么能专治它

### 漏洞 2：...

### 漏洞 3：...

（3–5 条）

---

## 五、考场 SOP（决胜流程图）

### 完形填空 SOP
- 第 1 步：...
- 第 2 步：...
- 第 3 步：...

### 阅读 Part A SOP
- 第 1 步：...
- 第 2 步：...
- 第 3 步：...

### 阅读 Part B SOP
- 第 1 步：...
- 第 2 步：...
- 第 3 步：...

---

## 六、最后一句话

用一句 ≤ 30 字的话概括本次复盘的**最关键收获**（考前贴桌上那句）。

===== 以下是前面各段的诊断摘要（仅供参考）=====

""" + sections_text

def prompt_vocab():
    """词汇表：直接输出 Markdown（不用 JSON 模式，更稳定）"""
    return """你是考研英语词汇专家，正在为一位学生整理 2023 考研英语（二）试卷的高价值词汇表。
请直接阅读 PDF 原文，提取以下两类词。

**学生的自述痛点**：
> 我做错题的 90% 原因是单词意思不够了解。要么是常见词这里用的是生偏义我没记住，要么是逻辑连接词没吃透。

所以你**只收录两类词**：一词多义（生偏义）+ 逻辑连接词/信号词。

## 七、高价值词汇表

> 基于学生自述痛点定制：只收**一词多义（生偏义）**和**逻辑连接词**两类。
> 每类独立成表，每个题型独立成节。按本卷里出现的**顺序**排列。

### 7.1 完形填空（Section I）

#### 一词多义
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

#### 逻辑连接词
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

### 7.2 阅读 Text 1 (RHS 草坪)

#### 一词多义
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

#### 逻辑连接词
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

### 7.3 阅读 Text 2 (国家公园)

#### 一词多义
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

#### 逻辑连接词
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

### 7.4 阅读 Text 3 (Sparrow 互联网记忆)

#### 一词多义
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

#### 逻辑连接词
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

### 7.5 阅读 Text 4 (青少年 prosocial)

#### 一词多义
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

#### 逻辑连接词
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

### 7.6 Part B (房屋能效新规)

#### 一词多义
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

#### 逻辑连接词
| 词 | 词性 | 本卷含义 | 常见义 |
|----|------|----------|--------|

---

**总结大段**（放在所有表格之后，80–120 字）：
请用 80–120 字写一段总结性文字：
- 总结本卷最有价值的 5–6 个高频生偏义/逻辑连接词
- 说明这些词对做题的实际帮助
- 给学生一句鼓励性的话（≤ 15 字）

**每个小节严格 15–25 词，表格必须用 Markdown 标准格式，最后必须有总结大段。**
"""

# ==================== 开头总览（纯数据，不调 API） ====================
def render_overview():
    cloze_data = build_cloze_data()
    partb_data = build_partb_data()

    n_cloze_wrong = sum(1 for r in cloze_data if r["is_wrong"])
    n_read_wrong  = sum(
        1 for t in READING_TEXTS
        for r in build_reading_data(t) if r["is_wrong"]
    )
    n_partb_wrong = sum(1 for r in partb_data if r["is_wrong"])
    n_total_wrong = n_cloze_wrong + n_read_wrong + n_partb_wrong

    lose_cloze  = n_cloze_wrong  * 0.5
    lose_read   = n_read_wrong   * 2
    lose_partb  = n_partb_wrong  * 2
    total_lose  = lose_cloze + lose_read + lose_partb
    score       = 60.0 - total_lose

    rows = []
    cloze_wrong_qs = ", ".join("Q" + str(r["q"]) for r in cloze_data if r["is_wrong"])
    rows.append("| 完形 | " + str(n_cloze_wrong) + " / 20 | " + cloze_wrong_qs + " |")
    for text_label, (start, end, theme) in READING_TEXTS.items():
        rdata    = build_reading_data(text_label)
        wrong_qs = ", ".join("Q" + str(r["q"]) for r in rdata if r["is_wrong"])
        nw       = sum(1 for r in rdata if r["is_wrong"])
        rows.append("| 阅读 " + text_label + " (" + theme + ") | " + str(nw) + " / 5 | " + (wrong_qs or "—") + " |")
    partb_wrong_qs = ", ".join("Q" + str(r["q"]) for r in partb_data if r["is_wrong"])
    rows.append("| Part B | " + str(n_partb_wrong) + " / 5 | " + partb_wrong_qs + " |")
    dist_table = "\n".join(rows)

    your_cloze = " ".join(CLOZE_USER)
    ans_cloze  = " ".join(CLOZE_ANS)
    your_read  = " ".join(READ_USER)
    ans_read   = " ".join(READ_ANS)
    your_partb = " ".join(PARTB_USER)
    ans_partb  = " ".join(PARTB_ANS)
    gen_time   = datetime.now().strftime("%Y-%m-%d %H:%M")

    return (
        "# 2023 考研英语（二）客观题 · 全题复习手册 v8.2\n\n"
        "> **生成时间**：" + gen_time + "\n"
        "> **模型**：" + MODEL_NAME + " · 温度 " + str(TEMPERATURE) + "\n"
        "> **源 PDF**：" + GCS_URI + "\n\n"
        "---\n\n"
        "## 成绩快照\n\n"
        "- **客观题得分**：**%.1f / 60** 分\n" % score +
        "  - 完形扣分：%.1f 分（错 %d 题 × 0.5）\n" % (lose_cloze, n_cloze_wrong) +
        "  - 阅读 Part A 扣分：%.1f 分（错 %d 题 × 2）\n" % (lose_read, n_read_wrong) +
        "  - Part B 扣分：%.1f 分（错 %d 题 × 2）\n" % (lose_partb, n_partb_wrong) +
        "- **错题总数**：%d 题 / 45 客观题\n\n" % n_total_wrong +
        "## 错题分布\n\n"
        "| 题型 | 错/总 | 错题号 |\n"
        "|------|-------|--------|\n" +
        dist_table + "\n\n"
        "## 答案对照\n\n"
        "**完形填空（Q1–Q20）**\n\n"
        "```\n"
        "     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20\n"
        "你选：" + your_cloze + "\n"
        "标答：" + ans_cloze  + "\n"
        "```\n\n"
        "**阅读 Part A（Q21–Q40）**\n\n"
        "```\n"
        "     21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40\n"
        "你选：" + your_read + "\n"
        "标答：" + ans_read  + "\n"
        "```\n\n"
        "**Part B（Q41–Q45）**\n\n"
        "```\n"
        "     41 42 43 44 45\n"
        "你选：" + your_partb + "\n"
        "标答：" + ans_partb  + "\n"
        "```\n\n"
        "---\n\n"
        "## 使用指南\n\n"
        "- 本手册**全题覆盖**：错题给深度 4 行诊断，对题给一行要点。\n"
        "- 每个大段结构：**文章全局 → 命题套路 → 逐题分析 → 共性教训**。\n"
        "- **末尾第七章是高价值词汇表**，专治常见词生偏义 + 逻辑连接词没吃透。\n"
        "- 跨题型漏洞总结和考场 SOP 是本手册最高 ROI 的部分。\n\n"
        "---\n"
    )

# ==================== 各段任务表 ====================
# ★ 修复：统一 5 元素元组 (key, title, prompt_fn, use_pdf, use_json)
#         词汇段改为 use_json=False，直接输出 Markdown，无需二次转换
def section_prompt_t1(): return prompt_reading("Text 1")
def section_prompt_t2(): return prompt_reading("Text 2")
def section_prompt_t3(): return prompt_reading("Text 3")
def section_prompt_t4(): return prompt_reading("Text 4")

SECTIONS = [
    ("01_完形",   "完形填空（全 20 题）",    prompt_cloze,     True,  False),
    ("02_阅读T1", "阅读 Text 1（全 5 题）", section_prompt_t1, True,  False),
    ("03_阅读T2", "阅读 Text 2（全 5 题）", section_prompt_t2, True,  False),
    ("04_阅读T3", "阅读 Text 3（全 5 题）", section_prompt_t3, True,  False),
    ("05_阅读T4", "阅读 Text 4（全 5 题）", section_prompt_t4, True,  False),
    ("06_PartB",  "Part B（全 5 题）",       prompt_partb,     True,  False),
    ("08_词汇",   "高价值词汇表（全卷）",    prompt_vocab,     True,  False),
]

SECTION_TITLES = {
    "01_完形":   "一、完形填空",
    "02_阅读T1": "二、阅读 Text 1 (RHS 草坪)",
    "03_阅读T2": "二、阅读 Text 2 (国家公园)",
    "04_阅读T3": "二、阅读 Text 3 (Sparrow)",
    "05_阅读T4": "二、阅读 Text 4 (青少年 prosocial)",
    "06_PartB":  "三、Part B",
    "08_词汇":   "七、高价值词汇表",
}

# ==================== 组装最终 MD ====================
def assemble_final_md(ckpt):
    parts = [render_overview()]

    main_order = ["01_完形","02_阅读T1","03_阅读T2","04_阅读T3","05_阅读T4","06_PartB"]
    for key in main_order:
        section_md = ckpt.get("results", {}).get(key, "").strip()
        if section_md:
            parts.append(section_md)
        else:
            parts.append("## " + SECTION_TITLES[key] + "\n\n> ⚠ 此段未生成，重跑脚本可继续\n")
        parts.append("---")

    summary_md = ckpt.get("results", {}).get("99_总结", "").strip()
    parts.append(summary_md if summary_md
                 else "## 四、跨题型思维漏洞总览\n\n> ⚠ 总结段未生成（需所有前置段完成后才会生成）\n")
    parts.append("---")

    vocab_md = ckpt.get("results", {}).get("08_词汇", "").strip()
    parts.append(vocab_md if vocab_md
                 else "## 七、高价值词汇表\n\n> ⚠ 词汇表未生成，重跑脚本可继续\n")

    footer = ("---\n\n> *本手册由 Gemini " + MODEL_NAME +
              " 生成 · temperature=" + str(TEMPERATURE) +
              " · 总成本 " + cost_tracker.status() +
              " · 共 " + str(cost_tracker.call_count) + " 次 API 调用*")
    parts.append(footer)
    return "\n\n".join(parts)

# ==================== 主程序 ====================
def main():
    n_cloze_w = sum(1 for r in build_cloze_data() if r["is_wrong"])
    n_read_w  = sum(
        1 for t in READING_TEXTS
        for r in build_reading_data(t) if r["is_wrong"]
    )
    n_partb_w = sum(1 for r in build_partb_data() if r["is_wrong"])
    n_total_w = n_cloze_w + n_read_w + n_partb_w

    print("\n" + C + "=" * 80 + R)
    print(C + "  ██  2023 考研英语（二）全题复盘 v8.2 · 45 题全覆盖 + 词汇表（Bug修复版）" + R)
    print(C + "  ██  文件：" + GCS_URI + R)
    print(C + "  ██  模型：" + MODEL_NAME + "  |  temperature: " + str(TEMPERATURE) + R)
    print(C + "  ██  分段：" + str(len(SECTIONS)) + " + 1 总结段 = " + str(len(SECTIONS) + 1) + R)
    print(C + "  ██  错题：完形 " + str(n_cloze_w) + " + 阅读 " + str(n_read_w) +
          " + PartB " + str(n_partb_w) + " = " + str(n_total_w) + " 题" + R)
    print(C + "  ██  预算：$%.0f USD" % BUDGET_LIMIT_USD + R)
    print(C + "=" * 80 + R + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = (json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
            if CHECKPOINT_PATH.exists() else {"results": {}})
    ckpt.setdefault("results", {})

    def save_ckpt():
        ckpt["_meta"] = {"last_saved": datetime.now().isoformat()}
        CHECKPOINT_PATH.write_text(
            json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ★ 修复：解包现在统一是 5 元素
    done_count = sum(
        1 for key, _, _, _, _ in SECTIONS
        if ckpt["results"].get(key, "").strip()
    )
    if done_count > 0:
        print(G + "✓ 检查点命中：%d/%d 段已完成" % (done_count, len(SECTIONS)) + R + "\n")

    total_steps = len(SECTIONS) + 1

    for i, (key, title, prompt_fn, use_pdf, use_json) in enumerate(SECTIONS):
        current = i + 1
        print(progress_bar(current - 1, total_steps, "  总进度"))

        if ckpt["results"].get(key, "").strip():
            print("  " + D + "[%d/%d] %s 已缓存，跳过" % (current, len(SECTIONS), title) + R + "\n")
            continue

        print("  " + C + "┌─ [%d/%d] %s" % (current, len(SECTIONS), title) +
              R + " " + D + "[" + cost_tracker.status() + "]" + R)
        try:
            md = call_gemini(prompt_fn(), step=title, timeout=420,
                             use_pdf=use_pdf, use_json=use_json)
            ckpt["results"][key] = md.strip()
            save_ckpt()
            print("  " + C + "└─ " + G + "✓ " + title + " 完成并已保存" + R + "\n")
        except BudgetExceeded as exc:
            print("\n  " + Y + "⚠ " + str(exc) + R)
            print("  " + Y + "  进度已保存，可重跑继续" + R)
            break
        except Exception as exc:
            print("  " + C + "└─ " + Y + "⚠ " + title + " 失败：" + str(exc) + "（其它段继续）" + R + "\n")
            continue

        if current < len(SECTIONS):
            time.sleep(2)

    main_keys = ["01_完形","02_阅读T1","03_阅读T2","04_阅读T3","05_阅读T4","06_PartB"]
    main_done   = all(ckpt["results"].get(k, "").strip() for k in main_keys)
    need_summary = main_done and not ckpt["results"].get("99_总结", "").strip()

    if need_summary:
        print(progress_bar(len(SECTIONS), total_steps, "  总进度"))
        print("  " + C + "┌─ [%d/%d] 跨题型总结" % (len(SECTIONS) + 1, total_steps) +
              R + " " + D + "[" + cost_tracker.status() + "]" + R)
        try:
            summary_md = call_gemini(
                prompt_summary(ckpt["results"]),
                step="跨题型总结", timeout=240, use_pdf=False,
            )
            ckpt["results"]["99_总结"] = summary_md.strip()
            save_ckpt()
            print("  " + C + "└─ " + G + "✓ 跨题型总结完成" + R + "\n")
        except Exception as exc:
            print("  " + C + "└─ " + Y + "⚠ 总结段失败：" + str(exc) + R + "\n")
    elif main_done:
        print("  " + D + "[%d/%d] 跨题型总结已缓存，跳过" % (len(SECTIONS) + 1, total_steps) + R + "\n")
    else:
        print("  " + Y + "⚠ 还有前置段未完成，跳过总结段（下次重跑时自动生成）" + R + "\n")

    print(progress_bar(total_steps, total_steps, "  总进度"))
    print()

    print(C + "▶ 组装最终 MD..." + R)
    final_md = assemble_final_md(ckpt)
    MD_PATH.write_text(final_md, encoding="utf-8")

    completed   = sum(1 for key, _, _, _, _ in SECTIONS if ckpt["results"].get(key, "").strip())
    has_summary = "1" if ckpt["results"].get("99_总结", "").strip() else "0"

    print("\n" + C + "=" * 80 + R)
    print("  " + G + "✓ 复习手册已保存 → " + str(MD_PATH) + R)
    print("  " + G + "✓ 已完成 %d/%d 主段 + %s/1 总结段" % (completed, len(SECTIONS), has_summary) + R)
    print("  " + C + "  API 调用 %d 次 | 预估成本 %s" % (cost_tracker.call_count, cost_tracker.status()) + R)
    if completed < len(SECTIONS):
        print("  " + Y + "  未完成部分可直接重跑脚本继续（检查点已保存）" + R)
    print("  " + C + "  用 Typora 打开 .md 即可高效复习（A4 友好）" + R)
    print(C + "=" * 80 + R + "\n")

if __name__ == "__main__":
    try:
        main()
    except BudgetExceeded as e:
        print("\n" + Y + "⚠ 预算熔断：" + str(e) + R)
        print(Y + "  进度已保存到检查点，可重跑继续" + R)
    except KeyboardInterrupt:
        print("\n" + Y + "⚠ 用户中断，进度已保存到检查点" + R)
    except Exception as e:
        print("\n" + Y + "⚠ 未预期错误：" + str(e) + R)
        traceback.print_exc()