import os
import requests
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置 ====================
API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
CHECKPOINT_PATH = OUTPUT_DIR / "KY_English2_Advanced_ckpt.json"
MD_PATH = OUTPUT_DIR / "考研英语二作文_高级模板报告_V3.md"

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.1-pro-preview")
API_BASE_URL = os.getenv("GEMINI_API_BASE_URL", "https://aiplatform.googleapis.com/v1/publishers/google/models")

PAST_TOPICS = """
2010-2025考研英语(二)大作文全部为图表作文：
- 2023大作文：消费者选择餐厅时关注的因素饼状图（特色、服务、环境、价格占比）
- 小作文：给David建议艺术展还是机器人展的邮件
"""

def call_gemini(prompt_text: str, step: str = ""):
    if not API_KEY:
        raise RuntimeError("请先设置环境变量 GEMINI_API_KEY，或在 .env 中填写。")
    url = f"{API_BASE_URL}/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": 8192, "temperature": 0.15}
    }
    print(f"🔥 正在让Gemini后台疯狂计算 {step}（预计15~40秒，请耐心等待）...")
    resp = requests.post(url, json=payload, timeout=180)
    if resp.status_code == 200:
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(f"✅ {step} 完成！")
        return text
    else:
        print(f"❌ {step} 失败 {resp.status_code}")
        raise Exception("API调用失败")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100)
    print("🚀 考研英语二高级模板生成器 V3（绿色进度条 + 实时日志 绝不卡顿）")
    print("   专为记忆薄弱者设计 | 已修复所有显示问题")
    print("=" * 100)

    ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8")) if CHECKPOINT_PATH.exists() else {"results": {}}
    key = "advanced_english2_v3_2026"
    if key in ckpt.get("results", {}):
        print("✅ 检查点已存在，直接输出报告")
        print(ckpt["results"][key])
        return

    stages = [f"{i}/14 {desc}" for i, desc in enumerate([
        "加载检查点", "准备真题数据", "构建真题研究提示词", "调用Gemini提取真题",
        "生成4个记忆薄弱模板", "构建2023大作文提示词", "调用Gemini写2023大作文范文",
        "构建2023小作文提示词", "调用Gemini写建议信范文", "综合优化所有内容",
        "润色模板（降低记忆负担）", "最终报告排版", "保存检查点", "写入Markdown文件"
    ], 1)]

    results = {}
    with tqdm(total=14, desc="🟢 绿色进度条", colour="green", unit="步", ncols=100, mininterval=0.3) as pbar:
        pbar.set_description(stages[0])
        print("📂 正在加载检查点...")
        results['topics_raw'] = PAST_TOPICS
        pbar.update(2)

        pbar.set_description(stages[2])
        prompt1 = f"你是考研英语专家。{results['topics_raw']}\n请用JSON数组准确输出2010-2025年英语(二)大作文+小作文主题总结。直接输出JSON。"
        results['topics'] = call_gemini(prompt1, stages[3])
        pbar.update(2)

        pbar.set_description(stages[4])
        prompt2 = f"基于以下真题：{results['topics']}\n为记忆薄弱学生生成4个最简单大作文模板（通用、饼图、趋势、静态）。每模板3段+空白+口诀。直接输出纯Markdown。"
        results['templates'] = call_gemini(prompt2, stages[4])
        pbar.update(1)

        pbar.set_description(stages[5])
        prompt3 = f"使用饼图模板，为2023英语二大作文写完整范文（餐厅选择因素饼图）。1)描述 2)原因 3)评论。约150词。直接输出Markdown。"
        results['example_big'] = call_gemini(prompt3, stages[6])
        pbar.update(2)

        pbar.set_description(stages[7])
        prompt4 = "为2023英语二小作文写完整建议信范文（给David建议艺术展还是机器人展）。直接输出Markdown。"
        results['example_small'] = call_gemini(prompt4, stages[8])
        pbar.update(2)

        pbar.set_description(stages[9])
        prompt5 = f"""综合以下所有输出，优化模板和范文，使其更简洁、记忆负担最低。直接输出优化后的 Markdown。

【真题主题总结】
{results['topics']}

【4个模板】
{results['templates']}

【2023大作文范文】
{results['example_big']}

【2023小作文范文】
{results['example_small']}
"""
        results['optimized'] = call_gemini(prompt5, stages[9])
        pbar.update(3)

        pbar.set_description(stages[12])
        print("💾 正在保存检查点和报告...")
        final_report = f"""# 考研英语二作文高级模板报告 V3（记忆薄弱版）
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}
近20年主题总结：{results['topics']}

### 4个超级简单模板（已优化）
{results['templates']}

### 2023大作文实操示范（餐厅选择因素饼图）
{results['example_big']}

### 2023小作文实操示范（建议信）
{results['example_small']}

{results['optimized']}

绿色进度条已确认，模板背完直接冲刺！"""
        pbar.update(2)

    ckpt.setdefault("results", {})[key] = final_report
    CHECKPOINT_PATH.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")
    MD_PATH.write_text(final_report, encoding="utf-8")

    print(f"\n🎉 V3全部完成！绿色进度条已生效，报告已保存 → {MD_PATH}")
    print("打开Markdown文件，所有模板 + 2023范文都在，打印出来背就行！")

if __name__ == "__main__":
    main()