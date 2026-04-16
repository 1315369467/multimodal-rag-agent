"""
基于完整 OCR Markdown 文件生成评估数据集
──────────────────────────────────────────────────────────────────────────────
读取 data/ocr_output 下的 Markdown 文件，将整篇文档（或截断后的版本）
一次性交给 LLM，生成指定数量的高质量 Q&A 对，写入 JSONL 文件。

用法
────
python scripts/generate_eval_dataset.py
python scripts/generate_eval_dataset.py --questions 20 --output data/eval_ocr_whole.jsonl
python scripts/generate_eval_dataset.py --filter "Qwen3" --questions 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from openai import OpenAI

from config.settings import get_settings

settings = get_settings()

DEFAULT_OCR_DIR = Path(__file__).resolve().parents[1] / "data" / "ocr_output"
DEFAULT_OUTPUT  = Path(__file__).resolve().parents[1] / "data" / "eval_dataset_ocr.jsonl"



_SYSTEM_PROMPT = """\
你是一个专业的 RAG 系统评估数据集构建专家。给定一篇完整的技术文档，生成若干高质量问答对，\
专门用于评估检索增强生成（RAG）系统的检索与理解能力。

## 核心目标
生成的问答对必须能够有效区分"好的检索"和"差的检索"——\
即答案只存在于文档某一具体段落中，非针对该段落的检索无法正确回答。

## 问题类别定义（按比例均衡分配）

1. **factual**（精确事实）
   - 涉及具体数值、参数、指标、配置，答案精确唯一
   - 示例：层数、隐层维度、训练数据量、benchmark 分数、学习率

2. **procedural**（流程步骤）
   - 涉及算法流程、训练阶段、推理步骤的具体描述
   - 答案需体现步骤的顺序或关键操作

3. **table_lookup**（表格检索）
   - 答案直接来源于文档中的某张表格内的数据
   - 此类问题专门测试系统对结构化表格内容的检索能力

4. **definition**（概念定义）
   - 针对文档中某个特定术语、方法或模块的定义或功能说明
   - 答案需引用文档对该概念的原始表述，而非通用解释

5. **limitation**（局限与挑战）
   - 涉及方法的局限性、失败案例、未来工作方向
   - 此类信息通常分散在文档末尾或讨论章节，检索难度较高

## 问题质量要求

- **唯一定位**：问题必须能精确指向文档中某一具体段落，其他段落无法正确回答
- **不可猜测**：问题不能被背景知识直接回答，必须依赖文档内容
- **无歧义**：问题表述清晰，读者明确知道在问什么
- **禁止是/否题**：所有问题必须要求实质性答案，禁止"是否……"类问题
- **禁止泛化题**：禁止"这篇文章讲了什么"等无法精确检索的宽泛问题

## 答案质量要求

- 答案直接引自原文，保留关键数值、专有名词、原始表述
- 答案长度适中：factual/table_lookup 可为短语或数值；procedural/definition 可为 1-3 句

## evidence 字段
`evidence` 填写原文中**包含答案的最小完整片段**（1-2 句或表格的关键行），\
用于在评估时验证检索是否命中了正确的文档块。

## 输出格式（严格 JSON 数组，不输出任何其他内容）
[
  {
    "question": "问题（中文）",
    "answer": "答案（保留原文语言和关键术语）",
    "category": "factual",
    "evidence": "原文中包含答案的关键句或数据片段"
  },
  ...
]
"""


_CATEGORIES = ["factual", "procedural", "table_lookup", "definition", "limitation"]


def build_user_prompt(doc_name: str, content: str, n: int) -> str:
    # 按 5 个类别均衡分配，余数从前往后补
    num_cats = len(_CATEGORIES)
    per_cat = max(1, n // num_cats)
    remainder = n - per_cat * num_cats
    counts = [per_cat + (1 if i < remainder else 0) for i in range(num_cats)]
    distribution = "".join(
        f"- {cat}: {cnt} 个\n" for cat, cnt in zip(_CATEGORIES, counts)
    )
    return (
        f"文档名称：{doc_name}\n\n"
        f"文档内容：\n{content}\n\n"
        f"请从上述文档中生成恰好 {n} 个问答对，按以下类别分配（若文档中某类内容不足，"
        f"可适当调整至相邻类别，但总数必须为 {n} 个）：\n"
        f"{distribution}\n"
        f"每个问题必须覆盖文档的不同章节或知识点，不得重复考察同一段落。\n"
        f"只输出 JSON 数组，不要添加任何其他文字。"
    )


def collect_md_files(ocr_dir: Path, name_filter: str | None) -> list[tuple[Path, str]]:
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR 目录不存在：{ocr_dir}")
    results = []
    for doc_dir in sorted(ocr_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_name = doc_dir.name
        if name_filter and name_filter.lower() not in doc_name.lower():
            continue
        md_files = sorted(doc_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"{doc_name}：未找到 Markdown 文件，跳过")
            continue
        results.append((md_files[0], doc_name))
    return results


_MAX_RETRIES = 3


def call_llm(client: OpenAI, doc_name: str, content: str, n: int) -> list[dict]:
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=settings.effective_llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(doc_name, content, n)},
                ],
                temperature=0.7,
                max_tokens=20000,
            )
            raw = response.choices[0].message.content or ""

            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                raise ValueError(f"LLM 响应中未找到 JSON 数组，原始输出：\n{raw[:300]}")

            pairs: list[dict] = json.loads(json_match.group())
            valid = [
                p for p in pairs
                if p.get("question", "").strip() and p.get("answer", "").strip()
            ]
            return valid

        except (ValueError, json.JSONDecodeError) as exc:
            last_err = exc
            logger.warning(f"  第 {attempt} 次尝试失败：{exc}，{'重试...' if attempt < _MAX_RETRIES else '放弃'}")
            if attempt < _MAX_RETRIES:
                time.sleep(2)

    raise RuntimeError(f"连续 {_MAX_RETRIES} 次均失败，最后错误：{last_err}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整篇文档一次性生成评估问答对。")
    parser.add_argument("--ocr-dir",   type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--output",    type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int,  default=10, help="每篇文档生成的问答对数量（默认 10）")
    parser.add_argument("--filter",    type=str,  default=None, help="文档名过滤（部分匹配）")
    parser.add_argument("--append",    action="store_true", help="追加写入而非覆盖")
    return parser.parse_args()


def main() -> None:
    _start_time = time.time()
    args = parse_args()

    client = OpenAI(
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
        timeout=120,
    )

    try:
        md_files = collect_md_files(args.ocr_dir, args.filter)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not md_files:
        logger.error("未找到任何文档，退出。")
        sys.exit(1)

    logger.info(f"共 {len(md_files)} 篇文档，每篇生成 {args.questions} 个问答对")

    write_mode = "a" if args.append else "w"
    total = 0

    with open(args.output, write_mode, encoding="utf-8") as out_f:
        for md_path, doc_name in md_files:
            logger.info(f"处理：{doc_name}")
            content = md_path.read_text(encoding="utf-8")

            try:
                pairs = call_llm(client, doc_name, content, args.questions)
            except Exception as exc:
                logger.error(f"  失败：{exc}")
                continue

            for pair in pairs:
                record = {
                    "question": pair["question"].strip(),
                    "answer":   pair["answer"].strip(),
                    "source":   doc_name,
                    "page":     0,
                    "category": pair.get("category", "factual").strip(),
                    "evidence": pair.get("evidence", "").strip(),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                total += 1

            logger.info(f"  生成 {len(pairs)} 个问答对")
            time.sleep(1)

    elapsed = time.time() - _start_time
    print("\n" + "=" * 50)
    print(f"  文档数     : {len(md_files)}")
    print(f"  问答对总数 : {total}")
    print(f"  输出文件   : {args.output}")
    print(f"  总耗时     : {elapsed:.1f}s")
    print("=" * 50)


if __name__ == "__main__":
    main()
