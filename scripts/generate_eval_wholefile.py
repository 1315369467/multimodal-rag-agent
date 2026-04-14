"""
基于完整 OCR Markdown 文件生成评估数据集
──────────────────────────────────────────────────────────────────────────────
读取 data/ocr_output 下的 Markdown 文件，将整篇文档（或截断后的版本）
一次性交给 LLM，生成指定数量的高质量 Q&A 对，写入 JSONL 文件。

用法
────
python scripts/generate_eval_wholefile.py
python scripts/generate_eval_wholefile.py --questions 20 --output data/eval_ocr_whole.jsonl
python scripts/generate_eval_wholefile.py --filter "Qwen3" --questions 30
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
你是一个专业的技术文档问答数据集标注员。
给定一篇完整的技术文档，请生成若干高质量的问答对，用于评估 RAG 检索系统的效果。

要求：
1. 问题必须能从文档中找到明确答案，不依赖外部知识。
2. 答案简洁准确，直接来自原文，不推断不发挥。
3. 覆盖文档中不同章节和不同类型的知识点：
   - factual：具体数值、参数、配置（如层数、维度、数据量）
   - comparative：与其他模型/方法的对比
   - procedural：训练步骤、方法描述
4. 问题用中文提问，答案保持原文语言。
5. 只输出 JSON 数组，不要添加任何其他文字。

输出格式：
[
  {"question": "问题", "answer": "答案", "category": "factual"},
  ...
]
"""


def build_user_prompt(doc_name: str, content: str, n: int) -> str:
    return (
        f"文档名称：{doc_name}\n\n"
        f"文档内容：\n{content}\n\n"
        f"请从上述文档中生成 {n} 个问答对（JSON 数组）。"
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


def call_llm(client: OpenAI, doc_name: str, content: str, n: int) -> list[dict]:
    response = client.chat.completions.create(
        model=settings.effective_llm_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(doc_name, content, n)},
        ],
        temperature=0.4,
        # max_tokens=4096,
    )
    raw = response.choices[0].message.content or ""

    json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"LLM 响应中未找到 JSON 数组，原始输出：\n{raw[:500]}")

    pairs: list[dict] = json.loads(json_match.group())
    valid = [
        p for p in pairs
        if p.get("question", "").strip() and p.get("answer", "").strip()
    ]
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整篇文档一次性生成评估问答对。")
    parser.add_argument("--ocr-dir",   type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--output",    type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=int,  default=10, help="每篇文档生成的问答对数量（默认 20）")
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
