"""
基于 OCR Markdown 自动生成评估数据集
──────────────────────────────────────────────────────────────────────────────
读取 data/ocr_output 下的所有 Markdown 文件，逐节调用 LLM 生成 Q&A 对，
并写入 JSONL 评估数据集文件。

生成的 JSONL 格式（与现有 eval_dataset.jsonl 一致）：
  {"question": "...", "answer": "...", "source": "文档名", "page": 0, "category": "factual"}

用法
────
# 生成并写入默认输出文件 data/eval_dataset_ocr.jsonl
python scripts/generate_eval_from_ocr.py

# 指定 OCR 目录和输出文件
python scripts/generate_eval_from_ocr.py --ocr-dir data/ocr_output --output data/my_eval.jsonl

# 控制每个章节生成的问题数量
python scripts/generate_eval_from_ocr.py --questions-per-section 3

# 仅处理指定文档（部分名称匹配即可）
python scripts/generate_eval_from_ocr.py --filter "Qwen3"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from config.settings import get_settings

settings = get_settings()

DEFAULT_OCR_DIR = Path(__file__).resolve().parents[1] / "data" / "ocr_output"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "eval_dataset_ocr.jsonl"

# 章节内容长度阈值：过短的章节跳过（字符数）
MIN_SECTION_CHARS = 150
# LLM 请求之间的间隔（秒），避免触发限流
REQUEST_INTERVAL = 0.5


# ── 数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class Section:
    heading: str          # 标题文本（不含 # 符号）
    heading_level: int    # 标题层级（1=H1, 2=H2 ...）
    content: str          # 该章节下的正文内容
    source: str           # 所属文档名


@dataclass
class QAPair:
    question: str
    answer: str
    source: str
    page: int = 0
    category: str = "factual"


# ── Markdown 章节解析 ─────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def parse_sections(md_text: str, source: str, max_level: int = 2) -> list[Section]:
    """
    将 Markdown 文本切分为章节列表。
    只在 <= max_level 的标题处切分，避免粒度过细。
    """
    sections: list[Section] = []
    current_heading = ""
    current_level = 1
    current_lines: list[str] = []

    def flush():
        content = "\n".join(current_lines).strip()
        if content and len(content) >= MIN_SECTION_CHARS:
            sections.append(
                Section(
                    heading=current_heading,
                    heading_level=current_level,
                    content=content,
                    source=source,
                )
            )

    for line in md_text.split("\n"):
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            if level <= max_level:
                flush()
                current_heading = heading_text
                current_level = level
                current_lines = []
            else:
                # 子级标题保留在正文中
                current_lines.append(line)
        else:
            current_lines.append(line)

    flush()
    return sections


# ── LLM 调用 ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
你是一个专业的技术文档问答数据集标注员。
给定一段技术文档文本，请从中提取若干问答对，用于评估 RAG 系统的检索效果。

要求：
1. 问题必须能从给定文本中找到明确答案，不依赖外部知识。
2. 答案应简洁准确，直接来自原文，不要推断或发挥。
3. 问题应涵盖不同粒度：具体数值、架构特点、方法描述等。
4. 问题用中文或英文均可（与原文语言一致）。
5. 严格按照 JSON 数组格式输出，不要添加任何其他文字。

输出格式（JSON 数组）：
[
  {"question": "问题1", "answer": "答案1", "category": "factual"},
  {"question": "问题2", "answer": "答案2", "category": "factual"}
]

category 可选值：factual（事实类）、comparative（比较类）、procedural（步骤类）。
"""


def build_user_prompt(section: Section, n: int) -> str:
    # 限制输入长度，避免超出 token 限制
    content = section.content[:3000]
    return (
        f"文档章节标题：{section.heading}\n\n"
        f"章节内容：\n{content}\n\n"
        f"请从上述内容中生成 {n} 个问答对（JSON 数组）。"
    )


def call_llm(client: OpenAI, section: Section, n: int) -> list[QAPair]:
    """调用 LLM 生成 Q&A，解析 JSON 响应，失败时返回空列表。"""
    try:
        response = client.chat.completions.create(
            model=settings.effective_llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(section, n)},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        # 提取 JSON 数组（兼容 LLM 在回复前后添加说明文字的情况）
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not json_match:
            logger.warning(f"[LLM] {section.source} / {section.heading}：未找到 JSON 数组，跳过")
            return []
        pairs_raw: list[dict] = json.loads(json_match.group())
        result = []
        for item in pairs_raw:
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            cat = str(item.get("category", "factual")).strip()
            if q and a:
                result.append(QAPair(question=q, answer=a, source=section.source, category=cat))
        return result
    except Exception as exc:
        logger.error(f"[LLM] {section.source} / {section.heading}：{exc}")
        return []


# ── 文件收集 ─────────────────────────────────────────────────────────────

def collect_md_files(ocr_dir: Path, name_filter: str | None) -> list[tuple[Path, str]]:
    """返回 (md文件路径, 文档名) 列表。"""
    if not ocr_dir.exists():
        raise FileNotFoundError(f"OCR 输出目录不存在：{ocr_dir}")
    results = []
    for doc_dir in sorted(ocr_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_name = doc_dir.name
        if name_filter and name_filter.lower() not in doc_name.lower():
            continue
        md_files = sorted(doc_dir.rglob("*.md"))
        if not md_files:
            logger.warning(f"[扫描] {doc_name}：未找到 Markdown 文件，跳过")
            continue
        results.append((md_files[0], doc_name))
    return results


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 OCR Markdown 自动生成评估数据集。"
    )
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--questions-per-section", type=int, default=2,
        help="每个章节生成的问答对数量（默认：2）。",
    )
    parser.add_argument(
        "--max-sections", type=int, default=None,
        help="每个文档最多处理的章节数（默认：全部）。",
    )
    parser.add_argument(
        "--max-level", type=int, default=2,
        help="按几级标题切分章节（默认：2，即 H1/H2）。",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="只处理文档名包含该字符串的文档（大小写不敏感）。",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="追加写入输出文件（默认：覆盖）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = OpenAI(
        api_key=settings.effective_llm_api_key,
        base_url=settings.effective_llm_base_url,
        timeout=60,
    )

    try:
        md_files = collect_md_files(args.ocr_dir, args.filter)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not md_files:
        logger.error("未找到任何 Markdown 文件，退出。")
        sys.exit(1)

    logger.info(f"找到 {len(md_files)} 个文档，模型：{settings.effective_llm_model}")

    write_mode = "a" if args.append else "w"
    total_pairs = 0
    failed_sections = 0

    with open(args.output, write_mode, encoding="utf-8") as out_f:
        for md_path, doc_name in md_files:
            logger.info(f"处理文档：{doc_name}")
            md_text = md_path.read_text(encoding="utf-8")
            sections = parse_sections(md_text, doc_name, max_level=args.max_level)

            if args.max_sections:
                sections = sections[: args.max_sections]

            logger.info(f"  切分出 {len(sections)} 个章节")

            for section in tqdm(sections, desc=doc_name, unit="节", leave=False):
                pairs = call_llm(client, section, args.questions_per_section)
                if not pairs:
                    failed_sections += 1
                for pair in pairs:
                    record = {
                        "question": pair.question,
                        "answer": pair.answer,
                        "source": pair.source,
                        "page": pair.page,
                        "category": pair.category,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_pairs += 1
                time.sleep(REQUEST_INTERVAL)

    print("\n" + "=" * 60)
    print(f"  处理文档数   : {len(md_files)}")
    print(f"  生成问答对数 : {total_pairs}")
    print(f"  失败章节数   : {failed_sections}")
    print(f"  输出文件     : {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
