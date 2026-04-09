"""
图像描述 Skill
────────────────────────────────────────────────────────────────────────────
使用 Qwen-VL（qwen-vl-max）多模态模型，对本地图像文件或图像 URL 给出
详细的自然语言描述，支持图表、表格截图、示意图等场景。
"""
from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from loguru import logger


def _load_image_as_base64(path_or_url: str) -> tuple[str, str]:
    """
    将图像加载为 base64 字符串。
    若 path_or_url 以 http 开头，直接返回 URL；
    否则读取本地文件。

    返回 (media_type, base64_or_url)
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return "url", path_or_url

    p = Path(path_or_url)
    if not p.exists():
        raise FileNotFoundError(f"图像文件不存在：{path_or_url}")

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode()
    return mime, data


def build_image_describer_tool(vl_llm: Any):
    """
    返回图像描述 LangChain Tool。

    参数
    ----
    vl_llm : 支持视觉输入的 LangChain ChatModel（如 ChatOpenAI 指向 qwen-vl-max）。
    """

    @tool
    def image_describer(image_path: str, question: str = "") -> str:
        """
        使用多模态视觉模型描述图像内容。

        适用场景：
        - 分析 PDF 中提取的图表（折线图、柱状图、饼图等）
        - 识别表格截图中的数值
        - 理解技术架构图、流程图、示意图
        - 提取图像中的文字信息（OCR 补充）

        参数
        ----
        image_path : 本地图像文件路径（支持 PNG/JPG/WEBP）或公开图像 URL。
        question   : 可选，针对图像的具体问题。
                     留空时默认请求全面描述。
        """
        logger.info(f"[Skill:image_describer] path='{image_path}' q='{question[:60]}'")

        try:
            media_type, image_data = _load_image_as_base64(image_path)
        except Exception as exc:
            return f"图像加载失败：{exc}"

        # 构造 OpenAI vision 格式的消息
        if media_type == "url":
            image_content = {"type": "image_url", "image_url": {"url": image_data}}
        else:
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_data}"},
            }

        text_prompt = question.strip() if question.strip() else (
            "请详细描述这张图像的内容，包括：\n"
            "1. 图像类型（图表/表格/示意图/照片等）\n"
            "2. 主要信息和数据（如有数值请列出）\n"
            "3. 图像所表达的核心结论或含义"
        )

        from langchain_core.messages import HumanMessage as _HumanMessage
        msg = _HumanMessage(content=[
            image_content,
            {"type": "text", "text": text_prompt},
        ])

        try:
            response = vl_llm.invoke([msg])
            description = response.content.strip()
            logger.info(f"[Skill:image_describer] 描述 {len(description)} 字")
            return description
        except Exception as exc:
            logger.warning(f"[Skill:image_describer] 模型调用失败：{exc}")
            return f"图像描述失败：{exc}"

    return image_describer
