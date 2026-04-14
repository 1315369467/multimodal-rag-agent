"""
视觉通道解析器（Qwen-VL）
──────────────────────────────────────────────────────────────────────────────
通过 Qwen3-VL 多模态理解，将独立图像文件转换为高质量文本。
支持以下场景：
  • 复杂图表与示意图
  • 数学公式
  • 扫描图像文件

模型接收 base64 编码的图像和结构化提示词，返回 Markdown 格式的转录文本。
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

from loguru import logger
from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import get_settings
from .base_parser import BaseDocumentParser, BlockType, ParsedBlock

settings = get_settings()

_VL_SYSTEM_PROMPT = """\
你是一个专业的文档解析助手。请对提供的文档图像进行完整的结构化转录，要求：
1. 保留所有文字内容，包括页眉页脚。
2. 表格用 Markdown 表格格式输出，保持行列结构。
3. 数学公式用 LaTeX 格式：行内公式用 $...$，独立公式用 $$...$$。
4. 图表请给出详细的图注描述，说明图表类型、坐标轴、趋势等关键信息。
5. 保持原文档的逻辑段落结构，用空行分隔段落。
6. 输出纯文本/Markdown，不要输出任何解释性前缀。
"""


def _encode_image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """将 PIL Image 编码为 base64 字符串，供视觉 API 调用。"""
    buf = io.BytesIO()
    image.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VisionParser(BaseDocumentParser):
    """
    使用 Qwen-VL 处理独立图像文件的多模态解析器。

    使用示例
    --------
    parser = VisionParser()
    blocks = parser.parse(Path("chart.png"))
    """

    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )

    @property
    def supported_extensions(self) -> set[str]:
        return {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def parse(self, file_path: Path) -> list[ParsedBlock]:
        """直接解析一个图像文件。"""
        image = Image.open(str(file_path)).convert("RGB")
        content = self._call_vl_model(image, source=file_path.name)
        return [
            ParsedBlock(
                content=content,
                block_type=BlockType.FIGURE,
                page_num=0,
                metadata={"source": "vision_parser", "file": file_path.name},
            )
        ]

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_vl_model(self, image: Image.Image, source: str = "") -> str:
        """
        通过 OpenAI 兼容接口将图像发送给 Qwen-VL，返回转录文本。
        遇到瞬时错误最多自动重试 3 次。
        """
        b64 = _encode_image_to_base64(image)
        mime = "image/jpeg"

        response = self._client.chat.completions.create(
            model=settings.vl_model,
            messages=[
                {"role": "system", "content": _VL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "请对以上文档图像进行完整转录，输出结构化的Markdown文本。",
                        },
                    ],
                },
            ],
            max_tokens=4096,
            temperature=0.0,
        )

        text = response.choices[0].message.content or ""
        logger.debug(f"[VisionParser] {source}: 返回 {len(text)} 个字符")
        return text.strip()
