"""
FastAPI 请求/响应层的 Pydantic 数据模型定义。
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from config.settings import get_settings

_settings = get_settings()


# ── 请求模型 ───────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class IngestRequest(BaseModel):
    file_paths: list[str] = Field(..., min_length=1)
    collection_name: Optional[str] = None


# ── 响应模型 ───────────────────────────────────────────────────────────────

class SourceReference(BaseModel):
    source: str
    page: str
    block_type: str
    content: str = ""
    source_path: str = ""   # 图片绝对路径（block_type=="figure" 时填充）
    image_url: str = ""     # 后端图片服务 URL（如 /v1/images/file?path=...）
    score: float | None = None  # 相似度分数（image_search 返回）


class ToolCallStep(BaseModel):
    """Agent 单次工具调用的摘要，用于前端展示思考过程。"""
    tool: str                          # 工具名称
    args: dict[str, Any]               # 调用参数
    result_summary: str = ""           # 结果摘要（可读文本）
    status: str = "done"               # running | done | error
    elapsed_ms: Optional[int] = None   # 工具执行耗时（毫秒）


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    question: str
    total_sources: int
    tool_steps: list[ToolCallStep] = Field(default_factory=list)
    total_elapsed_ms: Optional[int] = None   # 整体耗时（毫秒）

    @classmethod
    def from_agent_result(cls, question: str, result: dict[str, Any]) -> "QueryResponse":
        sources = [SourceReference(**s) for s in result.get("sources", [])]
        tool_steps = [ToolCallStep(**s) for s in result.get("tool_steps", [])]
        return cls(
            answer=result["answer"],
            sources=sources,
            question=question,
            total_sources=len(sources),
            tool_steps=tool_steps,
            total_elapsed_ms=result.get("total_elapsed_ms"),
        )


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    files_processed: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    collection: str
    document_count: int
    image_count: int = 0
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    detail: str
    error_type: str


# ── 文档管理模型 ─────────────────────────────────────────────────────────

class DocumentItem(BaseModel):
    """单个文档条目，包含 Chroma ID、内容与元数据。"""
    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    """分页文档列表响应，包含文档条目和总数。"""
    items: list[DocumentItem]
    total: int


class DocumentUpdateRequest(BaseModel):
    """文档更新请求，内容和元数据均为可选字段。"""
    content: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


# ── 图片检索模型 ──────────────────────────────────────────────────────────

class ImageSearchRequest(BaseModel):
    """图片检索请求：用文本 query 检索最相关的图片。"""
    query: str = Field(..., min_length=1, max_length=2048)
    top_k: int = Field(default_factory=lambda: _settings.image_top_k, ge=1, le=50)


class ImageSearchResult(BaseModel):
    """单条图片检索结果。"""
    source_path: str          # 图片绝对路径
    file_name: str            # 图片文件名
    alt_text: str = ""        # Markdown 中的 alt 描述
    caption: str = ""         # 图注文字
    source: str = ""          # 来源 md 文件名
    score: float              # 余弦相似度分数（0~1）
    image_url: str = ""       # 后端图片服务 URL（/v1/images/file?path=...）


class ImageSearchResponse(BaseModel):
    """图片检索响应。"""
    results: list[ImageSearchResult]
    total: int                # 本次返回数量
    image_count: int          # 图片库中图片总数
