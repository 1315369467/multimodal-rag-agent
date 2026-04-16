"""
FastAPI 应用——多模态 RAG Agent
──────────────────────────────────────────────────────────────────────────────
接口列表
────────
GET  /health                  系统健康检查。
POST /v1/query                单轮或多轮问答。
POST /v1/ingest               解析并索引文档（服务端路径）。
POST /v1/ingest/upload        multipart 文件上传 + 入库。
DELETE /v1/collection         重置向量集合。

所有重型组件在 lifespan 上下文中初始化一次，
通过 FastAPI 依赖注入在请求间共享。
"""
from __future__ import annotations

import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
import json

from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger

from config.settings import get_settings
from src.agent.rag_agent import MultimodalRAGAgent
from src.document_parser.chunker import SemanticChunker
from src.document_parser.router import DocumentRouter
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import QwenReranker
from src.retrieval.sparse_retriever import SparseRetriever
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.image_store import ImageChromaStore
from .schemas import (
    DocumentItem,
    DocumentListResponse,
    DocumentUpdateRequest,
    ErrorResponse,
    HealthResponse,
    ImageSearchRequest,
    ImageSearchResponse,
    ImageSearchResult,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

settings = get_settings()

# ── 应用级单例 ─────────────────────────────────────────────────────────────
_store: ChromaVectorStore | None = None
_image_store: ImageChromaStore | None = None
_agent: MultimodalRAGAgent | None = None
_sparse_retriever: SparseRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """在服务启动时初始化所有重型组件。"""
    global _store, _image_store, _agent, _sparse_retriever

    logger.info("正在初始化多模态 RAG 系统……")
    _store = ChromaVectorStore()

    # 图片向量库（懒加载：仅当集合中已有数据或可创建时才加载 VL 模型）
    try:
        _image_store = ImageChromaStore()
        logger.info(f"图片向量库已就绪，共 {_image_store.count()} 张图片")
    except Exception as exc:
        logger.warning(f"图片向量库初始化失败，图片检索不可用：{exc}")
        _image_store = None

    dense = DenseRetriever(_store)
    _sparse_retriever = SparseRetriever()

    # 从已有语料库引导 BM25 索引
    existing_docs = _store.get_all_documents(limit=10_000)
    if existing_docs:
        _sparse_retriever.build_index(existing_docs)
        logger.info(f"BM25 索引引导完成：{len(existing_docs)} 条文档")

    reranker = QwenReranker()
    hybrid = HybridRetriever(dense, _sparse_retriever, reranker, image_store=_image_store)
    _agent = MultimodalRAGAgent(hybrid)

    logger.info("系统就绪。")
    yield

    logger.info("正在关闭服务……")


# ── FastAPI 应用 ───────────────────────────────────────────────────────────

app = FastAPI(
    title="多模态 RAG Agent API",
    description="企业级 PDF、扫描件、图表与公式的智能问答系统。",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 依赖注入辅助函数 ───────────────────────────────────────────────────────

def get_agent() -> MultimodalRAGAgent:
    if _agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 尚未初始化",
        )
    return _agent


def get_store() -> ChromaVectorStore:
    if _store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="向量库尚未初始化",
        )
    return _store


def get_image_store() -> ImageChromaStore:
    if _image_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="图片向量库尚未初始化，请先运行 scripts/ingest_images_from_md.py 入库图片",
        )
    return _image_store


AgentDep = Annotated[MultimodalRAGAgent, Depends(get_agent)]
StoreDep = Annotated[ChromaVectorStore, Depends(get_store)]
ImageStoreDep = Annotated[ImageChromaStore, Depends(get_image_store)]


# ── 路由 ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health(store: StoreDep) -> HealthResponse:
    """返回系统健康状态及集合统计信息。"""
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        collection=settings.chroma_collection_name,
        document_count=store.count(),
        image_count=_image_store.count() if _image_store is not None else 0,
    )


@app.post("/v1/query", response_model=QueryResponse, tags=["问答"])
async def query(request: QueryRequest, agent: AgentDep) -> QueryResponse:
    """
    使用完整的多模态 RAG 流水线回答问题。

    通过 chat_history 支持多轮对话。
    """
    t0 = time.perf_counter()

    history = [m.model_dump() for m in request.chat_history]

    try:
        result = agent.query(request.question, chat_history=history)
    except Exception as exc:
        logger.exception("问答错误：{}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[API] /v1/query 完成，耗时 {elapsed_ms:.0f}ms")

    return QueryResponse.from_agent_result(request.question, result)


@app.post("/v1/query/stream", tags=["问答"])
async def query_stream(request: QueryRequest, agent: AgentDep):
    """
    流式问答接口，以 Server-Sent Events (SSE) 格式逐步推送 Agent 思考过程。

    事件格式：`data: <JSON>\\n\\n`，以 `data: [DONE]\\n\\n` 结束。
    """
    history = [m.model_dump() for m in request.chat_history]

    def generate():
        try:
            for event in agent.stream_query(request.question, chat_history=history):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("流式问答错误：{}", exc)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/v1/ingest", response_model=IngestResponse, tags=["入库"])
async def ingest(request: IngestRequest, store: StoreDep) -> IngestResponse:
    """
    解析并索引服务端路径下的文档列表。

    支持 DocumentRouter 处理的所有格式（PDF、PNG、JPG 等）。
    """
    router = DocumentRouter()
    all_chunks = []
    processed: list[str] = []

    for file_path_str in request.file_paths:
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文件不存在：{file_path_str}",
            )

        logger.info(f"[Ingest] 处理：{file_path.name}")
        blocks = router.route(file_path)
        chunker = SemanticChunker(source_name=file_path.name)
        chunks = chunker.chunk(blocks)
        all_chunks.extend(chunks)
        processed.append(file_path.name)

    if all_chunks:
        store.add_documents(all_chunks)
        # 重建 BM25 索引
        if _sparse_retriever is not None:
            corpus = store.get_all_documents(limit=10_000)
            _sparse_retriever.build_index(corpus)

    return IngestResponse(
        status="success",
        documents_ingested=len(request.file_paths),
        chunks_created=len(all_chunks),
        files_processed=processed,
    )


@app.post("/v1/ingest/upload", response_model=IngestResponse, tags=["入库"])
async def ingest_upload(
    store: StoreDep,
    files: list[UploadFile] = File(...),
) -> IngestResponse:
    """
    直接上传文件并完成入库。接受 multipart/form-data 格式。
    """
    router = DocumentRouter()
    all_chunks = []
    processed: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload in files:
            tmp_path = Path(tmp_dir) / (upload.filename or "upload")
            content = await upload.read()
            tmp_path.write_bytes(content)

            logger.info(f"[Ingest] 已上传：{upload.filename}")
            blocks = router.route(tmp_path)
            chunker = SemanticChunker(source_name=upload.filename or tmp_path.name)
            chunks = chunker.chunk(blocks)
            all_chunks.extend(chunks)
            processed.append(upload.filename or tmp_path.name)

    if all_chunks:
        store.add_documents(all_chunks)
        if _sparse_retriever is not None:
            corpus = store.get_all_documents(limit=10_000)
            _sparse_retriever.build_index(corpus)

    return IngestResponse(
        status="success",
        documents_ingested=len(files),
        chunks_created=len(all_chunks),
        files_processed=processed,
    )


# ── 文档管理接口 ─────────────────────────────────────────────────────────

@app.get("/v1/documents", response_model=DocumentListResponse, tags=["文档管理"])
async def list_documents(
    store: StoreDep,
    offset: int = 0,
    limit: int = 20,
) -> DocumentListResponse:
    """
    分页获取文档列表。

    返回每条文档的 ID、完整内容和元数据，以及文档总数。
    """
    data = store.get_documents_paginated(offset=offset, limit=limit)
    items = [
        DocumentItem(
            id=doc_id,
            content=text or "",
            metadata=meta or {},
        )
        for doc_id, text, meta in zip(
            data["ids"], data["documents"], data["metadatas"]
        )
    ]
    return DocumentListResponse(items=items, total=data["total"])


@app.get("/v1/documents/{doc_id}", response_model=DocumentItem, tags=["文档管理"])
async def get_document(doc_id: str, store: StoreDep) -> DocumentItem:
    """根据 Chroma ID 获取单个文档的完整内容与元数据。"""
    col = store._client.get_collection(store.collection_name)
    result = col.get(ids=[doc_id], include=["documents", "metadatas"])
    if not result["ids"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在：{doc_id}",
        )
    return DocumentItem(
        id=result["ids"][0],
        content=result["documents"][0] or "",
        metadata=result["metadatas"][0] or {},
    )


@app.put("/v1/documents/{doc_id}", response_model=DocumentItem, tags=["文档管理"])
async def update_document(
    doc_id: str,
    body: DocumentUpdateRequest,
    store: StoreDep,
) -> DocumentItem:
    """
    更新文档内容和/或元数据，更新后会重新计算嵌入向量。

    content 和 metadata 均为可选字段，未提供的字段保持原值。
    """
    # 先获取现有文档
    col = store._client.get_collection(store.collection_name)
    existing = col.get(ids=[doc_id], include=["documents", "metadatas"])
    if not existing["ids"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在：{doc_id}",
        )

    # 合并更新字段
    new_content = body.content if body.content is not None else existing["documents"][0]
    new_metadata = body.metadata if body.metadata is not None else (existing["metadatas"][0] or {})

    store.update_document(doc_id, new_content, new_metadata)

    return DocumentItem(id=doc_id, content=new_content, metadata=new_metadata)


@app.delete("/v1/documents/{doc_id}", tags=["文档管理"])
async def delete_document(doc_id: str, store: StoreDep) -> dict[str, str]:
    """根据 Chroma ID 删除单个文档。"""
    # 检查文档是否存在
    col = store._client.get_collection(store.collection_name)
    existing = col.get(ids=[doc_id])
    if not existing["ids"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在：{doc_id}",
        )
    store.delete_documents_by_ids([doc_id])
    return {"status": "已删除", "doc_id": doc_id}


@app.delete("/v1/collection", tags=["管理"])
async def reset_collection(store: StoreDep) -> dict[str, str]:
    """
    删除并重建向量集合。**破坏性操作——请谨慎使用。**
    """
    store.delete_collection()
    if _sparse_retriever is not None:
        _sparse_retriever._bm25 = None
        _sparse_retriever._documents = []
    logger.warning("[API] 集合已通过 API 重置")
    return {"status": "集合已重置", "collection": settings.chroma_collection_name}


# ── 图片检索接口 ──────────────────────────────────────────────────────────

@app.post("/v1/images/search", response_model=ImageSearchResponse, tags=["图片检索"])
async def image_search(
    request: ImageSearchRequest,
    img_store: ImageStoreDep,
) -> ImageSearchResponse:
    """
    用文本 query 检索最相关的图片（跨模态检索）。

    返回图片路径、相关元数据及相似度分数。
    图片可通过 GET /v1/images/file?path=<source_path> 接口获取原图。
    """
    try:
        raw = img_store.search_by_text(request.query, k=request.top_k)
    except Exception as exc:
        logger.exception("图片检索错误：{}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    results: list[ImageSearchResult] = []
    for doc, score in raw:
        meta = doc.metadata
        src_path = meta.get("source_path", "")
        results.append(
            ImageSearchResult(
                source_path=src_path,
                file_name=meta.get("file_name", Path(src_path).name if src_path else ""),
                alt_text=meta.get("alt_text", ""),
                caption=meta.get("caption", ""),
                source=meta.get("source", ""),
                score=round(float(score), 4),
                image_url=f"/v1/images/file?path={src_path}" if src_path else "",
            )
        )

    return ImageSearchResponse(
        results=results,
        total=len(results),
        image_count=img_store.count(),
    )


@app.get("/v1/images/file", tags=["图片检索"])
async def get_image_file(
    path: str = Query(..., description="图片绝对路径"),
) -> FileResponse:
    """
    根据绝对路径返回图片文件（供前端展示使用）。

    出于安全考虑，仅允许访问已入库的图片（路径须存在于文件系统中）。
    """
    img_path = Path(path)
    if not img_path.is_absolute():
        raise HTTPException(status_code=400, detail="必须提供绝对路径")
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail=f"图片文件不存在：{path}")

    # 仅允许图片类型
    _ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
    if img_path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="不支持的图片格式")

    _MEDIA_TYPES = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".gif": "image/gif",
    }
    media_type = _MEDIA_TYPES.get(img_path.suffix.lower(), "image/png")
    return FileResponse(str(img_path), media_type=media_type)


@app.get("/v1/images/count", tags=["图片检索"])
async def image_count() -> dict[str, int]:
    """返回图片向量库中已索引的图片数量。"""
    count = _image_store.count() if _image_store is not None else 0
    return {"count": count}


# ── 全局异常处理 ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc: Exception):
    logger.exception("未处理的异常：{}", exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail=str(exc), error_type=type(exc).__name__
        ).model_dump(),
    )
