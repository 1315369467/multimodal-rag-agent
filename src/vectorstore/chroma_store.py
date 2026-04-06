"""
Chroma 向量库封装
──────────────────────────────────────────────────────────────────────────────
对 LangChain Chroma 集成的轻量封装，新增以下能力：
  • 持久化存储，自动创建目录。
  • 按文档 ID 去重的批量 upsert。
  • 元数据过滤辅助方法。
  • 集合统计信息。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings
from src.embeddings.qwen_embedding import QwenEmbeddings

settings = get_settings()

_UPSERT_BATCH = 100  # Chroma 推荐的单次写入批大小


class ChromaVectorStore:
    """
    支持 upsert、检索和统计的持久化 Chroma 集合。

    参数
    ----
    collection_name : Chroma 集合标识符。
    persist_dir     : 本地持久化目录。
    embedding_fn    : 向量化函数（默认使用 QwenEmbeddings）。
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | Path | None = None,
        embedding_fn: QwenEmbeddings | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = embedding_fn or QwenEmbeddings()

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._store = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self._embedding_fn,
        )
        logger.info(
            f"[ChromaStore] 集合 '{self.collection_name}' "
            f"路径：{self.persist_dir} | "
            f"已有文档数：{self.count()}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 写入
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _flatten_metadata(doc: Document) -> Document:
        """将 metadata 中的 dict/list 等复杂类型扁平化或移除，以兼容 Chroma。"""
        clean = {}
        for key, value in doc.metadata.items():
            if isinstance(value, dict):
                # 将嵌套 dict 展开为 key.subkey 形式
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (str, int, float, bool)):
                        clean[f"{key}_{sub_key}"] = sub_val
            elif isinstance(value, list):
                # 列表转为逗号分隔字符串
                clean[key] = ", ".join(str(v) for v in value)
            elif value is None or isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        doc.metadata = clean
        return doc

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        """
        批量 upsert 文档，返回已存储的 ID 列表。

        若提供 ids，其长度须与 documents 对齐；
        否则由 LangChain 自动生成 UUID。
        """
        # 扁平化 metadata，确保所有值均为 Chroma 支持的标量类型
        documents = [self._flatten_metadata(doc) for doc in documents]
        all_ids: list[str] = []
        for i in range(0, len(documents), _UPSERT_BATCH):
            batch = documents[i : i + _UPSERT_BATCH]
            batch_ids = ids[i : i + _UPSERT_BATCH] if ids else None
            returned = self._store.add_documents(batch, ids=batch_ids)
            all_ids.extend(returned)
            logger.debug(f"[ChromaStore] 已写入第 {i // _UPSERT_BATCH + 1} 批：{len(batch)} 条")

        logger.info(f"[ChromaStore] 共写入 {len(all_ids)} 条文档")
        return all_ids

    # ──────────────────────────────────────────────────────────────────────
    # 读取 / 检索
    # ──────────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """
        稠密向量相似度检索。

        参数
        ----
        query  : 查询字符串（内部自动向量化）。
        k      : 返回结果数量。
        filter : Chroma 元数据过滤字典，例如 {"source": {"$eq": "doc.pdf"}}。
        """
        k = k or settings.dense_top_k
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Document, float]]:
        """返回 (Document, 分数) 对；L2 距离越小表示越相似。"""
        k = k or settings.dense_top_k
        return self._store.similarity_search_with_score(query, k=k, filter=filter)

    def as_retriever(self, **kwargs: Any):
        """返回 LangChain VectorStoreRetriever 实例。"""
        return self._store.as_retriever(**kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """返回当前集合中存储的文档数量。"""
        try:
            col = self._client.get_collection(self.collection_name)
            return col.count()
        except Exception:
            return 0

    def delete_collection(self) -> None:
        """删除并重建集合（破坏性操作，谨慎使用）。"""
        self._client.delete_collection(self.collection_name)
        self._store = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self._embedding_fn,
        )
        logger.warning(f"[ChromaStore] 集合 '{self.collection_name}' 已删除并重建")

    def get_all_documents(self, limit: int = 1000) -> list[Document]:
        """取出所有已存储文档（用于引导 BM25 语料库）。"""
        col = self._client.get_collection(self.collection_name)
        result = col.get(limit=limit, include=["documents", "metadatas"])
        docs = []
        for text, meta in zip(result["documents"], result["metadatas"]):
            docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    # ──────────────────────────────────────────────────────────────────────
    # 文档 CRUD（供前端管理接口使用）
    # ──────────────────────────────────────────────────────────────────────

    def get_documents_paginated(
        self, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """
        分页获取文档，返回包含 Chroma ID 的文档列表及总数。

        返回格式
        --------
        {
            "ids": [...],
            "documents": [...],
            "metadatas": [...],
            "total": int,
        }
        """
        col = self._client.get_collection(self.collection_name)
        total = col.count()
        result = col.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        return {
            "ids": result["ids"],
            "documents": result["documents"],
            "metadatas": result["metadatas"],
            "total": total,
        }

    def delete_documents_by_ids(self, ids: list[str]) -> None:
        """根据 Chroma ID 列表批量删除文档。"""
        col = self._client.get_collection(self.collection_name)
        col.delete(ids=ids)
        logger.info(f"[ChromaStore] 已删除 {len(ids)} 条文档")

    def update_document(self, doc_id: str, content: str, metadata: dict) -> None:
        """
        更新单个文档的内容和元数据，并重新计算嵌入向量。

        参数
        ----
        doc_id   : Chroma 文档 ID。
        content  : 新的文档文本。
        metadata : 新的元数据字典。
        """
        col = self._client.get_collection(self.collection_name)
        # 使用嵌入函数重新计算向量
        new_embedding = self._embedding_fn.embed_documents([content])[0]
        col.update(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata],
            embeddings=[new_embedding],
        )
        logger.info(f"[ChromaStore] 已更新文档 {doc_id}")
