"""
图片向量库封装（ImageChromaStore）
──────────────────────────────────────────────────────────────────────────────
专门用于存储和检索图片的向量集合，使用 Qwen3-Embedding-VL-2B 进行
图片→向量 和 文本→向量 的统一映射（同一向量空间，支持跨模态检索）。

主要能力
--------
• add_images(paths, metadatas)  — 批量图片入库
• search_by_text(query, k)      — 文本 query 检索最相关图片
• search_by_image(path, k)      — 图片 query 检索相似图片（以图搜图）
• count()                       — 当前库中图片数量
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document
from loguru import logger

from config.settings import get_settings
from src.embeddings.qwen_vl_embedding import QwenVLLocalEmbeddings

settings = get_settings()

_UPSERT_BATCH = 20  # 图片批量写入大小（图片 embedding 更耗内存，适当减小）


class ImageChromaStore:
    """
    基于 Chroma 的图片向量库，使用 Qwen3-Embedding-VL 进行跨模态嵌入。

    参数
    ----
    collection_name : Chroma 集合标识符（默认从 settings 读取）。
    persist_dir     : 本地持久化目录（与文本库共用同一目录，集合名不同）。
    vl_model        : 已初始化的 QwenVLLocalEmbeddings 实例，为 None 时自动创建。
    """

    def __init__(
        self,
        collection_name: str | None = None,
        persist_dir: str | Path | None = None,
        vl_model: QwenVLLocalEmbeddings | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_image_collection_name
        self.persist_dir = Path(persist_dir or settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._vl = vl_model or QwenVLLocalEmbeddings(
            dimensions=settings.vl_embedding_dim
        )

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 获取或创建集合（不通过 LangChain，直接用 chromadb 原生 API 以便
        # 传入预先计算好的 embedding，避免 LangChain 再次调用 embed_documents）
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"[ImageStore] 集合 '{self.collection_name}' "
            f"路径：{self.persist_dir} | 已有图片数：{self.count()}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 写入
    # ──────────────────────────────────────────────────────────────────────

    def add_images(
        self,
        image_paths: list[str | Path],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """
        批量将图片向量化并存入 Chroma。

        参数
        ----
        image_paths : 图片文件路径列表。
        metadatas   : 与各图片对应的元数据列表（source、alt_text、caption 等）。
        ids         : 自定义 ID 列表；为 None 时自动使用文件的绝对路径作为 ID
                      （同一图片重复入库不会重复存储）。

        返回
        ----
        已写入的 Chroma ID 列表。
        """
        if not image_paths:
            return []

        image_paths = [Path(p) for p in image_paths]
        metadatas = metadatas or [{}] * len(image_paths)

        # 默认 ID = 图片绝对路径（保证幂等）
        if ids is None:
            ids = [str(p.resolve()) for p in image_paths]

        # 补充 source_path 到 metadata
        enriched_metas = []
        for i, (path, meta) in enumerate(zip(image_paths, metadatas)):
            m = dict(meta)
            m.setdefault("source_path", str(path.resolve()))
            m.setdefault("file_name", path.name)
            m["block_type"] = "figure"
            enriched_metas.append(self._flatten(m))

        all_ids: list[str] = []

        for i in range(0, len(image_paths), _UPSERT_BATCH):
            batch_paths = image_paths[i : i + _UPSERT_BATCH]
            batch_metas = enriched_metas[i : i + _UPSERT_BATCH]
            batch_ids = ids[i : i + _UPSERT_BATCH]

            # 向量化
            embeddings = self._vl.embed_images(batch_paths)

            # 图片没有文本内容，用文件名+alt_text 作为 document 字段（供调试）
            documents = [
                m.get("alt_text") or m.get("caption") or m.get("file_name", "")
                for m in batch_metas
            ]

            self._col.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=batch_metas,
            )
            all_ids.extend(batch_ids)
            logger.debug(f"[ImageStore] 写入第 {i // _UPSERT_BATCH + 1} 批：{len(batch_paths)} 张图片")

        logger.info(f"[ImageStore] 共写入 {len(all_ids)} 张图片")
        return all_ids

    # ──────────────────────────────────────────────────────────────────────
    # 检索
    # ──────────────────────────────────────────────────────────────────────

    def search_by_text(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        用文本 query 检索最相关的图片（跨模态检索）。

        返回
        ----
        (Document, 余弦相似度分数) 列表，分数越高越相关。
        Document.page_content = alt_text / caption（供 LLM 引用）
        Document.metadata 包含 source_path、file_name 等字段。
        """
        query_vec = self._vl.embed_query(query)
        return self._query(query_vec, k=k, filter=filter)

    def search_by_image(
        self,
        image_path: str | Path,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        用图片 query 检索相似图片（以图搜图）。
        """
        image_vec = self._vl.embed_image(image_path)
        return self._query(image_vec, k=k, filter=filter)

    def _query(
        self,
        query_embedding: list[float],
        k: int,
        filter: dict[str, Any] | None,
    ) -> list[tuple[Document, float]]:
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=min(k, max(self.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
        if filter:
            kwargs["where"] = filter

        result = self._col.query(**kwargs)

        docs: list[tuple[Document, float]] = []
        for doc_text, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            # cosine distance → similarity score
            score = 1.0 - dist
            docs.append((Document(page_content=doc_text or "", metadata=meta), score))

        docs.sort(key=lambda x: x[1], reverse=True)
        return docs

    # ──────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """返回当前集合中存储的图片数量。"""
        try:
            return self._col.count()
        except Exception:
            return 0

    def delete_collection(self) -> None:
        """删除并重建集合（破坏性操作，谨慎使用）。"""
        self._client.delete_collection(self.collection_name)
        self._col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"[ImageStore] 集合 '{self.collection_name}' 已删除并重建")

    @staticmethod
    def _flatten(meta: dict) -> dict:
        """将元数据中的复杂类型扁平化，确保 Chroma 可接受。"""
        clean: dict = {}
        for k, v in meta.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, (str, int, float, bool)):
                        clean[f"{k}_{sk}"] = sv
            elif isinstance(v, list):
                clean[k] = ", ".join(str(x) for x in v)
            else:
                clean[k] = str(v)
        return clean
