"""
检索子系统的单元测试。
所有外部调用（Chroma、DashScope）均已 mock。
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.sparse_retriever import SparseRetriever, _tokenise
from src.retrieval.hybrid_retriever import HybridRetriever


# ── 分词器 ────────────────────────────────────────────────────────────────

class TestTokenise:
    def test_latin(self):
        tokens = _tokenise("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_cjk(self):
        tokens = _tokenise("人工智能")
        assert "人" in tokens
        assert "工" in tokens
        assert "智" in tokens
        assert "能" in tokens

    def test_mixed(self):
        tokens = _tokenise("AI人工智能 model")
        assert "model" in tokens
        assert "人" in tokens

    def test_punctuation_removed(self):
        tokens = _tokenise("hello, world!")
        assert "hello" in tokens
        assert "," not in tokens


# ── SparseRetriever ───────────────────────────────────────────────────────

class TestSparseRetriever:
    def _make_docs(self, texts: list[str]) -> list[Document]:
        return [Document(page_content=t) for t in texts]

    def test_not_ready_before_build(self):
        retriever = SparseRetriever()
        assert not retriever.is_ready

    def test_ready_after_build(self):
        retriever = SparseRetriever()
        retriever.build_index(self._make_docs(["hello world"]))
        assert retriever.is_ready

    def test_retrieve_raises_if_not_built(self):
        retriever = SparseRetriever()
        with pytest.raises(RuntimeError, match="build_index"):
            retriever.retrieve("query")

    def test_retrieve_returns_relevant_doc(self):
        retriever = SparseRetriever()
        docs = self._make_docs(
            ["Python is great", "Machine learning is fun", "Deep learning models"]
        )
        retriever.build_index(docs)
        results = retriever.retrieve("Python", k=1)
        assert len(results) == 1
        assert "Python" in results[0][0].page_content

    def test_scores_descending(self):
        retriever = SparseRetriever()
        docs = self._make_docs(["deep learning deep", "shallow text", "deep deep deep"])
        retriever.build_index(docs)
        results = retriever.retrieve("deep", k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_corpus(self):
        retriever = SparseRetriever()
        retriever.build_index([])
        assert not retriever.is_ready


# ── HybridRetriever（已 mock）────────────────────────────────────────────

class TestHybridRetriever:
    def _make_doc(self, content: str) -> Document:
        return Document(page_content=content, metadata={"source": "test.pdf"})

    def test_rrf_deduplication(self):
        """同时出现在两路结果中的文档应融合为一条，而非重复出现。"""
        dense_mock = MagicMock()
        sparse_mock = MagicMock()
        sparse_mock.is_ready = True

        shared_doc = self._make_doc("shared content")
        dense_mock.retrieve.return_value = [(shared_doc, 0.9), (self._make_doc("d2"), 0.5)]
        sparse_mock.retrieve.return_value = [(shared_doc, 0.8), (self._make_doc("s2"), 0.4)]

        hybrid = HybridRetriever(dense_mock, sparse_mock, reranker=None)
        result = hybrid.retrieve("query", dense_k=2, sparse_k=2, final_k=5)

        # shared_doc 应仅出现一次
        shared_count = sum(1 for d in result if d.page_content == "shared content")
        assert shared_count == 1

    def test_falls_back_when_sparse_not_ready(self):
        dense_mock = MagicMock()
        sparse_mock = MagicMock()
        sparse_mock.is_ready = False  # BM25 索引未构建

        dense_mock.retrieve.return_value = [(self._make_doc("dense result"), 0.9)]

        hybrid = HybridRetriever(dense_mock, sparse_mock, reranker=None)
        result = hybrid.retrieve("query", final_k=5)

        assert len(result) >= 1
        sparse_mock.retrieve.assert_not_called()

    def test_reranker_called_when_provided(self):
        dense_mock = MagicMock()
        sparse_mock = MagicMock()
        sparse_mock.is_ready = False
        reranker_mock = MagicMock()

        docs = [self._make_doc(f"doc {i}") for i in range(3)]
        dense_mock.retrieve.return_value = [(d, 0.5) for d in docs]
        reranker_mock.rerank.return_value = docs[:2]

        hybrid = HybridRetriever(dense_mock, sparse_mock, reranker=reranker_mock)
        result = hybrid.retrieve("query", final_k=2)

        reranker_mock.rerank.assert_called_once()
        assert len(result) == 2


# ── ChromaVectorStore metadata 扁平化 ────────────────────────────────────

class TestMetadataFlatten:
    def test_flatten_nested_dict(self):
        from src.vectorstore.chroma_store import ChromaVectorStore

        doc = Document(
            page_content="test",
            metadata={
                "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.8},
                "source": "test.pdf",
                "page_num": 1,
            },
        )
        result = ChromaVectorStore._flatten_metadata(doc)
        assert result.metadata["source"] == "test.pdf"
        assert result.metadata["page_num"] == 1
        assert result.metadata["bbox_x0"] == 0.1
        assert result.metadata["bbox_y1"] == 0.8
        assert "bbox" not in result.metadata

    def test_flatten_list_to_string(self):
        from src.vectorstore.chroma_store import ChromaVectorStore

        doc = Document(
            page_content="test",
            metadata={"block_types": ["text", "header"]},
        )
        result = ChromaVectorStore._flatten_metadata(doc)
        assert result.metadata["block_types"] == "text, header"

    def test_flatten_preserves_scalars(self):
        from src.vectorstore.chroma_store import ChromaVectorStore

        doc = Document(
            page_content="test",
            metadata={"source": "a.pdf", "page": 3, "score": 0.95, "valid": True, "note": None},
        )
        result = ChromaVectorStore._flatten_metadata(doc)
        assert result.metadata["source"] == "a.pdf"
        assert result.metadata["page"] == 3
        assert result.metadata["score"] == 0.95
        assert result.metadata["valid"] is True
        assert result.metadata["note"] is None
