"""
全局配置管理（基于 pydantic-settings）
所有配置项均可通过环境变量或 .env 文件覆盖。
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── DashScope / OpenAI 兼容接口 ────────────────────────────────────────
    dashscope_api_key: str = Field("", alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # ── 模型名称 ───────────────────────────────────────────────────────────
    llm_mode: str = "api"                        # "api" 使用 DashScope，"local" 使用本地部署模型
    llm_model: str = "qwen3.6-plus"          # API 模式使用的模型名称
    local_llm_base_url: str = "http://localhost:9000/v1"  # 本地模型地址
    local_llm_model: str = "qwen3-8b"              # 本地模型名称
    vl_model: str = "qwen-vl-max"               # Qwen3-VL 多模态视觉
    
    embedding_mode: str = "local"              # "local" 或 "api"
    embedding_model: str = "text-embedding-v4"  # API 模式使用的模型名称
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # 本地模型路径
    embedding_dim: int = 1024                   # 向量维度
    
    enable_rerank: bool = False                   # 是否启用精排（ False True ）
    reranker_mode: str = "local"                # "local" 或 "api"
    reranker_model: str = "qwen3-rerank"          # API 模式使用的模型名称
    local_reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"  # 本地模型路径

    # ── 向量数据库 ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "multimodal_rag"

    # ── 文档处理 ───────────────────────────────────────────────────────────
    # chunk 切分策略： fixed | sentence | paragraph | markdown | recursive | semantic
    chunk_strategy: str = "semantic"
    chunk_size: int = 1024
    chunk_overlap: int = 32
    max_workers: int = 4

    # ── 检索参数 ───────────────────────────────────────────────────────────
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rerank_top_k: int = 10
    final_top_k: int = 5
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # ── LLM 生成参数 ───────────────────────────────────────────────────────
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    llm_request_timeout: int = 30

    # ── API 服务 ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── 日志 ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # ── 派生属性 ──────────────────────────────────────────────────────────────
    @property
    def effective_llm_model(self) -> str:
        return self.local_llm_model if self.llm_mode == "local" else self.llm_model

    @property
    def effective_llm_base_url(self) -> str:
        return self.local_llm_base_url if self.llm_mode == "local" else self.dashscope_base_url

    @property
    def effective_llm_api_key(self) -> str:
        return "local" if self.llm_mode == "local" else self.dashscope_api_key

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def log_file_path(self) -> Path:
        return Path(self.log_file)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回 Settings 单例（首次调用后缓存，后续调用直接复用）。"""
    return Settings()