"""
Qwen3-VL-Embedding 本地多模态向量化
──────────────────────────────────────────────────────────────────────────────
基于 Qwen/Qwen3-VL-Embedding-2B 官方推荐流程（format_model_input + process_vision_info）
实现文本/图片统一向量空间编码。

• embed_documents(texts)  — 与 LangChain Embeddings 接口兼容（文本）
• embed_query(text)       — 查询向量化（与文本相同路径）
• embed_image(path)       — 单张图片向量化
• embed_images(paths)     — 批量图片向量化
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from langchain_core.embeddings import Embeddings
from loguru import logger

from config.settings import get_settings

settings = get_settings()

IMAGE_PATCH_SIZE = 16
IMAGE_FACTOR = IMAGE_PATCH_SIZE * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
DEFAULT_INSTRUCTION = "Represent the user's input."


class QwenVLLocalEmbeddings(Embeddings):
    """基于 Qwen3-VL-Embedding-2B 的多模态向量化。"""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        dimensions: int = -1,
        max_length: int = 8192,
        batch_size: int = 4,
        instruction: str | None = None,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
    ) -> None:
        from transformers import AutoProcessor
        from transformers.utils import is_flash_attn_2_available

        self.model_name = model_name_or_path or settings.local_vl_embedding_model
        self.dimensions = dimensions
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction or DEFAULT_INSTRUCTION
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda else torch.float32

        kwargs: dict = dict(trust_remote_code=True, dtype=dtype, local_files_only=True)
        if is_flash_attn_2_available() and use_cuda:
            kwargs["attn_implementation"] = "flash_attention_2"

        # 必须使用官方 Qwen3VLForEmbedding（scripts 里的 trust_remote_code 脚本），
        # AutoModel 会加载成 Qwen3VLModel，safetensors 里的 "model." 前缀权重会失配。
        EmbeddingCls = self._load_embedding_class(self.model_name)
        self._model = EmbeddingCls.from_pretrained(self.model_name, **kwargs)
        if use_cuda:
            self._model = self._model.cuda()
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
            padding_side="right",
        )

        logger.info(f"[VL-Embedding] 模型已加载: {self.model_name} (cuda={use_cuda})")

    @staticmethod
    def _load_embedding_class(model_name: str):
        """动态导入 model repo 中 scripts/qwen3_vl_embedding.py 的 Qwen3VLForEmbedding。"""
        import importlib.util

        repo_dir = QwenVLLocalEmbeddings._resolve_repo_dir(model_name)
        script_path = repo_dir / "scripts" / "qwen3_vl_embedding.py"
        if not script_path.exists():
            raise FileNotFoundError(f"找不到官方脚本: {script_path}")
        spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_script", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.Qwen3VLForEmbedding

    @staticmethod
    def _resolve_repo_dir(model_name: str) -> Path:
        """解析模型仓库目录：本地绝对路径 → snapshot_download → HF cache 直接扫描。"""
        direct = Path(model_name)
        if direct.is_dir() and (direct / "config.json").exists():
            return direct

        from huggingface_hub import snapshot_download
        try:
            return Path(snapshot_download(model_name, local_files_only=True))
        except Exception as exc:
            logger.debug(f"[VL-Embedding] snapshot_download 失败，回退到缓存扫描: {exc}")

        from huggingface_hub import constants
        cache_root = Path(constants.HF_HUB_CACHE)
        repo_folder = "models--" + model_name.replace("/", "--")
        snapshots_dir = cache_root / repo_folder / "snapshots"
        if snapshots_dir.is_dir():
            candidates = [p for p in snapshots_dir.iterdir() if p.is_dir() and (p / "config.json").exists()]
            if candidates:
                # 选最近修改的一个
                return max(candidates, key=lambda p: p.stat().st_mtime)

        raise FileNotFoundError(
            f"无法在本地解析模型仓库: {model_name}（检查 HF_HOME={cache_root} 下是否存在有效 snapshot）"
        )

    # ──────────────────────────────────────────────────────────────────────
    # LangChain Embeddings 接口
    # ──────────────────────────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode([{"text": t} for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return self._encode([{"text": text}])[0]

    def embed_image(self, image_path: str | Path) -> List[float]:
        return self.embed_images([image_path])[0]

    def embed_images(self, image_paths: List[str | Path]) -> List[List[float]]:
        if not image_paths:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_inputs: list[dict] = []
            valid_idx: list[int] = []
            for j, p in enumerate(batch_paths):
                p = Path(p)
                if not p.exists():
                    logger.warning(f"[VL-Embedding] 图片不存在: {p}")
                    continue
                batch_inputs.append({"image": str(p.resolve())})
                valid_idx.append(j)

            vecs = self._encode(batch_inputs) if batch_inputs else []

            # 填充零向量给失败的样本
            batch_out: list[list[float]] = [[] for _ in batch_paths]
            zero: list[float] | None = None
            for k, v in zip(valid_idx, vecs):
                batch_out[k] = v
            for k in range(len(batch_paths)):
                if not batch_out[k]:
                    if zero is None:
                        zero = [0.0] * (len(vecs[0]) if vecs else 1536)
                    batch_out[k] = zero

            all_vectors.extend(batch_out)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_vectors

    # ──────────────────────────────────────────────────────────────────────
    # 核心编码流程（对齐官方 Qwen3VLEmbedder.process）
    # ──────────────────────────────────────────────────────────────────────

    def _format_conversation(self, item: dict[str, Any]) -> list[dict]:
        """构造 system + user 对话，对齐官方 format_model_input。"""
        instruction = (item.get("instruction") or self.instruction).strip()
        if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
            instruction = instruction + "."

        content: list[dict] = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content},
        ]

        text = item.get("text")
        image = item.get("image")

        if not text and not image:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        if image:
            if isinstance(image, str):
                image_content = image if image.startswith(("http", "oss", "file://")) else "file://" + image
            else:
                image_content = image
            content.append({
                "type": "image",
                "image": image_content,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            })

        if text:
            content.append({"type": "text", "text": text})

        return conversation

    def _encode(self, items: list[dict[str, Any]]) -> list[list[float]]:
        from qwen_vl_utils.vision_process import process_vision_info

        if not items:
            return []

        conversations = [self._format_conversation(it) for it in items]
        text_prompts = self._processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                conversations,
                image_patch_size=IMAGE_PATCH_SIZE,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except Exception as exc:
            logger.debug(f"[VL-Embedding] process_vision_info 无视觉内容: {exc}")
            images, video_inputs, video_kwargs = None, None, {"do_sample_frames": False}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self._processor(
            text=text_prompts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {
            k: (v.to(self._model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state
            attn_mask = inputs["attention_mask"]
            pooled = self._last_token_pool(hidden, attn_mask)
            if self.dimensions != -1:
                pooled = pooled[:, : self.dimensions]
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled.float().cpu().tolist()

    @staticmethod
    def _last_token_pool(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """返回每个序列最后一个 attended token 的 hidden state（与官方一致）。"""
        flipped = attention_mask.flip(dims=[1])
        last_one_positions = flipped.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]


# ═══════════════════════════════════════════════════════════════════════════
# 快速测试入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    model_path = settings.local_vl_embedding_model
    print(f"加载模型: {model_path}")
    t0 = time.perf_counter()
    model = QwenVLLocalEmbeddings(model_name_or_path=model_path)
    print(f"加载耗时: {time.perf_counter() - t0:.2f}s")

    queries = ["What does this chart show?", "图表中的趋势是什么？"]
    t0 = time.perf_counter()
    vecs = [model.embed_query(q) for q in queries]
    print(f"文本向量维度: {len(vecs[0])}  耗时: {(time.perf_counter()-t0)*1000:.1f}ms")

    test_images = list(Path("./data").rglob("*.png"))[:2]
    if test_images:
        t0 = time.perf_counter()
        img_vecs = model.embed_images(test_images)
        print(f"图片向量维度: {len(img_vecs[0])}  耗时: {(time.perf_counter()-t0)*1000:.1f}ms")

        q_arr = np.array(vecs)
        i_arr = np.array(img_vecs)
        scores = (q_arr @ i_arr.T) * 100
        print("\n文本 query ↔ 图片相似度 (×100):")
        for i, q in enumerate(queries):
            for j, p in enumerate(test_images):
                print(f"  [{q[:30]}] ↔ [{p.name}]: {scores[i][j]:.2f}")
    else:
        print("（未找到测试图片，跳过图片测试）")
