"""直接诊断：文本 query 与图片的余弦相似度是否合理（不走 chroma）。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import numpy as np
import torch
from src.embeddings.qwen_vl_embedding import QwenVLLocalEmbeddings

model = QwenVLLocalEmbeddings()

# 找几张图做测试
imgs = sorted(Path("data/ocr_output").rglob("*.jpg"))[:5]
print(f"使用图片: {[p.name for p in imgs]}\n")

queries = [
    "qwen framework diagram",
    "A chart showing model architecture",
    "random unrelated text about cooking recipes",
]

q_vecs = np.array([model.embed_query(q) for q in queries])
i_vecs = np.array(model.embed_images(imgs))

print(f"query shape: {q_vecs.shape}, image shape: {i_vecs.shape}")
print(f"query norm: {np.linalg.norm(q_vecs, axis=1)}")
print(f"image norm: {np.linalg.norm(i_vecs, axis=1)}")

sims = q_vecs @ i_vecs.T
print("\n文本 ↔ 图片 余弦相似度：")
for i, q in enumerate(queries):
    print(f"\n[{q}]")
    for j, p in enumerate(imgs):
        print(f"  {p.name[:20]}  {sims[i][j]:+.4f}")

# 同源：文本↔文本 的相似度
tt = q_vecs @ q_vecs.T
print(f"\n文本↔文本对角: {np.diag(tt)}（应为 1.0）")
print(f"文本↔文本非对角平均: {(tt.sum() - np.trace(tt)) / (tt.size - tt.shape[0]):.4f}")

# 图↔图
ii = i_vecs @ i_vecs.T
print(f"图↔图对角: {np.diag(ii)}")
