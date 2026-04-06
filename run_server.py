"""
FastAPI 服务器启动入口。

用法
────
  # 开发模式（热重载）
  python run_server.py

  # 生产模式
  python run_server.py --host 0.0.0.0 --port 8000 --workers 4
"""
import argparse
import sys

import uvicorn

from config.settings import get_settings

settings = get_settings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动多模态 RAG API 服务器。")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true", help="启用热重载（开发模式）。")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
