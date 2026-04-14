# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import os
import tempfile
import time
from pathlib import Path

import httpx

from mineru.cli import api_client as _api_client
from mineru.cli.common import image_suffixes, office_suffixes, pdf_suffixes
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

# 支持的输入文件类型：PDF、图片、Office 文档（合并三个列表并去重）
SUPPORTED_INPUT_SUFFIXES = set(pdf_suffixes + image_suffixes + office_suffixes)


def collect_input_files(input_path: str | Path) -> list[Path]:
    """收集输入路径下所有支持格式的文件，输入可以是单个文件或目录。"""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    # 如果是单个文件，直接校验格式后返回
    if path.is_file():
        file_suffix = guess_suffix_by_path(path)
        if file_suffix not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {path.name}")
        return [path]

    if not path.is_dir():
        raise ValueError(f"Input path must be a file or directory: {path}")

    # 如果是目录，扫描其中所有支持格式的文件并按文件名排序
    input_files = sorted(
        (
            candidate.resolve()
            for candidate in path.iterdir()
            if candidate.is_file()
            and guess_suffix_by_path(candidate) in SUPPORTED_INPUT_SUFFIXES
        ),
        key=lambda item: item.name,
    )
    if not input_files:
        raise ValueError(f"No supported files found in directory: {path}")
    return input_files


def build_form_data(
    language: str,
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: str | None,
    start_page_id: int,
    end_page_id: int | None,
) -> dict[str, str | list[str]]:
    """构造提交解析任务时的表单参数。"""
    return _api_client.build_parse_request_form_data(
        lang_list=[language],
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        return_md=True,              # 返回 Markdown 格式结果
        return_middle_json=False,    # 不返回中间 JSON
        return_model_output=False,   # 不返回模型原始输出
        return_content_list=False,   # 不返回内容列表
        return_images=True,          # 返回提取出的图片
        response_format_zip=True,    # 结果打包为 zip
        return_original_file=False,  # 不返回原始文件
    )


def format_status_message(status_snapshot: _api_client.TaskStatusSnapshot) -> str:
    """将任务状态快照格式化为可读字符串，若有排队信息则一并显示。"""
    if status_snapshot.queued_ahead is None:
        return status_snapshot.status
    return f"{status_snapshot.status} (queued_ahead={status_snapshot.queued_ahead})"


def prepare_local_api_temp_dir() -> None:
    """WSL 环境下修正临时目录路径。

    vLLM/ZeroMQ 的 IPC socket 在 WSL 的 drvfs 挂载目录（/mnt/...）下会失败，
    需要将临时目录切换到原生 Linux 的 /tmp。
    """
    current_temp_dir = Path(tempfile.gettempdir())
    if os.name == "nt" or not Path("/tmp").exists():
        return
    if not str(current_temp_dir).startswith("/mnt/"):
        return

    # vLLM/ZeroMQ IPC sockets fail on drvfs-backed temp directories under WSL.
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = None


async def run_demo(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    api_url: str | None = None,
    backend: str = "hybrid-auto-engine",
    parse_method: str = "auto",
    language: str = "ch",
    formula_enable: bool = True,
    table_enable: bool = True,
    server_url: str | None = None,
    start_page_id: int = 0,
    end_page_id: int | None = None,
) -> None:
    """主流程：收集文件 → 启动/连接 API → 提交任务 → 轮询状态 → 下载并解压结果。"""
    api_url = api_url or None
    server_url = server_url or None
    # http-client 类后端需要指定远程服务地址
    if backend.endswith("http-client") and not server_url:
        raise ValueError(f"backend={backend} requires server_url")

    input_files = collect_input_files(input_path)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # 构造解析参数表单
    form_data = build_form_data(
        language=language,
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )
    # 将每个文件封装为上传资产对象
    upload_assets = [
        _api_client.UploadAsset(path=file_path, upload_name=file_path.name)
        for file_path in input_files
    ]

    local_server: _api_client.LocalAPIServer | None = None
    result_zip_path: Path | None = None
    task_label = f"{len(input_files)} file(s)"

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        try:
            if api_url is None:
                # 未指定远程 API 时，自动在本地启动 mineru-api 子进程
                prepare_local_api_temp_dir()
                local_server = _api_client.LocalAPIServer()
                base_url = local_server.start()
                print(f"Started local mineru-api: {base_url}")
                # 等待本地服务健康检查通过
                server_health = await _api_client.wait_for_local_api_ready(
                    http_client,
                    local_server,
                )
            else:
                # 使用已有的远程 API，获取服务信息
                server_health = await _api_client.fetch_server_health(
                    http_client,
                    _api_client.normalize_base_url(api_url),
                )

            print(f"Using API: {server_health.base_url}")
            print(f"Submitting {len(upload_assets)} file(s)")
            # 上传文件并提交解析任务，返回任务 ID 等信息
            submit_response = await _api_client.submit_parse_task(
                base_url=server_health.base_url,
                upload_assets=upload_assets,
                form_data=form_data,
            )
            print(f"task_id: {submit_response.task_id}")
            if submit_response.queued_ahead is not None:
                print(f"status: pending (queued_ahead={submit_response.queued_ahead})")

            last_status_message = None

            def on_status_update(status_snapshot: _api_client.TaskStatusSnapshot) -> None:
                # 状态变化时才打印，避免重复输出相同状态
                nonlocal last_status_message
                message = format_status_message(status_snapshot)
                if message == last_status_message:
                    return
                last_status_message = message
                print(f"status: {message}")

            # 轮询任务状态，直到完成或失败
            await _api_client.wait_for_task_result(
                client=http_client,
                submit_response=submit_response,
                task_label=task_label,
                status_snapshot_callback=on_status_update,
            )
            print("status: completed")
            # 下载结果 zip 到临时文件
            result_zip_path = await _api_client.download_result_zip(
                client=http_client,
                submit_response=submit_response,
                task_label=task_label,
            )
        finally:
            # 无论成功还是失败，都停止本地服务（如果是本地启动的）
            if local_server is not None:
                local_server.stop()

    assert result_zip_path is not None
    try:
        # 将结果 zip 解压到输出目录
        _api_client.safe_extract_zip(result_zip_path, output_path)
    finally:
        # 清理临时 zip 文件
        result_zip_path.unlink(missing_ok=True)
    print(f"Extracted result to: {output_path}")


def main() -> None:
    # demo_dir = Path(__file__).resolve().parent
    demo_dir = Path("data")

    # 输入可以是单个支持格式的文件，也可以是包含多个文件的目录
    input_path = demo_dir / "pdfs"
    # 解析结果将解压到此目录
    output_dir = demo_dir / "ocr_output"
    # 指定已有的 MinerU FastAPI 服务地址，例如：
    # "http://127.0.0.1:8000"
    # 设为 None 时会自动在本地启动临时 mineru-api 服务
    api_url = None

    # 可用的解析后端：
    # "hybrid-auto-engine"   -> 本地混合解析，推荐默认
    # "pipeline"             -> 通用 OCR/文本流水线
    # "vlm-auto-engine"      -> 本地 VLM 解析
    # "vlm-http-client"      -> 远程 OpenAI 兼容 VLM 服务
    # "hybrid-http-client"   -> 远程 OpenAI 兼容混合服务
    backend = "hybrid-auto-engine"
    # 解析方式：
    # "auto" -> 自动选择文本提取或 OCR
    # "txt"  -> 强制文本提取
    # "ocr"  -> 强制 OCR
    parse_method = "auto"
    # OCR 语言提示，主要用于 pipeline 和 hybrid 后端
    language = "ch"
    # 是否启用公式解析
    formula_enable = True
    # 是否启用表格解析
    table_enable = True
    # 仅 "*-http-client" 后端需要，例如：
    # "http://127.0.0.1:30000"
    server_url = None
    # 页码范围（从 0 开始），end_page_id 为 None 表示解析到最后一页
    start_page_id = 0
    end_page_id = None

    """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    _start_time = time.time()
    asyncio.run(
        run_demo(
            input_path=input_path,
            output_dir=output_dir,
            api_url=api_url,
            backend=backend,
            parse_method=parse_method,
            language=language,
            formula_enable=formula_enable,
            table_enable=table_enable,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    )
    print(f"总耗时: {time.time() - _start_time:.1f}s")


if __name__ == "__main__":
    main()
