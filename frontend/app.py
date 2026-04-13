"""
多模态 RAG 知识库 - Streamlit 前端
==================================
提供知识库浏览、智能问答、文档入库、系统管理四大功能模块。
后端为 FastAPI 服务，默认地址 http://localhost:8000。
"""

import json

import requests
import streamlit as st
import time
from pathlib import Path

# ==================== 全局配置 ====================

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="多模态 RAG 知识库",
    page_icon="📚",
    layout="wide",
)

# ==================== 全局 CSS ====================

_css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ==================== 工具函数 ====================


def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确认服务已启动。")
    except requests.exceptions.HTTPError as e:
        st.error(f"请求失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"请求异常：{e}")
    return None


def api_post(path: str, json_body: dict | None = None, files=None) -> dict | None:
    try:
        resp = requests.post(
            f"{API_BASE}{path}", json=json_body, files=files, timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确认服务已启动。")
    except requests.exceptions.HTTPError as e:
        st.error(f"请求失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"请求异常：{e}")
    return None


def api_post_stream(path: str, json_body: dict | None = None):
    """生成器：以 SSE 格式消费流式接口，逐个 yield 解析后的事件字典。"""
    try:
        with requests.post(
            f"{API_BASE}{path}",
            json=json_body,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                yield json.loads(data)
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确认服务已启动。")
    except requests.exceptions.HTTPError as e:
        st.error(f"请求失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"请求异常：{e}")


def api_delete(path: str) -> dict | None:
    try:
        resp = requests.delete(f"{API_BASE}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("无法连接到后端服务，请确认服务已启动。")
    except requests.exceptions.HTTPError as e:
        st.error(f"请求失败：{e.response.status_code} - {e.response.text}")
    except Exception as e:
        st.error(f"请求异常：{e}")
    return None


def truncate(text: str, max_len: int = 80) -> str:
    if not text:
        return ""
    return text[:max_len] + "…" if len(text) > max_len else text


# ==================== 侧边栏导航 ====================

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 8px 0 4px;">
            <span style="font-size:2.4rem;">📚</span>
            <div style="font-size:1.15rem; font-weight:700; margin-top:4px; color:#1a1a1a;">
                多模态 RAG 知识库
            </div>
            <div style="font-size:0.78rem; color:#888; margin-top:2px;">
                Multimodal Retrieval-Augmented Generation
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    _NAV_ICONS = {
        "智能问答": "💬",
        "知识库浏览": "📂",
        "文档入库": "📤",
        "系统管理": "⚙️",
    }
    _NAV_DESCS = {
        "智能问答": "多轮 Agent 问答",
        "知识库浏览": "浏览、检索文档块",
        "文档入库": "上传文件入库",
        "系统管理": "状态监控与维护",
    }

    page = st.radio(
        "功能导航",
        list(_NAV_ICONS.keys()),
        index=0,
        format_func=lambda x: f"{_NAV_ICONS[x]}  {x}",
    )

    st.divider()
    st.markdown(
        f"<div style='font-size:0.8rem; color:#888; padding: 0 4px;'>"
        f"{_NAV_DESCS[page]}</div>",
        unsafe_allow_html=True,
    )

# ==================== 页面 1：知识库浏览 ====================

if page == "知识库浏览":
    st.markdown("## 📂 知识库浏览")
    st.caption("浏览、查看和管理知识库中的所有文档块。")

    if "browse_page" not in st.session_state:
        st.session_state.browse_page = 0
    if "browse_page_size" not in st.session_state:
        st.session_state.browse_page_size = 20

    col_size, col_spacer = st.columns([1, 4])
    with col_size:
        page_size = st.selectbox(
            "每页显示",
            [10, 20, 50, 100],
            index=[10, 20, 50, 100].index(st.session_state.browse_page_size),
            key="page_size_selector",
        )
        if page_size != st.session_state.browse_page_size:
            st.session_state.browse_page_size = page_size
            st.session_state.browse_page = 0
            st.rerun()

    offset = st.session_state.browse_page * st.session_state.browse_page_size
    data = api_get(
        "/v1/documents",
        params={"offset": offset, "limit": st.session_state.browse_page_size},
    )

    if data is not None:
        items = data.get("items", [])
        total = data.get("total", 0)
        total_pages = max(
            1,
            (total + st.session_state.browse_page_size - 1)
            // st.session_state.browse_page_size,
        )

        # 统计栏
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("文档块总数", total)
        with c2:
            st.metric("当前页", f"{st.session_state.browse_page + 1} / {total_pages}")
        with c3:
            st.metric("本页条目", len(items))

        if "selected_ids" not in st.session_state:
            st.session_state.selected_ids = set()

        if items:
            col_batch, _ = st.columns([1, 4])
            with col_batch:
                if st.button("🗑️ 批量删除选中", type="secondary"):
                    if st.session_state.selected_ids:
                        st.session_state.confirm_batch_delete = True
                    else:
                        st.warning("请先勾选要删除的文档。")

            if st.session_state.get("confirm_batch_delete"):
                st.warning(
                    f"确定要删除选中的 **{len(st.session_state.selected_ids)}** 条文档吗？此操作不可撤销！"
                )
                c1, c2, _ = st.columns([1, 1, 5])
                with c1:
                    if st.button("确认删除", type="primary", key="batch_confirm"):
                        deleted = 0
                        failed = 0
                        progress = st.progress(0)
                        ids_to_delete = list(st.session_state.selected_ids)
                        for i, doc_id in enumerate(ids_to_delete):
                            result = api_delete(f"/v1/documents/{doc_id}")
                            if result:
                                deleted += 1
                            else:
                                failed += 1
                            progress.progress((i + 1) / len(ids_to_delete))
                        st.session_state.selected_ids.clear()
                        st.session_state.confirm_batch_delete = False
                        st.success(f"成功删除 {deleted} 条，失败 {failed} 条。")
                        time.sleep(1)
                        st.rerun()
                with c2:
                    if st.button("取消", key="batch_cancel"):
                        st.session_state.confirm_batch_delete = False
                        st.rerun()

            st.divider()

            # 文档卡片列表
            _TYPE_COLORS = {
                "text": "#dbeafe",
                "image": "#fce7f3",
                "table": "#d1fae5",
            }
            for item in items:
                doc_id = item.get("id", "")
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                source = metadata.get("source", "未知")
                page_num = metadata.get("page", "-")
                block_type = metadata.get("block_type", "-")
                badge_bg = _TYPE_COLORS.get(str(block_type).lower(), "#f3f4f6")

                col_check, col_card = st.columns([0.04, 0.96])
                with col_check:
                    checked = st.checkbox(
                        "选择",
                        value=doc_id in st.session_state.selected_ids,
                        key=f"chk_{doc_id}",
                        label_visibility="collapsed",
                    )
                    if checked:
                        st.session_state.selected_ids.add(doc_id)
                    else:
                        st.session_state.selected_ids.discard(doc_id)

                with col_card:
                    with st.expander(
                        f"📄 {source}  ·  第 {page_num} 页  ·  {block_type}  —  {truncate(content, 60)}"
                    ):
                        # 元数据徽章行
                        st.markdown(
                            f"""
                            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">
                                <span style="background:#eff6ff;color:#1d4ed8;border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:600;">
                                    🆔 {truncate(doc_id, 18)}
                                </span>
                                <span style="background:#f0fdf4;color:#166534;border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:600;">
                                    📁 {source}
                                </span>
                                <span style="background:#faf5ff;color:#6b21a8;border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:600;">
                                    📃 第 {page_num} 页
                                </span>
                                <span style="background:{badge_bg};color:#374151;border-radius:6px;padding:3px 10px;font-size:0.8rem;font-weight:600;">
                                    🏷️ {block_type}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown("**完整内容**")
                        st.text_area(
                            "content",
                            value=content,
                            height=120,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"ta_{doc_id}",
                        )
                        with st.expander("查看原始元数据"):
                            st.json(metadata)

                        if st.button("🗑️ 删除此文档", key=f"del_{doc_id}", type="secondary"):
                            st.session_state[f"confirm_del_{doc_id}"] = True

                        if st.session_state.get(f"confirm_del_{doc_id}"):
                            st.warning("确定删除？此操作不可撤销！")
                            dc1, dc2, _ = st.columns([1, 1, 5])
                            with dc1:
                                if st.button("确认", key=f"yes_del_{doc_id}", type="primary"):
                                    result = api_delete(f"/v1/documents/{doc_id}")
                                    if result:
                                        st.success("已删除。")
                                        st.session_state.selected_ids.discard(doc_id)
                                        del st.session_state[f"confirm_del_{doc_id}"]
                                        time.sleep(0.5)
                                        st.rerun()
                            with dc2:
                                if st.button("取消", key=f"no_del_{doc_id}"):
                                    del st.session_state[f"confirm_del_{doc_id}"]
                                    st.rerun()
        else:
            st.info("当前页没有文档。")

        # 分页控制
        st.divider()
        col_prev, col_page_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("⬅️ 上一页", disabled=(st.session_state.browse_page <= 0)):
                st.session_state.browse_page -= 1
                st.rerun()
        with col_page_info:
            st.markdown(
                f"<div class='pagination-info'>第 {st.session_state.browse_page + 1} / {total_pages} 页</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            if st.button(
                "下一页 ➡️",
                disabled=(st.session_state.browse_page >= total_pages - 1),
            ):
                st.session_state.browse_page += 1
                st.rerun()

# ==================== 页面 2：智能问答 ====================

elif page == "智能问答":
    st.markdown("## 💬 智能问答")
    st.caption("基于知识库进行多模态检索增强问答，支持多轮对话。")

    _TOOL_LABELS = {
        "knowledge_base_search": ("🔍", "知识库检索"),
        "knowledge_base_search_with_filter": ("📄", "按文件检索"),
        "query_rewrite": ("✏️", "查询改写"),
        "multi_round_search": ("🔄", "多轮检索"),
        "calculator": ("🔢", "数值计算"),
        "web_search": ("🌐", "网络搜索"),
        "table_analyzer": ("📊", "表格分析"),
        "image_describer": ("🖼️", "图像理解"),
        "summarizer": ("📝", "文本摘要"),
    }

    def _render_tool_steps(tool_steps: list[dict]) -> None:
        for idx, step in enumerate(tool_steps, 1):
            tool_name = step.get("tool", "")
            args = step.get("args", {})
            summary = step.get("result_summary", "")
            icon, label = _TOOL_LABELS.get(tool_name, ("🛠️", tool_name))

            st.markdown(
                f"""
                <div class="tool-step">
                    <div class="step-title">{icon} 步骤 {idx}：{label}</div>
                """,
                unsafe_allow_html=True,
            )
            if args:
                arg_md = "\n".join(
                    f"- `{k}`: {str(v)[:120]}" + ("…" if len(str(v)) > 120 else "")
                    for k, v in args.items()
                )
                st.markdown(arg_md)
            if summary:
                st.markdown(
                    f"<div class='step-summary'>↳ {summary}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    def _render_sources(sources: list[dict]) -> None:
        with st.expander(f"📎 参考来源（{len(sources)} 条）"):
            for i, src in enumerate(sources, 1):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**来源 {i}**　📁 {src.get('source', '未知')}")
                with c2:
                    st.markdown(f"📃 第 {src.get('page', '-')} 页")
                with c3:
                    st.markdown(f"🏷️ {src.get('block_type', '-')}")
                content = src.get("content", "")
                if content:
                    st.text_area(
                        "chunk内容",
                        value=content,
                        height=120,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"src_content_{i}_{hash(content)}",
                    )
                if i < len(sources):
                    st.divider()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []

    # 顶部操作栏
    top_col1, top_col2 = st.columns([6, 1])
    with top_col2:
        if st.session_state.chat_display:
            if st.button("🔄 清空对话", use_container_width=True):
                st.session_state.chat_history.clear()
                st.session_state.chat_display.clear()
                st.rerun()

    # 聊天历史
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("tool_steps"):
                with st.expander(
                    f"🧠 思考过程（{len(msg['tool_steps'])} 步工具调用）",
                    expanded=False,
                ):
                    _render_tool_steps(msg["tool_steps"])
            st.markdown(msg["content"])
            if msg.get("sources"):
                _render_sources(msg["sources"])

    # 输入框
    question = st.chat_input("请输入您的问题，按 Enter 发送…")

    if question:
        st.session_state.chat_display.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # 实时渲染工具步骤
            tool_placeholders: dict[int, object] = {}   # step_idx -> st.empty()
            tool_steps_data: dict[int, dict] = {}        # step_idx -> merged step dict
            answer = ""
            sources: list[dict] = []
            total_ms = 0
            had_error = False

            with st.status("🤖 Agent 正在思考…", expanded=True) as status_box:
                for event in api_post_stream(
                    "/v1/query/stream",
                    json_body={
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                    },
                ):
                    etype = event.get("type")

                    if etype == "tool_start":
                        idx = event["step_idx"]
                        tool = event["tool"]
                        icon, label = _TOOL_LABELS.get(tool, ("🛠️", tool))
                        args = event.get("args", {})
                        arg_str = "、".join(
                            f"`{k}`: {str(v)[:60]}" for k, v in args.items()
                        )
                        ph = st.empty()
                        ph.markdown(
                            f"⏳ **步骤 {idx + 1}**　{icon} {label}（运行中…）"
                            + (f"\n\n{arg_str}" if arg_str else "")
                        )
                        tool_placeholders[idx] = ph
                        tool_steps_data[idx] = {
                            "tool": tool,
                            "args": args,
                            "result_summary": "",
                            "status": "running",
                            "elapsed_ms": None,
                        }

                    elif etype == "tool_done":
                        idx = event["step_idx"]
                        if idx in tool_placeholders:
                            tool = event["tool"]
                            icon, label = _TOOL_LABELS.get(tool, ("🛠️", tool))
                            elapsed = event.get("elapsed_ms", 0)
                            summary = event.get("result_summary", "")
                            args = tool_steps_data.get(idx, {}).get("args", {})
                            arg_str = "、".join(
                                f"`{k}`: {str(v)[:60]}" for k, v in args.items()
                            )
                            tool_placeholders[idx].markdown(
                                f"✅ **步骤 {idx + 1}**　{icon} {label}（{elapsed}ms）"
                                + (f"\n\n{arg_str}" if arg_str else "")
                                + (f"\n\n> {summary}" if summary else "")
                            )
                            if idx in tool_steps_data:
                                tool_steps_data[idx].update({
                                    "result_summary": summary,
                                    "status": "done",
                                    "elapsed_ms": elapsed,
                                })

                    elif etype == "answer":
                        answer = event.get("answer", "")
                        sources = event.get("sources", [])
                        total_ms = event.get("total_elapsed_ms", 0)
                        tool_count = event.get("tool_count", len(tool_steps_data))
                        status_box.update(
                            label=f"✅ 思考完成（{tool_count} 步工具调用，{total_ms}ms）",
                            state="complete",
                            expanded=False,
                        )

                    elif etype == "error":
                        had_error = True
                        status_box.update(label="❌ 请求失败", state="error", expanded=False)

            if had_error or not answer:
                fallback = answer or "抱歉，请求出现错误，请稍后重试。"
                st.markdown(fallback)
                st.session_state.chat_display.append(
                    {"role": "assistant", "content": fallback}
                )
            else:
                st.markdown(answer)
                if sources:
                    _render_sources(sources)

                tool_steps_list = [
                    tool_steps_data[i] for i in sorted(tool_steps_data.keys())
                ]
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.chat_display.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "tool_steps": tool_steps_list,
                    }
                )

# ==================== 页面 3：文档入库 ====================

elif page == "文档入库":
    st.markdown("## 📤 文档入库")
    st.caption("上传文件到知识库，支持 PDF、TXT、Markdown、PNG、JPG 格式。")

    # 支持格式说明
    fmt_cols = st.columns(5)
    _FMT_INFO = [
        ("📄", "PDF", "#dbeafe"),
        ("📝", "TXT / MD", "#d1fae5"),
        ("🖼️", "PNG", "#fce7f3"),
        ("📷", "JPG / JPEG", "#fef3c7"),
        ("🗂️", "多文件批量", "#ede9fe"),
    ]
    for col, (icon, label, bg) in zip(fmt_cols, _FMT_INFO):
        with col:
            st.markdown(
                f"""<div style="background:{bg};border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:1.5rem;">{icon}</div>
                    <div style="font-size:0.82rem;font-weight:600;margin-top:4px;">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "拖拽文件到此处，或点击选择文件（可多选）",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"**已选择 {len(uploaded_files)} 个文件：**")
        file_cols = st.columns(min(len(uploaded_files), 4))
        _EXT_ICONS = {
            "pdf": "📄", "txt": "📝", "md": "📝",
            "png": "🖼️", "jpg": "📷", "jpeg": "📷",
        }
        for i, f in enumerate(uploaded_files):
            ext = f.name.rsplit(".", 1)[-1].lower()
            icon = _EXT_ICONS.get(ext, "📁")
            size_kb = len(f.getvalue()) / 1024
            with file_cols[i % 4]:
                st.markdown(
                    f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                                   padding:10px 12px;margin-bottom:8px;">
                        <div style="font-size:1.3rem;">{icon}</div>
                        <div style="font-size:0.85rem;font-weight:600;word-break:break-all;">{f.name}</div>
                        <div style="font-size:0.75rem;color:#888;">{size_kb:.1f} KB</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 开始上传并入库", type="primary", use_container_width=False):
            files_payload = [
                ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                for f in uploaded_files
            ]

            progress_bar = st.progress(0, text="准备上传…")
            try:
                progress_bar.progress(20, text="正在上传文件…")
                resp = requests.post(
                    f"{API_BASE}/v1/ingest/upload",
                    files=files_payload,
                    timeout=300,
                )
                progress_bar.progress(80, text="正在处理文件…")
                resp.raise_for_status()
                result = resp.json()
                progress_bar.progress(100, text="处理完成！")

                st.success("文件上传并入库成功！")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(
                        f"""<div class="metric-card" style="background:linear-gradient(135deg,#3b82f6,#6366f1);">
                            <div class="metric-label">处理文件数</div>
                            <div class="metric-value">{result.get("files_processed", "-")}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f"""<div class="metric-card" style="background:linear-gradient(135deg,#10b981,#059669);">
                            <div class="metric-label">入库文档数</div>
                            <div class="metric-value">{result.get("documents_ingested", "-")}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        f"""<div class="metric-card" style="background:linear-gradient(135deg,#f59e0b,#d97706);">
                            <div class="metric-label">生成块数</div>
                            <div class="metric-value">{result.get("chunks_created", "-")}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            except requests.exceptions.ConnectionError:
                progress_bar.empty()
                st.error("无法连接到后端服务，请确认服务已启动。")
            except requests.exceptions.HTTPError as e:
                progress_bar.empty()
                st.error(f"上传失败：{e.response.status_code} - {e.response.text}")
            except Exception as e:
                progress_bar.empty()
                st.error(f"上传异常：{e}")

# ==================== 页面 4：系统管理 ====================

elif page == "系统管理":
    st.markdown("## ⚙️ 系统管理")
    st.caption("查看系统状态、监控指标和管理知识库集合。")

    st.subheader("系统状态")
    health = api_get("/health")

    if health:
        status_val = health.get("status", "unknown")
        is_healthy = status_val in ("healthy", "ok")
        badge_html = (
            f'<span class="status-healthy">● {status_val}</span>'
            if is_healthy
            else f'<span class="status-error">● {status_val}</span>'
        )

        st.markdown(f"服务状态：{badge_html}", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        _CARD_STYLES = [
            "linear-gradient(135deg,#3b82f6,#6366f1)",
            "linear-gradient(135deg,#10b981,#059669)",
            "linear-gradient(135deg,#8b5cf6,#7c3aed)",
            "linear-gradient(135deg,#f59e0b,#d97706)",
        ]
        _CARD_DATA = [
            ("文档总数", health.get("document_count", "-"), "📦"),
            ("模型", health.get("model", "-"), "🤖"),
            ("集合名称", health.get("collection", "-"), "🗄️"),
            ("版本", health.get("version", "v1.0"), "🏷️"),
        ]
        for col, style, (label, value, icon) in zip(
            [mc1, mc2, mc3, mc4], _CARD_STYLES, _CARD_DATA
        ):
            with col:
                st.markdown(
                    f"""<div class="metric-card" style="background:{style};">
                        <div style="font-size:1.4rem;">{icon}</div>
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
    else:
        st.error("无法获取系统状态，请检查后端服务是否正常运行。")

    st.divider()

    # 危险操作区
    st.subheader("⚠️ 危险操作")
    st.markdown(
        """<div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;padding:14px 18px;color:#92400e;">
            <strong>警告：</strong>以下操作将永久删除知识库中的所有数据，请在充分确认后再执行。
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if st.button("🗑️ 重置知识库集合", type="secondary"):
        st.session_state.confirm_reset = True

    if st.session_state.confirm_reset:
        st.error("⚠️ 此操作将永久删除所有文档数据，不可恢复！请再次确认。")
        c1, c2, _ = st.columns([1, 1, 5])
        with c1:
            if st.button("确认重置", type="primary", key="reset_yes"):
                with st.spinner("正在重置集合…"):
                    result = api_delete("/v1/collection")
                if result:
                    st.success(f"集合已重置：{result.get('collection', '')}")
                    st.session_state.confirm_reset = False
                    time.sleep(1)
                    st.rerun()
                else:
                    st.session_state.confirm_reset = False
        with c2:
            if st.button("取消", key="reset_no"):
                st.session_state.confirm_reset = False
                st.rerun()
