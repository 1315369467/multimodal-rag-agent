"""
多模态 RAG 知识库 - Streamlit 前端
==================================
提供知识库浏览、智能问答、文档入库、系统管理四大功能模块。
后端为 FastAPI 服务，默认地址 http://localhost:8000。
"""

import streamlit as st
import requests
import time

# ==================== 全局配置 ====================

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="多模态 RAG 知识库",
    page_icon="📚",
    layout="wide",
)

# ==================== 工具函数 ====================


def api_get(path: str, params: dict | None = None) -> dict | None:
    """发送 GET 请求到后端，返回 JSON 或在出错时显示错误"""
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
    """发送 POST 请求到后端"""
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


def api_delete(path: str) -> dict | None:
    """发送 DELETE 请求到后端"""
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
    """截断文本并添加省略号"""
    if not text:
        return ""
    return text[:max_len] + "…" if len(text) > max_len else text


# ==================== 侧边栏导航 ====================

with st.sidebar:
    st.title("📚 多模态 RAG 知识库")
    st.divider()
    page = st.radio(
        "功能导航",
        ["知识库浏览", "智能问答", "文档入库", "系统管理"],
        index=0,
    )

# ==================== 页面 1：知识库浏览 ====================

if page == "知识库浏览":
    st.header("📂 知识库浏览")
    st.caption("浏览、查看和管理知识库中的所有文档块。")

    # 初始化分页状态
    if "browse_page" not in st.session_state:
        st.session_state.browse_page = 0
    if "browse_page_size" not in st.session_state:
        st.session_state.browse_page_size = 20

    # 分页大小选择
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
    data = api_get("/v1/documents", params={"offset": offset, "limit": st.session_state.browse_page_size})

    if data is not None:
        items = data.get("items", [])
        total = data.get("total", 0)
        total_pages = max(1, (total + st.session_state.browse_page_size - 1) // st.session_state.browse_page_size)

        st.info(f"共 **{total}** 条文档块，当前第 **{st.session_state.browse_page + 1}** / **{total_pages}** 页")

        # 批量删除：用 session_state 存储选中的文档 ID
        if "selected_ids" not in st.session_state:
            st.session_state.selected_ids = set()

        if items:
            # 批量删除按钮
            col_batch, _ = st.columns([1, 4])
            with col_batch:
                if st.button("🗑️ 批量删除选中", type="secondary"):
                    if st.session_state.selected_ids:
                        st.session_state.confirm_batch_delete = True
                    else:
                        st.warning("请先勾选要删除的文档。")

            # 批量删除确认
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

            # 文档列表
            for item in items:
                doc_id = item.get("id", "")
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                source = metadata.get("source", "未知")
                page_num = metadata.get("page", "-")
                block_type = metadata.get("block_type", "-")

                col_check, col_info = st.columns([0.05, 0.95])
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

                with col_info:
                    with st.expander(
                        f"**ID:** `{truncate(doc_id, 16)}` | **来源:** {source} | **页码:** {page_num} | **类型:** {block_type} | {truncate(content, 60)}"
                    ):
                        st.markdown("**完整内容：**")
                        st.text(content)
                        st.markdown(f"**元数据：** `{metadata}`")

                        # 单条删除
                        if st.button("删除此文档", key=f"del_{doc_id}", type="secondary"):
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
                f"<div style='text-align:center; padding-top:8px;'>第 {st.session_state.browse_page + 1} / {total_pages} 页</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            if st.button("下一页 ➡️", disabled=(st.session_state.browse_page >= total_pages - 1)):
                st.session_state.browse_page += 1
                st.rerun()

# ==================== 页面 2：智能问答 ====================

elif page == "智能问答":
    st.header("💬 智能问答")
    st.caption("基于知识库进行多模态检索增强问答。")

    # 初始化聊天历史
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [{role, content}]
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []  # [{role, content, sources}]

    # 显示历史消息
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 参考来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**来源 {i}：** {src.get('source', '未知')} "
                            f"(页码: {src.get('page', '-')}, "
                            f"类型: {src.get('block_type', '-')})"
                        )
                        st.divider()

    # 输入区域
    question = st.chat_input("请输入您的问题…")

    if question:
        # 显示用户消息
        st.session_state.chat_display.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # 调用后端
        with st.chat_message("assistant"):
            with st.spinner("正在思考中…"):
                result = api_post(
                    "/v1/query",
                    json_body={
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                    },
                )

            if result:
                answer = result.get("answer", "未能获取回答。")
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📎 参考来源"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f"**来源 {i}：** {src.get('source', '未知')} "
                                f"(页码: {src.get('page', '-')}, "
                                f"类型: {src.get('block_type', '-')})"
                            )
                            st.divider()

                # 更新聊天历史（用于后端上下文）
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # 更新显示历史
                st.session_state.chat_display.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            else:
                fallback = "抱歉，请求出现错误，请稍后重试。"
                st.markdown(fallback)
                st.session_state.chat_display.append(
                    {"role": "assistant", "content": fallback}
                )

    # 清空历史按钮
    if st.session_state.chat_display:
        if st.button("🔄 清空对话历史"):
            st.session_state.chat_history.clear()
            st.session_state.chat_display.clear()
            st.rerun()

# ==================== 页面 3：文档入库 ====================

elif page == "文档入库":
    st.header("📤 文档入库")
    st.caption("上传文件到知识库，支持 PDF、TXT、Markdown、PNG、JPG 格式。")

    uploaded_files = st.file_uploader(
        "选择文件（可多选）",
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info(f"已选择 **{len(uploaded_files)}** 个文件：{', '.join(f.name for f in uploaded_files)}")

        if st.button("🚀 开始上传并入库", type="primary"):
            # 构建 multipart 文件列表
            files_payload = []
            for f in uploaded_files:
                files_payload.append(("files", (f.name, f.getvalue(), f.type or "application/octet-stream")))

            with st.spinner("正在上传和处理文件，请稍候…"):
                progress_bar = st.progress(0, text="上传中…")
                try:
                    resp = requests.post(
                        f"{API_BASE}/v1/ingest/upload",
                        files=files_payload,
                        timeout=300,
                    )
                    progress_bar.progress(100, text="处理完成")
                    resp.raise_for_status()
                    result = resp.json()

                    st.success("文件上传并入库成功！")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("处理文件数", result.get("files_processed", "-"))
                    with col2:
                        st.metric("文档数", result.get("documents_ingested", "-"))
                    with col3:
                        st.metric("生成块数", result.get("chunks_created", "-"))

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
    st.header("⚙️ 系统管理")
    st.caption("查看系统状态和管理知识库集合。")

    # 健康检查
    st.subheader("系统状态")
    health = api_get("/health")

    if health:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = health.get("status", "unknown")
            if status == "healthy" or status == "ok":
                st.success(f"服务状态：{status}")
            else:
                st.warning(f"服务状态：{status}")
        with col2:
            st.metric("文档总数", health.get("document_count", "-"))
        with col3:
            st.info(f"模型：{health.get('model', '-')}")
        with col4:
            st.info(f"集合：{health.get('collection', '-')}")
    else:
        st.error("无法获取系统状态。")

    st.divider()

    # 重置集合（危险操作）
    st.subheader("危险操作")
    st.warning("以下操作将删除知识库中的所有数据，请谨慎操作！")

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
