'''读取本地 TXT、Markdown、PDF 和 Word 文档，并切分文字。'''

from pathlib import Path

# 资料文件夹位于本文件所在的项目目录中。
DOCUMENTS_DIR = Path(__file__).resolve().parent / "资料"
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def read_pdf(path):
    '''按页读取 PDF；返回有文字的页面，以及没有文字的页码。'''
    import pymupdf  # PyMuPDF 提供 PDF 的打开和文字提取功能。

    sections = []
    empty_pages = []
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError("PDF 需要密码")

        for index, page in enumerate(pdf):
            # 页码从 1 开始，与 PDF 阅读器中的页码一致。
            page_number = index + 1
            text = page.get_text("text")
            if text.strip():
                sections.append({"page": page_number, "text": text})
            else:
                empty_pages.append(page_number)

    return sections, empty_pages


def read_word(path):
    '''按文档顺序读取 Word 正文段落和表格文字。'''
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    word = Document(path)
    lines = []
    # iter_inner_content 按原顺序返回正文段落和表格。
    for block in word.iter_inner_content():
        if isinstance(block, Paragraph):
            lines.append(block.text)
        elif isinstance(block, Table):
            for row in block.rows:
                # 同一行的单元格用竖线分隔，保留表格中的文字。
                lines.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(lines)


def read_documents():
    #读取支持的文件，返回成功读取的资料和逐文件提示。
    documents = []
    warnings = []

    for path in sorted(DOCUMENTS_DIR.rglob("*")):
        # Word 和 WPS 打开文档时会创建临时锁文件，不是可读取的正文。
        if not path.is_file() or path.name.startswith(("~$", ".~")):
            continue

        source = path.relative_to(DOCUMENTS_DIR).as_posix()
        suffix = path.suffix.lower()

        if suffix == ".doc":
            warnings.append({"source": source, "message": "旧版 .doc 暂不支持，请另存为 .docx。"})
            continue

        if suffix not in SUPPORTED_SUFFIXES:
            continue

        try:
            if suffix == ".pdf":
                sections, empty_pages = read_pdf(path)
                if empty_pages:
                    warnings.append({
                        "source": source,
                        "message": f"{len(empty_pages)} 页没有可提取文字；如果是扫描图片，请先 OCR。",
                    })
            else:
                if suffix == ".docx":
                    text = read_word(path)
                else:
                    # UTF-8-SIG 同时支持普通 UTF-8 和带 BOM 的文本。
                    text = path.read_text(encoding="utf-8-sig")
                sections = [{"page": None, "text": text}] if text.strip() else []

            if sections:
                documents.append({"source": source, "sections": sections})
            elif suffix != ".pdf" or not empty_pages:
                warnings.append({"source": source, "message": "没有可提取的正文文字。"})
        except UnicodeDecodeError:
            warnings.append({"source": source, "message": "文本不是 UTF-8 编码，请另存为 UTF-8。"})
        except ValueError as error:
            # 不显示文件内容或原始错误，避免在网页暴露文档信息。
            message = "PDF 需要密码，请先解除密码保护。" if str(error) == "PDF 需要密码" else "文件无法读取，请检查格式。"
            warnings.append({"source": source, "message": message})
        except Exception as error:
            # 单个文件出错时继续读取其他文件。
            warnings.append({"source": source, "message": f"读取失败（{type(error).__name__}），请检查文件是否完整。"})
    return documents, warnings


def split_documents(documents, chunk_size=300, chunk_overlap=50):
    '''按字符数切分文字，PDF 的片段不会跨页，并保留页码。'''
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("片段长度必须大于 0，重叠长度必须大于等于 0 且小于片段长度。")

    chunks = []
    for document in documents:
        number = 1
        for section in document["sections"]:
            text = section["text"]
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append({
                    "source": document["source"],
                    "page": section["page"],
                    "number": number,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                })
                number += 1
                if end == len(text):
                    break
                start = end - chunk_overlap
    return chunks


def render_document_preview():
    '''展示本地资料读取、切分和分页预览界面。'''
    # 只有显示网页时才导入 Streamlit；读取和切分函数仍可单独使用。
    import streamlit as st

    with st.expander("📄 学习：读取并切分本地资料"):
        st.caption("把 TXT、Markdown、PDF 或 Word（.docx）文件放入“资料”文件夹。此处仅预览；要参与问答需建立或更新索引。")

        if st.button("读取并切分资料"):
            # 结果保存在当前网页会话中；更新资料后清除旧的检索结果。
            st.session_state.pop("retrieval_demo_result", None)
            documents, warnings = read_documents()
            chunks = split_documents(documents)
            st.session_state["document_preview"] = {
                "sources": [document["source"] for document in documents],
                "chunks": chunks,
                "warnings": warnings,
            }
            # 重新读取后从第一页开始，避免旧页码超过新的范围。
            for key in list(st.session_state):
                if key.startswith("chunk_page::"):
                    del st.session_state[key]
            st.session_state["preview_source"] = "全部资料"

        preview = st.session_state.get("document_preview")
        if preview is not None:
            for warning in preview["warnings"]:
                st.warning(f"{warning['source']}：{warning['message']}")
            if not preview["sources"]:
                st.info("没有读取到包含文字的支持格式文件。")
            else:
                st.success(f"已读取 {len(preview['sources'])} 个文件，切分为 {len(preview['chunks'])} 个片段。")
                selected = st.selectbox("选择要预览的资料", ["全部资料"] + preview["sources"], key="preview_source")
                shown = [chunk for chunk in preview["chunks"] if selected == "全部资料" or chunk["source"] == selected]
                page_size = 10
                page_count = max(1, (len(shown) + page_size - 1) // page_size)
                page = st.number_input("预览页码", min_value=1, max_value=page_count, value=1, step=1, key=f"chunk_page::{selected}")
                st.caption(f"共 {len(shown)} 个片段，每页最多 {page_size} 个，共 {page_count} 页。字符位置从当前文件或 PDF 页面开头计算。")
                for chunk in shown[(page - 1) * page_size:page * page_size]:
                    page_label = f" · PDF 第 {chunk['page']} 页" if chunk["page"] is not None else ""
                    st.text(f"文件：{chunk['source']}{page_label} · 第 {chunk['number']} 段 · 字符 {chunk['start']}–{chunk['end']}（结束位置不包含）")
                    st.text(chunk["text"])
