# 本地知识库助手 · 学习版

这是一个逐步学习 RAG（检索增强生成）的 Python 项目：使用 Streamlit 构建网页，在本机读取资料、生成文本向量并通过 Chroma 检索，再把参考片段交给 DeepSeek 生成回答。

**入口是 `app.py`。** 项目使用本地嵌入模型，但 DeepSeek 回答通过远程 API 生成；知识库问答会将检索到的片段、来源、本次问题和此前对话发送给 DeepSeek。

## 已实现的功能

- **普通对话**：使用 DeepSeek API，在网页中提问并显示回答、返回模型和 Token 用量。
- **知识库问答**：选择本地 Chroma 知识库，用当前问题检索相关片段，将片段作为参考资料提供给 DeepSeek；回答下可查看片段来源和相似度。
- **资料上传与更新**：支持 TXT、Markdown（`.md`、`.markdown`）、PDF 和 DOCX；上传后保存到 `资料/上传/`，并可重新读取全部资料、创建新的知识库快照。
- **启动检查本地资料**：每次重启服务后首次打开网页，自动比较 `资料/` 及其子目录与上次检查的文件清单，显示新增、修改、删除的文件；运行期间也可以点击“重新检查资料文件”。
- **本地对话历史**：每个新对话在首次提问时创建 JSON 文件；下拉框可选择并继续已有对话，回答、模型名称、用量和参考片段会随历史保存。
- **旧索引迁移**：可将兼容的旧 JSON 索引导入 Chroma，复用已有向量；旧文件保留，同一文件和内容重复导入会复用已导入集合。

## 项目架构

```text
浏览器
  │ 访问本地 Streamlit 网页
  ▼
app.py：页面入口、会话选择、模式选择和 DeepSeek 请求
  ├─ chat_ui.py：显示聊天消息、Token 用量及参考片段
  ├─ upload_ui.py：文件上传、重建按钮和进度显示
  │    ├─ document_monitor_ui.py：启动检查、手动检查和变化提示
  │    │    └─ document_monitor.py：扫描资料文件信息、比较并保存检查清单
  │    └─ knowledge_upload.py：保存上传文件、组织知识库更新
  │         ├─ document_reader.py：读取资料并切分文本
  │         └─ chroma_index.py：生成并保存 Chroma 知识库
  ├─ rag_demo.py：检索参考资料，组装问题、历史和参考片段
  │    └─ chroma_index.py：查询本地 Chroma
  │         └─ embedding_demo.py：本地加载模型并生成向量
  └─ history.py：创建、读取和保存 Histor 中的对话
```

`app.py` 是 Streamlit 脚本。用户交互会触发脚本重新执行；`st.session_state` 用于保留当前网页会话的对话和选择，长期保存则由本地文件负责。

### 更新知识库的数据流

```text
网页上传文件，或手动放入资料/ 后点击重新读取
    ↓
读取资料/ 下所有支持的文件，按默认 300 字符切分、重叠 50 字符
    ↓
本地 FastEmbed 模型为全部片段生成向量
    ↓
本地 Chroma 创建新的集合并保存原文、向量及来源位置
    ↓
写入完成且数量核对通过后，网页自动选中新集合
```

每次更新是**全量重建一份新快照**：会重新读取所有资料并重新生成片段向量，不会覆盖原集合。文件可能已经上传成功但建库失败，此时修复问题后可点击“重新读取资料并更新知识库”。启动检查会提示文件变化；要让变化用于问答，仍需点击该更新按钮。

### 启动时检查手动加入的资料

Streamlit 在浏览器连接后才执行页面脚本，因此检查发生在**每次启动服务后首次打开网页时**。检查会递归扫描 `资料/`（包括 `RRAM/`、`上传/` 等子目录），支持 TXT、Markdown、PDF、DOCX，并忽略 Office 临时锁文件及不支持的格式。

- 第一次启用时，在 `.local/document_inventory.json` 建立现有文件的检查基准；因为没有以前的清单，首次不会把所有旧资料标为新增。
- 后续检查按相对路径识别新增、删除，并按文件大小和修改时间识别修改；保持大小和修改时间不变的内容替换不能被识别为修改。
- 同一服务进程通过 `st.cache_resource` 保存检查结果，聊天、切换历史和新增浏览器会话都不会重复执行启动检查。重启服务后会再次扫描。
- 服务运行期间手动添加文件，可点击“重新检查资料文件”，立即扫描并更新比较基准。检查结果表示**相较上次检查的变化**，不等于尚未入库的文件清单；连续两次检查且文件未变时，第二次显示没有变化。
- 检查只读取文件名、大小和修改时间，不切分文本、不生成向量，也不调用 DeepSeek。读取清单、扫描目录或保存失败时会提示，旧清单会保留，其他网页功能仍可使用。

每次成功检查后保存当前清单。检测到变化后，可使用“重新读取资料并更新知识库”生成新的全量索引；检查本身不会修改任何 Chroma 集合。

### 知识库问答的数据流

```text
用户问题 → 保存用户消息到 Histor
    ↓
问题加检索提示 → 本地模型生成问题向量
    ↓
Chroma 使用余弦距离检索所选集合（默认返回前 3 个片段）
    ↓
组织参考片段、来源、本次问题及此前对话
    ↓
DeepSeek 远程 API 生成回答
    ↓
网页显示回答和参考片段 → 保存回答、模型名称、Token 用量及来源
```

Chroma 当前使用 HNSW 近似检索；检索结果可能与逐个精确比较略有不同。显示的相似度由 `1 - 余弦距离` 得到，不代表答案正确的概率。普通对话则直接发送对话消息，不检索知识库。

## 文件职责

| 路径 | 作用 |
| --- | --- |
| `app.py` | 网页入口：读取配置、管理会话、选择模式、调用 DeepSeek 和显示结果 |
| `chat_ui.py` | 统一显示聊天消息、返回模型、Token 用量及参考片段 |
| `upload_ui.py` | 上传及更新知识库的 Streamlit 界面 |
| `document_monitor_ui.py` | 缓存启动检查结果，显示文件变化并提供手动重新检查按钮 |
| `document_monitor.py` | 递归扫描资料文件信息，与上次清单比较并原子保存新清单 |
| `knowledge_upload.py` | 保存上传文件、读取全部资料并触发索引构建 |
| `document_reader.py` | 读取 TXT、Markdown、PDF、DOCX，并按字符切分 |
| `embedding_demo.py` | 配置并缓存本地 FastEmbed 模型，提供文字向量化函数和问题检索前缀 |
| `chroma_index.py` | 创建、检查、列出和查询本地 Chroma 集合 |
| `rag_demo.py` | 检索片段并组织发送给模型的消息 |
| `history.py` | 对话 JSON 文件的创建、读取与保存 |
| `migrate_json_indexes.py` | 将兼容的旧 JSON 索引导入 Chroma |
| `similarity_demo.py`、`retrieval_demo.py` | 保留的向量与相似度学习代码；不在当前主页单独展示 |
| `index_demo.py`、`index_search.py` | 保留的早期 JSON 平面索引与逐条检索代码；当前主页使用 Chroma |
| `rag_trace.py` | 早期的问答过程展示代码；当前主页未接入 |
| `docs/学习过程.md` | 整理前的逐步学习记录；历史操作以本 README 为准 |

## 安装和启动

以下命令在项目根目录执行。当前项目已在 macOS、Python 3.13 的 `application-llm-universe` Conda 环境中验证；其他系统或 Python 版本需要自行确认依赖兼容性。

### 1. 创建环境并安装依赖

如果已有项目环境，可先激活它；也可以创建一个新的 Python 3.13 环境：

```bash
conda create -n application-llm-universe python=3.13
conda activate application-llm-universe
```

如果同名环境已经存在，直接使用 `conda activate`，不需要重复创建。

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` 包含网页和 API 的主依赖，并引用文档读取、向量模型、Chroma 的三个分项依赖文件。依赖安装需要网络；本地推理时模型不会自动下载。

### 2. 配置 DeepSeek API Key

在项目根目录创建 `.env.local`：

```dotenv
DEEPSEEK_API_KEY=在这里填写自己的Key
```

程序从这个文件读取 Key。`.env.local` 已由 `.gitignore` 排除；请不要把实际 Key 写进代码或提交到 Git。

### 3. 准备本地嵌入模型

在 `models/bge-small-zh-v1.5/` 中准备以下五个文件：

```text
model_optimized.onnx
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
```

项目使用 FastEmbed 的 `BAAI/bge-small-zh-v1.5` 配置，运行时只读取该目录，不会自动下载。可以从原项目已准备好的同款模型目录复制这些文件，或自行从 [Qdrant 的模型文件页面](https://huggingface.co/Qdrant/bge-small-zh-v1.5/tree/main)准备对应文件。模型目录约 91 MiB，不包含在 Git 提交中。

### 4. 启动网页

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8503
```

浏览器打开 `http://127.0.0.1:8503`。停止服务时，在运行命令的终端按 `Ctrl+C`。

在 PyCharm 中可以新建 Python 运行配置：选择项目环境作为解释器，**Module name** 填写 `streamlit`，参数填写 `run app.py --server.address 127.0.0.1 --server.port 8503`，工作目录设置为本项目根目录。随后可以用运行按钮启动。

## 使用方式

1. 启动并打开网页后，页面顶部会自动显示本地资料检查结果；首次启用时建立基准。可选择文件并点击“上传并更新知识库”，或先手动放入 `资料/`，再点击“重新读取资料并更新知识库”。服务运行期间手动添加资料，可先点击“重新检查资料文件”。
2. 建库完成后，选择“知识库问答”和需要使用的集合；也可以选择“普通对话”。
3. 输入问题并查看回答。知识库问答的问题最多 200 字，回答下可展开查看参考片段。
4. 使用“选择对话历史”继续已有对话，或选择“新对话”。新会话在首次提问后创建历史文件，文件名包含日期和首个问题。

TXT 和 Markdown 应为 UTF-8；PDF 按页读取、切分时不跨页。扫描版 PDF 需要先进行 OCR；旧版 `.doc` 需要先转换为 `.docx`。Word 图片、页眉页脚及复杂版式不保证完整提取；文件读取失败或部分页面没有文字时，网页会显示提示。

当前检索只使用本次输入的问题，不自动将前文代词改写为完整问题。检索片段和模型回答可能不准确，使用时应结合原文核对。

DeepSeek 当前启用思考模式，思考内容与最终回答共用输出 Token 额度。普通对话上限为 512，知识库问答上限为 8192，可在 `app.py` 中的 `CHAT_MAX_TOKENS` 和 `RAG_MAX_TOKENS` 调整。上限提高可能增加用量；如果达到输出或上下文限制，网页会提示缩短问题或历史。部分回答被截断时会保留并显示提醒。

## 本地数据与 Git

| 目录或文件 | 内容 | 当前忽略情况 |
| --- | --- | --- |
| `.env.local` | DeepSeek API Key | 已忽略 |
| `Histor/` | 明文聊天历史，可能包括参考片段 | 已忽略 |
| `models/` | 本地嵌入模型文件 | 已忽略 |
| `.local/chroma/` | Chroma 集合、向量、原文和来源 | `.local/` 已忽略 |
| `.local/document_inventory.json` | 上次检查的资料相对路径、大小、修改时间和检查时间 | `.local/` 已忽略 |
| `vector_index/` | 旧 JSON 索引快照 | 已忽略 |
| `资料/上传/` | 网页上传的资料 | 已忽略 |
| `资料/` 的其他内容 | 手动放入或复制的本地资料 | **默认未全部忽略**，上传 GitHub 前请检查 |

`.gitignore` 不会自动移除 Git 已经跟踪的文件或已有提交中的内容。历史记录、资料和索引均可能包含私人信息；选择知识库问答时，相关片段和对话会发送到 DeepSeek。请根据实际用途备份资料与历史；当前历史文件适合单会话使用，多个网页会话同时修改同一对话可能互相覆盖。

## 迁移旧 JSON 索引

如果项目已有 `vector_index/` 下的旧 JSON 快照，可运行：

```bash
python migrate_json_indexes.py
```

迁移会校验模型、预处理、维度和记录，再将兼容的原有向量及资料导入新的 Chroma 集合；原 JSON 不会被删除。资料变化后仍需要更新知识库。迁移脚本不读取 DeepSeek API Key，也不调用 DeepSeek。
