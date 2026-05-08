# 轻量级本地 RAG 系统技术方案（调整版）

## 1. 整体架构

```
用户查询 → Query Embedding → 向量检索 → 上下文拼接 → LLM 生成 → 返回结果
                              ↑
            文档 → 分块 → Embedding → 向量库
```

核心流程：**离线索引** + **在线检索生成**，两部分解耦。

**架构选择：Naive RAG**

理由详见第 2 节 RAG 架构分析。

---

## 2. RAG 架构分析与选型

| 架构 | 核心思路 | 优势 | 劣势 | 资源需求 | 适用场景 |
|------|---------|------|------|---------|---------|
| **Naive RAG** | 索引 → 检索 → 生成 | 实现简单、延迟低、资源占用少、易调试 | 检索质量依赖分块策略，无查询优化，可能遗漏相关上下文 | 低 | 轻量级应用、资源受限环境 |
| **Advanced RAG** | 在 Naive 基础上增加查询改写/重排序 | 检索质量高、减少幻觉、上下文更相关 | 实现复杂、需额外模型（重排序器）、延迟增加 | 中 | 对检索精度有较高要求的场景 |
| **Modular RAG** | 模块化可插拔管线 | 灵活度最高、组件可替换 | 架构复杂、简单场景过度设计、维护成本高 | 可变 | 需要频繁调整组件的实验性项目 |
| **Graph RAG** | 知识图谱 + 向量检索 | 擅长多跳推理、结构化知识表示 | 预处理重（图谱构建）、实现复杂、非所有文档类型适用 | 高 | 知识密集型、需关联推理的场景 |
| **Agentic RAG** | LLM 作为智能体决定检索策略 | 自适应检索、处理复杂查询 | 多次 LLM 调用（API 成本高）、延迟不可控、调试困难 | 本地低但 API 成本高 | 复杂推理、研究型任务 |

### 选型结论：Naive RAG

基于以下因素选择 Naive RAG：

1. **设备条件**：CPU-only embedding（i5-1135G7, 16GB RAM），Naive RAG 资源占用最低
2. **API 成本控制**：远程 LLM 调用按 token 计费，Naive RAG 仅需 1 次 LLM 调用，Advanced/Agentic RAG 需多次调用，成本显著增加
3. **实现复杂度**：项目定位为轻量级本地 RAG，Naive RAG 实现简单、易维护
4. **可扩展性**：后续可在 Naive RAG 基础上增量添加重排序（Re-ranking）模块，平滑升级为 Advanced RAG

---

## 3. 设备配置评估与 Embedding 模型选型

### 当前设备配置

| 项目 | 规格 |
|------|------|
| CPU | 11th Gen Intel Core i5-1135G7 @ 2.40GHz（4核8线程） |
| 内存 | 16 GB |
| GPU | 无独立 GPU（Intel Iris Xe 集成显卡） |

### Embedding 模型评估

| 模型 | 参数量 | 维度 | 最大 token | 磁盘占用 | CPU 推理速度 | 中文效果 | 推荐场景 |
|------|-------|------|-----------|---------|-------------|---------|---------|
| **BAAI/bge-small-zh-v1.5** | 33M | 512 | 512 | ~100MB | 快（~50ms/条） | 良好 | **默认推荐**：速度与质量的最佳平衡 |
| BAAI/bge-base-zh-v1.5 | 102M | 768 | 512 | ~400MB | 中（~120ms/条） | 较好 | 质量优先、对速度要求不高 |
| BAAI/bge-m3 | 568M | 1024 | 8192 | ~2.2GB | 慢（~300ms/条） | 优秀 | 多语言需求、长文本支持 |

### 选型结论：BAAI/bge-small-zh-v1.5

- CPU 推理速度快，适合实时查询场景
- 内存占用小，不会与 ChromaDB 和 Web 服务争抢资源
- 中文效果良好，在 C-MTEB 基准上表现优秀
- 模型首次下载后缓存至本地，后续使用无需联网

> 配置文件中支持切换为其他模型，用户可根据实际需求调整。

---

## 4. 技术选型

| 组件 | 方案 | 理由 |
|------|------|------|
| **语言** | Python 3.10+ | RAG 生态最成熟 |
| **包管理** | uv | 快速依赖解析与安装，严格项目隔离 |
| **文档加载** | PyMuPDF + python-docx + 标准库 | 按 PDF/DOCX/TXT/MD 分别处理，避免 langchain 重依赖 |
| **文本分块** | 自定义 RecursiveCharacterTextSplitter | 按 `["\n\n", "\n", "。", "！", "？", ".", " "]` 递归切分，无需引入 langchain |
| **Embedding** | `sentence-transformers` + `BAAI/bge-small-zh-v1.5` | 本地 CPU 运行，中文效果好，无需 GPU |
| **向量库** | ChromaDB（嵌入式，纯本地） | 零依赖启动，单文件持久化，轻量首选 |
| **LLM** | `openai` SDK（OpenAI 兼容接口） | minimax、GLM 等厂商均提供 OpenAI 兼容 API，一行 `base_url` 切换 |
| **Web 界面** | Gradio 4.x | 多文件上传组件、Tabs 布局、配置面板，开箱即用 |
| **配置管理** | YAML 配置文件 + Web 界面热更新 | 模型路径、API 地址、分块参数统一管理，界面可实时修改 |

### LLM API 兼容性设计

目标：支持 minimax、GLM 等主流厂商的 OpenAI 兼容接口。

| 厂商 | API Base URL | 模型示例 | 兼容方式 |
|------|-------------|---------|---------|
| Minimax | `https://api.minimax.chat/v1` | `MiniMax-Text-01` | OpenAI 兼容 |
| GLM（智谱） | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` | OpenAI 兼容 |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` | OpenAI 兼容 |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` | 原生 |

统一使用 `openai` Python SDK，通过修改 `base_url` 和 `api_key` 参数即可切换厂商，无需改动业务代码。

---

## 5. 项目结构

```
RAG/
├── pyproject.toml          # uv 项目配置（替代 requirements.txt）
├── config.yaml             # 全局配置
├── data/
│   └── raw/                # 原始文档存放目录
├── vectordb/               # ChromaDB 持久化目录
├── logs/                   # 执行日志目录
├── src/
│   ├── __init__.py
│   ├── config.py           # 配置加载与管理
│   ├── loader.py           # 文档加载与分块
│   ├── embedder.py         # Embedding 封装（本地模式）
│   ├── vectorstore.py      # 向量库操作（索引/检索）
│   ├── llm.py              # LLM 客户端（OpenAI 兼容，支持多厂商切换）
│   ├── rag.py              # RAG 主流程（检索+生成）
│   └── app.py              # Web 界面入口
└── README.md
```

与原方案的主要变更：
- `requirements.txt` → `pyproject.toml`（uv 管理）
- 新增 `llm.py`：独立 LLM 客户端模块，封装 OpenAI 兼容接口调用
- 新增 `logs/`：执行日志目录
- 移除 langchain-community 依赖，文档加载和分块改为自实现

---

## 6. 详细设计

### 6.1 项目初始化（uv）

```bash
# 在 RAG/ 目录下
uv init --no-readme
uv add sentence-transformers chromadb openai gradio PyMuPDF python-docx pyyaml
```

`pyproject.toml` 核心内容：

```toml
[project]
name = "local-rag"
version = "0.1.0"
description = "轻量级本地 RAG 系统"
requires-python = ">=3.10"
dependencies = [
    "sentence-transformers>=2.2.0",
    "chromadb>=0.4.0",
    "openai>=1.0.0",
    "gradio>=4.0.0",
    "PyMuPDF>=1.23.0",
    "python-docx>=0.8.11",
    "pyyaml>=6.0",
]
```

### 6.2 配置系统（`config.yaml`）

```yaml
embedding:
  model_name: BAAI/bge-small-zh-v1.5  # 本地 embedding 模型
  device: cpu                           # 运行设备

llm:
  api_base: https://open.bigmodel.cn/api/paas/v4  # 默认 GLM 接口
  api_key: ""                                       # 用户在界面中配置
  model_name: glm-4-flash                           # 默认模型

chunking:
  chunk_size: 512
  chunk_overlap: 64
  separators: ["\n\n", "\n", "。", "！", "？", ".", " "]

vectorstore:
  persist_directory: ./vectordb
  collection_name: documents

retrieval:
  top_k: 4

app:
  host: 0.0.0.0
  port: 7860
```

配置加载逻辑：
- 启动时从 `config.yaml` 读取默认值
- Web 界面修改的 LLM 配置实时生效（仅修改运行时配置，不写回文件，重启恢复默认）
- 如需持久化界面修改，可在界面中点击"保存配置"按钮写回 `config.yaml`

### 6.3 文档加载与分块（`loader.py`）

支持的文件格式及处理方式：

| 格式 | 处理库 | 说明 |
|------|-------|------|
| PDF | PyMuPDF (`fitz`) | 提取文本，保留页码元数据 |
| DOCX | python-docx | 按段落提取，保留标题层级 |
| TXT | 内置 `open()` | 直接读取 |
| MD | 内置 `open()` | 直接读取，Markdown 语法保留 |

分块策略：
- 使用自定义 `RecursiveCharacterTextSplitter`
- 分隔符优先级：`\n\n` > `\n` > `。` > `！` > `？` > `.` > ` `
- 每个 chunk 携带元数据：`{"source": 文件名, "page": 页码, "chunk_id": 序号}`

多文件上传处理：
- 支持一次上传多个文件（Gradio File 组件支持多选）
- 逐文件加载、分块，合并后统一入库
- 去重机制：基于文件名 + 内容 hash，跳过已索引文件

### 6.4 Embedding 封装（`embedder.py`）

- 使用 `sentence-transformers` 加载 `BAAI/bge-small-zh-v1.5`
- 首次运行自动从 HuggingFace 下载模型至缓存目录 `~/.cache/huggingface/`
- 统一接口：`embed_texts(texts: List[str]) -> List[List[float]]`
- 支持 batch 编码，避免逐条处理效率低下

```python
class LocalEmbedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
```

### 6.5 向量库操作（`vectorstore.py`）

- 使用 ChromaDB 持久化模式，数据存储在 `./vectordb/` 目录
- 核心方法：
  - `build_index(file_paths: List[str])`: 文件加载 → 分块 → embedding → 写入向量库（增量，仅处理新文件）
  - `query(query_text: str, top_k: int) -> List[dict]`: 查询 embedding → 余弦相似度检索 → 返回 top_k 结果（含原文 + 元数据）
- 增量索引：记录已入库文件的 hash，重复文件自动跳过

### 6.6 LLM 客户端（`llm.py`）

- 基于 `openai` SDK，通过修改 `base_url` 兼容不同厂商
- 核心接口：`chat(prompt: str, context: str) -> str`
- 支持运行时切换 `api_base`、`api_key`、`model_name`
- 异常处理：API 调用失败时返回友好错误信息

```python
from openai import OpenAI

class LLMClient:
    def __init__(self, api_base: str, api_key: str, model_name: str):
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def update_config(self, api_base: str, api_key: str, model_name: str):
        """运行时切换 LLM 配置"""
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model_name = model_name

    def chat(self, system_prompt: str, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content
```

### 6.7 RAG 主流程（`rag.py`）

```
输入: user_query
1. results = vectorstore.query(query_text=user_query, top_k=4)
2. context = "\n---\n".join([r["text"] for r in results])
3. system_prompt = "你是一个文档问答助手。基于以下参考资料回答问题。如果资料中没有相关信息，请明确说明。"
4. user_message = f"参考资料：\n{context}\n\n问题：{user_query}"
5. response = llm.chat(system_prompt, user_message)
6. 返回 response + 来源信息（文件名、页码等）
```

设计要点：
- Prompt 中明确要求"无信息则说明"，减少幻觉
- 返回来源信息便于用户核实
- 可扩展：后续可在检索后加入重排序（Re-ranking）步骤

### 6.8 Web 界面（`app.py`）

使用 Gradio 4.x 构建界面，采用 Tabs 布局：

```
┌─────────────────────────────────────────────────┐
│  [对话查询]  [文档管理]  [模型设置]              │
├─────────────────────────────────────────────────┤
│                                                  │
│  Tab 1: 对话查询                                │
│  ┌─────────────────────────────────────────┐    │
│  │  聊天历史展示区                          │    │
│  │  User: xxx                               │    │
│  │  Assistant: xxx                          │    │
│  │  [来源: doc1.pdf p3, doc2.txt p1]        │    │
│  └─────────────────────────────────────────┘    │
│  ┌─────────────────────────┬──────────┐         │
│  │  输入查询...             │  发送    │         │
│  └─────────────────────────┴──────────┘         │
│                                                  │
│  Tab 2: 文档管理                                │
│  ┌─────────────────────────────────────────┐    │
│  │  📎 拖拽或点击上传文件（支持多选）       │    │
│  │  支持 PDF / DOCX / TXT / MD             │    │
│  └─────────────────────────────────────────┘    │
│  [上传并索引]                                    │
│  ┌─────────────────────────────────────────┐    │
│  │  已索引文档列表                          │    │
│  │  - doc1.pdf (12 chunks)                 │    │
│  │  - doc2.txt (5 chunks)                  │    │
│  │  [删除]                                  │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│  Tab 3: 模型设置                                │
│  API Base URL:  [_____________________________]  │
│  API Key:       [_____________________________]  │
│  模型名称:      [_____________________________]  │
│  Embedding 模型: [____________________________]  │
│  [保存配置]  [测试连接]                          │
│                                                  │
└─────────────────────────────────────────────────┘
```

功能说明：

| Tab | 功能 | 交互 |
|-----|------|------|
| 对话查询 | 输入问题，显示回答和来源 | 文本框输入 + 按钮/回车提交 |
| 文档管理 | 上传多文件、查看已索引文档、删除文档 | 文件上传组件 + 列表 + 删除按钮 |
| 模型设置 | 配置 LLM API 信息、测试连接 | 表单输入 + 保存/测试按钮 |

### 6.9 日志系统

- 日志文件存储在 `logs/` 目录，按日期命名：`logs/2026-05-08.log`
- 使用 Python `logging` 模块，同时输出到控制台和文件
- 记录内容：
  - 文档加载状态（成功/失败、文件名、chunk 数量）
  - 索引构建进度（文件数、总 chunk 数、耗时）
  - 查询日志（查询内容、检索结果数、LLM 响应耗时）
  - 错误信息（异常堆栈、API 调用失败详情）

---

## 7. 实施步骤

| Step | 任务 | 产出文件 |
|------|------|---------|
| 1 | uv 项目初始化，配置 `pyproject.toml`，安装依赖 | `pyproject.toml` |
| 2 | 实现配置加载模块 | `src/config.py` |
| 3 | 实现文档加载与分块模块 | `src/loader.py` |
| 4 | 实现 Embedding 封装模块 | `src/embedder.py` |
| 5 | 实现向量库操作模块 | `src/vectorstore.py` |
| 6 | 实现 LLM 客户端模块 | `src/llm.py` |
| 7 | 实现 RAG 主流程模块 | `src/rag.py` |
| 8 | 实现 Web 界面 | `src/app.py` |
| 9 | 集成测试与调优 | — |
| 10 | 编写 README.md | `README.md` |

---

## 8. 可选扩展（后续按需添加）

- **Re-ranking**：检索后用 Cross-encoder 重排，提升精度（升级为 Advanced RAG）
- **流式输出**：LLM 逐 token 返回，体验更好
- **多轮对话**：维护对话历史，query 改写
- **多集合管理**：按主题分不同 ChromaDB collection
- **OCR 支持**：扫描版 PDF 的文字识别
