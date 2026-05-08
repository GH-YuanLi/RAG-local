# 轻量级本地 RAG 系统技术方案

## 1. 整体架构

```
用户查询 → Query Embedding → 向量检索 → 上下文拼接 → LLM 生成 → 返回结果
                              ↑
            文档 → 分块 → Embedding → 向量库
```

核心流程：**离线索引** + **在线检索生成**，两部分解耦。

---

## 2. 技术选型

| 组件 | 方案 | 理由 |
|------|------|------|
| **语言** | Python 3.10+ | RAG 生态最成熟 |
| **文档加载** | `langchain-community` loaders | 支持 PDF/Markdown/TXT/DOCX 等常见格式 |
| **文本分块** | RecursiveCharacterTextSplitter | 按语义边界递归切分，通用且效果好 |
| **Embedding** | 本地: `sentence-transformers` (如 `bge-small-zh-v1.5`) / 远程: OpenAI API | 本地免费用，远程质量更高，可切换 |
| **向量库** | ChromaDB (嵌入式，纯本地) | 零依赖启动，单文件持久化，轻量首选 |
| **LLM** | OpenAI 兼容 API（支持本地 Ollama / 远程 API 切换） | 最大灵活性，一行配置切换后端 |
| **Web 界面** | Gradio / Streamlit | 几行代码出 UI，轻量够用 |
| **配置管理** | YAML 配置文件 | 模型路径、API 地址、分块参数统一管理 |

---

## 3. 详细步骤

### Step 1: 项目结构初始化

```
local-rag/
├── config.yaml           # 全局配置
├── requirements.txt      # 依赖
├── data/
│   └── raw/              # 原始文档存放目录
├── vectordb/             # ChromaDB 持久化目录
├── src/
│   ├── __init__.py
│   ├── config.py         # 配置加载
│   ├── loader.py         # 文档加载与分块
│   ├── embedder.py       # Embedding 封装（本地/远程可切换）
│   ├── vectorstore.py    # 向量库操作（索引/检索）
│   ├── rag.py            # RAG 主流程（检索+生成）
│   └── app.py            # Web 界面入口
└── README.md
```

### Step 2: 配置系统 (`config.yaml`)

```yaml
embedding:
  provider: local          # local | openai
  model_name: BAAI/bge-small-zh-v1.5  # 本地模型名
  # api_key: sk-xxx       # 远程时启用
  # api_base: https://api.openai.com/v1

llm:
  provider: openai_compatible
  api_base: http://localhost:11434/v1  # Ollama 默认地址
  model_name: qwen2.5:7b
  # api_key: sk-xxx       # 远程时启用

chunking:
  chunk_size: 512
  chunk_overlap: 64

vectorstore:
  persist_directory: ./vectordb
  collection_name: documents

retrieval:
  top_k: 4
```

**解释**：集中管理所有可调参数，embedding/LLM 都支持本地与远程切换，方便在不同环境下使用。

### Step 3: 文档加载与分块 (`loader.py`)

- 支持 PDF、Markdown、TXT、DOCX
- 使用 `RecursiveCharacterTextSplitter`，按 `["\n\n", "\n", "。", "！", "？", ".", " "]` 分隔符递归切分
- 每个 chunk 保留元数据（来源文件名、页码等），用于溯源

**解释**：递归切分优先在段落/句子边界断开，避免语义断裂。中文分隔符确保中文文本切分质量。

### Step 4: Embedding 封装 (`embedder.py`)

- **本地模式**：用 `sentence-transformers` 加载 `bge-small-zh-v1.5`（~100MB，中文效果好，CPU 可跑）
- **远程模式**：调用 OpenAI Embedding API
- 统一暴露 `embed_texts(texts) -> List[List[float]]` 接口

**解释**：`bge-small-zh` 是目前中文场景性价比最高的本地 embedding 模型，无需 GPU。

### Step 5: 向量库操作 (`vectorstore.py`)

- 使用 ChromaDB 的持久化模式
- 提供两个核心方法：
  - `build_index(docs_dir)`: 扫描文档目录 → 加载 → 分块 → embedding → 写入向量库
  - `query(query_text, top_k)`: 查询 embedding → 余弦相似度检索 → 返回 top_k 结果（含原文 + 元数据）

**解释**：ChromaDB 嵌入式运行，无需额外服务进程，数据持久化到本地文件夹，重启不丢失。

### Step 6: RAG 主流程 (`rag.py`)

```
输入: user_query
1. query_embedding = embedder.embed_texts(user_query)
2. results = vectorstore.query(query_embedding, top_k=4)
3. context = "\n---\n".join([r.text for r in results])
4. prompt = f"""基于以下参考资料回答问题。如果资料中没有相关信息，请说明。
   参考资料：
   {context}

   问题：{user_query}
   回答："""
5. response = llm.chat(prompt)
6. 返回 response + 来源信息
```

**解释**：Prompt 中明确要求"无信息则说明"，减少幻觉。返回来源信息便于用户核实。

### Step 7: Web 界面 (`app.py`)

- 使用 Gradio 构建简单界面
- 包含：查询输入框、提交按钮、回答展示区、来源展示区
- 可选：添加文档上传按钮（触发增量索引）

**解释**：Gradio 几行代码就能出一个可用的 Web UI，适合快速验证。

### Step 8: 增量更新支持

- 记录已索引文件的 hash 值
- 再次运行 `build_index` 时只处理新增/变更文件
- 避免重复索引

---

## 4. 依赖清单 (`requirements.txt`)

```
chromadb>=0.4.0
sentence-transformers>=2.2.0
langchain-community>=0.0.10
openai>=1.0.0
gradio>=4.0.0
PyMuPDF>=1.23.0        # PDF 解析
python-docx>=0.8.11    # DOCX 解析
pyyaml>=6.0
```

---

## 5. 使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 把文档放入 data/raw/

# 3. 构建索引
python -m src.vectorstore --build

# 4. 启动 Web 界面
python -m src.app
```

---

## 6. 可选扩展（后续按需添加）

- **Re-ranking**：检索后用 Cross-encoder 重排，提升精度
- **多轮对话**：维护对话历史，query 改写
- **流式输出**：LLM 逐 token 返回，体验更好
- **多集合管理**：按主题分不同 ChromaDB collection
