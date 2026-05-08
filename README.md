# 本地 RAG 系统

基于本地 Embedding + 远程 LLM API 的轻量级检索增强生成系统。

## 功能特性

- 支持上传多个文档（PDF / DOCX / TXT / MD）
- 本地 Embedding 模型（BAAI/bge-small-zh-v1.5），无需 GPU
- 兼容 OpenAI 和 Anthropic 两种 LLM API 格式（GLM / Minimax / DeepSeek / Claude 等）
- Web 可视化界面：文档上传、对话查询、模型配置一站式操作
- ChromaDB 本地向量库，数据持久化

## 环境要求

- Python 3.12+
- 内存 ≥ 8GB
- 磁盘 ≥ 2GB（模型 + 依赖）

## 快速开始

### 1. 安装 uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖

```bash
cd RAG
uv sync
```

所有依赖安装在项目虚拟环境 `.venv` 中，不会影响系统环境。

### 3. 下载 Embedding 模型

首次使用前，需下载 Embedding 模型到本地（约 100MB）：

```bash
# 使用国内镜像下载
HF_ENDPOINT=https://hf-mirror.com uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')
print('模型下载完成')
"
```

### 4. 配置 LLM API

编辑 `config.yaml`，填入你的 LLM API 信息：

```yaml
llm:
  provider: openai                                    # 接口格式: openai | anthropic
  api_base: https://open.bigmodel.cn/api/paas/v4      # API 地址
  api_key: your-api-key-here                          # API Key
  model_name: glm-4-flash                             # 模型名称
```

常见厂商配置：

| 厂商 | provider | api_base | 模型示例 |
|------|----------|----------|---------|
| GLM（智谱） | `openai` | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| Minimax | `openai` | `https://api.minimax.chat/v1` | `MiniMax-Text-01` |
| DeepSeek | `openai` | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Claude（Anthropic） | `anthropic` | `https://api.anthropic.com` | `claude-sonnet-4-6` |
| Anthropic 代理 | `anthropic` | 代理地址 | 对应模型名 |

> 也可以在 Web 界面的"模型设置"Tab 中配置，无需手动编辑文件。

### 5. 启动服务

```bash
uv run python -m src.app
```

浏览器访问 `http://localhost:7860` 即可使用。

## 使用方式

### 启动服务

```bash
uv run python -m src.app
```

启动后浏览器访问 `http://localhost:7860`。界面包含三个 Tab：

---

### Tab 1：对话查询

在输入框中输入问题，点击"发送"或按回车，系统会：

1. 将问题转为向量，在向量库中检索最相关的文档片段
2. 将检索结果作为上下文，调用 LLM 生成回答
3. 回答下方显示来源文档和相似度

**注意**：查询前需先在"文档管理"中上传并索引文档，否则会提示"未找到相关文档内容"。

---

### Tab 2：文档管理

**上传文档：**

1. 点击上传区域或拖拽文件（支持多选）
2. 支持格式：PDF / DOCX / TXT / MD
3. 点击"上传并索引"，系统自动完成：加载 → 分块 → Embedding → 入库
4. 上传完成后状态栏显示"索引完成，新增 X 个分块"

**已索引文档：**

- 列表展示已索引文档及分块数
- 点击"刷新列表"更新
- 删除文档：在"要删除的文件名"输入框中填写文件名，点击"删除文件"

**增量索引：** 已索引的文件不会重复处理，修改后的文件会自动更新。

---

### Tab 3：模型设置

在界面中实时切换 LLM API，无需重启服务：

1. **接口格式**：选择 `openai` 或 `anthropic`
   - `openai`：适用于 GLM / Minimax / DeepSeek 等提供 OpenAI 兼容 API 的厂商
   - `anthropic`：适用于 Claude 或提供 Anthropic 兼容 API 的代理接口
2. **API Base URL**：填写厂商的 API 地址
3. **API Key**：填写你的 API Key（密码模式，不显示明文）
4. **模型名称**：填写对应的模型名
5. 点击"测试连接"验证配置是否正确
6. 点击"保存配置"将当前设置持久化到 `config.yaml`

常见厂商配置示例：

| 厂商 | 接口格式 | API Base URL | 模型名称 |
|------|---------|-------------|---------|
| GLM（智谱） | openai | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-flash` |
| Minimax | openai | `https://api.minimax.chat/v1` | `MiniMax-Text-01` |
| DeepSeek | openai | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Claude | anthropic | `https://api.anthropic.com` | `claude-sonnet-4-6` |

## 项目结构

```
RAG/
├── pyproject.toml          # uv 项目配置
├── config.yaml             # 全局配置
├── models/                 # 本地 Embedding 模型
│   └── bge-small-zh-v1.5/
├── data/raw/               # 原始文档目录
├── vectordb/               # ChromaDB 持久化目录
├── logs/                   # 执行日志
├── src/
│   ├── config.py           # 配置加载
│   ├── loader.py           # 文档加载与分块
│   ├── embedder.py         # Embedding 封装
│   ├── vectorstore.py      # 向量库操作
│   ├── llm.py              # LLM 客户端
│   ├── rag.py              # RAG 主流程
│   └── app.py              # Web 界面
└── README.md
```

## 配置说明

`config.yaml` 完整配置项：

```yaml
embedding:
  model_name: BAAI/bge-small-zh-v1.5   # Embedding 模型
  device: cpu                            # 运行设备
  hf_endpoint: https://hf-mirror.com    # HuggingFace 镜像

llm:
  provider: openai         # 接口格式: openai | anthropic
  api_base: https://open.bigmodel.cn/api/paas/v4
  api_key: ""
  model_name: glm-4-flash

chunking:
  chunk_size: 512          # 分块大小（字符数）
  chunk_overlap: 64        # 分块重叠
  separators: ["\n\n", "\n", "。", "！", "？", ".", " "]

vectorstore:
  persist_directory: ./vectordb
  collection_name: documents

retrieval:
  top_k: 4                 # 检索返回的最相关文档数

app:
  host: 0.0.0.0
  port: 7860
```
