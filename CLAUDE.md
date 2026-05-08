# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本地 RAG 系统：基于本地 Embedding + 远程 LLM API 的检索增强生成系统，支持 PDF/DOCX/TXT/MD 文档上传和问答。

## 常用命令

```bash
# 安装依赖
uv sync

# 启动 Web 服务（Gradio 界面）
uv run python -m src.app

# 下载 Embedding 模型（国内镜像）
HF_ENDPOINT=https://hf-mirror.com uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')
"
```

## 架构

```
src/app.py          # Gradio Web 界面入口
src/rag.py          # RAG 主流程（检索 + 生成）
src/llm.py          # LLM 客户端（openai / anthropic 双模式）
src/vectorstore.py  # ChromaDB 向量库（索引 + 检索）
src/embedder.py     # 本地 Embedding（sentence-transformers）
src/loader.py       # 文档加载 + 分块
src/config.py       # YAML 配置加载
```

核心流程：`query` → embed → vectorstore.query → llm.chat → response

## LLM Provider 切换

`llm.py` 支持 `openai` 和 `anthropic` 两种 provider，通过 `provider` 参数切换：
- `openai`：GLM / Minimax / DeepSeek 等 OpenAI 兼容 API
- `anthropic`：Claude 或 Anthropic 兼容代理接口（如 `api.z.ai`）

`rag.py` 的 `update_llm` 方法和 `app.py` 的模型设置 Tab 传递 `provider` 参数。

**注意**：`anthropic` provider 的 `_chat_anthropic` 遍历 `response.content` 列表，
优先取 `TextBlock.text`，忽略 `ThinkingBlock`（MiniMax 等模型开启扩展思考时返回）。
GLM（无 ThinkingBlock）和 MiniMax 均兼容。

## 配置

所有配置集中在 `config.yaml`，启动时由 `src/config.py` 加载。LLM API 配置支持运行时在界面中修改并持久化。