import logging
import sys
from pathlib import Path

import gradio as gr

from src.config import load_config, save_config
from src.embedder import LocalEmbedder
from src.llm import LLMClient
from src.rag import RAGEngine
from src.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent.parent / "logs" / "2026-05-08.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def init_rag() -> RAGEngine:
    cfg = load_config()
    embedder = LocalEmbedder(
        model_name=cfg["embedding"]["model_name"],
        device=cfg["embedding"]["device"],
    )
    vs = VectorStore(
        embedder=embedder,
        persist_directory=cfg["vectorstore"]["persist_directory"],
        collection_name=cfg["vectorstore"]["collection_name"],
    )
    llm = LLMClient(
        api_base=cfg["llm"]["api_base"],
        api_key=cfg["llm"]["api_key"],
        model_name=cfg["llm"]["model_name"],
        provider=cfg["llm"].get("provider", "openai"),
    )
    return RAGEngine(vectorstore=vs, llm=llm, top_k=cfg["retrieval"]["top_k"])


rag_engine: RAGEngine | None = None


def get_rag() -> RAGEngine:
    global rag_engine
    if rag_engine is None:
        rag_engine = init_rag()
    return rag_engine


def upload_files(files: list) -> str:
    if not files:
        return "未选择文件"
    rag = get_rag()
    cfg = load_config()
    paths = [f.name for f in files]
    count = rag.vectorstore.build_index(
        paths,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
        separators=cfg["chunking"]["separators"],
    )
    return f"索引完成，新增 {count} 个分块"


def refresh_file_list() -> str:
    rag = get_rag()
    files = rag.vectorstore.list_indexed_files()
    if not files:
        return "暂无已索引文档"
    lines = []
    for f in files:
        name = Path(f["path"]).name
        chunks = rag.vectorstore.get_chunk_count(f["path"])
        lines.append(f"- {name} ({chunks} 分块)")
    return "\n".join(lines)


def delete_file(file_list_text: str, selected: str) -> str:
    if not selected:
        return "未选择文件"
    rag = get_rag()
    name = selected.strip().split("(")[0].strip().lstrip("- ")
    for f in rag.vectorstore.list_indexed_files():
        if Path(f["path"]).name == name:
            deleted = rag.vectorstore.remove_file(f["path"])
            return f"已删除 {name}，移除 {deleted} 条记录"
    return f"未找到文件: {name}"


def chat_query(message: str, history: list) -> str:
    if not message.strip():
        return "请输入问题"
    rag = get_rag()
    result = rag.query(message)
    answer = result["answer"]
    if result["sources"]:
        src_text = "\n".join(
            f"  - {Path(s['source']).name} (第{s['page']}页, 相似度: {1 - s['distance']:.2f})"
            for s in result["sources"]
        )
        answer += f"\n\n---\n**来源：**\n{src_text}"
    return answer


def save_llm_config(provider: str, api_base: str, api_key: str, model_name: str) -> str:
    rag = get_rag()
    rag.update_llm(api_base, api_key, model_name, provider)
    cfg = load_config()
    cfg["llm"]["provider"] = provider
    cfg["llm"]["api_base"] = api_base
    cfg["llm"]["api_key"] = api_key
    cfg["llm"]["model_name"] = model_name
    save_config(cfg)
    return f"配置已保存并生效（{provider} 模式）"


def test_llm_connection(provider: str, api_base: str, api_key: str, model_name: str) -> str:
    rag = get_rag()
    rag.update_llm(api_base, api_key, model_name, provider)
    ok, msg = rag.test_llm_connection()
    return f"{'✅' if ok else '❌'} {msg}"


def build_app() -> gr.Blocks:
    cfg = load_config()

    with gr.Blocks(title="本地 RAG 系统") as app:
        gr.Markdown("# 本地 RAG 系统")

        with gr.Tabs():
            with gr.Tab("对话查询"):
                chatbot = gr.Chatbot(height=450)
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="输入你的问题...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                def chat_respond(message: str, history: list) -> tuple:
                    if not message.strip():
                        return "", history
                    answer = chat_query(message, history)
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": answer})
                    return "", history

                query_input.submit(chat_respond, [query_input, chatbot], [query_input, chatbot])
                send_btn.click(chat_respond, [query_input, chatbot], [query_input, chatbot])

            with gr.Tab("文档管理"):
                with gr.Row():
                    file_upload = gr.File(
                        label="上传文档",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md"],
                    )
                upload_btn = gr.Button("上传并索引", variant="primary")
                upload_status = gr.Textbox(label="状态", interactive=False)

                gr.Markdown("### 已索引文档")
                file_list = gr.Textbox(label="文档列表", value="暂无已索引文档", interactive=False, lines=8)
                refresh_btn = gr.Button("刷新列表")
                refresh_btn.click(refresh_file_list, outputs=file_list)

                delete_input = gr.Textbox(label="要删除的文件名", placeholder="从上方列表复制文件名")
                delete_btn = gr.Button("删除文件")
                delete_status = gr.Textbox(label="删除状态", interactive=False)

                upload_btn.click(upload_files, [file_upload], upload_status).then(
                    refresh_file_list, outputs=file_list
                )
                delete_btn.click(delete_file, [file_list, delete_input], delete_status).then(
                    refresh_file_list, outputs=file_list
                )

            with gr.Tab("模型设置"):
                gr.Markdown("### LLM API 配置")
                gr.Markdown(
                    "选择接口格式并填写对应配置：\n"
                    "- **OpenAI 兼容**：GLM / Minimax / DeepSeek 等\n"
                    "- **Anthropic 兼容**：Claude / 部分 Anthropic 代理接口"
                )
                provider_input = gr.Dropdown(
                    label="接口格式",
                    choices=["openai", "anthropic"],
                    value=cfg["llm"].get("provider", "openai"),
                )
                api_base_input = gr.Textbox(
                    label="API Base URL",
                    value=cfg["llm"]["api_base"],
                )
                api_key_input = gr.Textbox(
                    label="API Key",
                    value=cfg["llm"]["api_key"],
                    type="password",
                )
                model_name_input = gr.Textbox(
                    label="模型名称",
                    value=cfg["llm"]["model_name"],
                )

                with gr.Row():
                    save_btn = gr.Button("保存配置", variant="primary")
                    test_btn = gr.Button("测试连接")

                config_status = gr.Textbox(label="状态", interactive=False)

                save_btn.click(
                    save_llm_config,
                    [provider_input, api_base_input, api_key_input, model_name_input],
                    config_status,
                )
                test_btn.click(
                    test_llm_connection,
                    [provider_input, api_base_input, api_key_input, model_name_input],
                    config_status,
                )

    return app


if __name__ == "__main__":
    cfg = load_config()
    app = build_app()
    app.launch(
        server_name=cfg["app"]["host"],
        server_port=cfg["app"]["port"],
    )
