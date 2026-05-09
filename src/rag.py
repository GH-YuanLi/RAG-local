import logging

from src.llm import LLMClient
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


MAX_HISTORY_TURNS = 6


class RAGEngine:
    def __init__(self, vectorstore: VectorStore, llm: LLMClient, top_k: int = 4):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str, history: list | None = None) -> dict:
        logger.info("收到查询: %s", question[:50])

        results = self.vectorstore.query(question, top_k=self.top_k)
        if not results:
            return {
                "answer": "未找到相关文档内容，请先上传文档并完成索引。",
                "sources": [],
            }

        context = "\n---\n".join(r["text"] for r in results)

        if history:
            history = self._truncate_history(history)
            history_text = self._format_history(history)
            full_question = f"{history_text}\n\n当前问题：{question}"
        else:
            full_question = question

        answer = self.llm.chat(full_question, context)

        sources = []
        seen = set()
        for r in results:
            src = r["metadata"].get("source", "未知")
            page = r["metadata"].get("page", "")
            key = f"{src}_p{page}"
            if key not in seen:
                seen.add(key)
                sources.append({"source": src, "page": page, "distance": r["distance"]})

        logger.info("查询完成，检索到 %d 条相关内容", len(results))
        return {"answer": answer, "sources": sources}

    def _truncate_history(self, history: list) -> list:
        """截断过长历史，防止 token 溢出"""
        if len(history) <= MAX_HISTORY_TURNS * 2:
            return history
        return history[-(MAX_HISTORY_TURNS * 2):]

    def _format_history(self, history: list) -> str:
        """将历史格式化为可读的上下文"""
        lines = []
        for item in history:
            role = "用户" if item.get("role") == "user" else "助手"
            lines.append(f"{role}：{item.get('content', '')}")
        return "\n".join(lines)

    def update_llm(self, api_base: str, api_key: str, model_name: str, provider: str | None = None) -> None:
        self.llm.update_config(api_base, api_key, model_name, provider)

    def test_llm_connection(self) -> tuple[bool, str]:
        return self.llm.test_connection()
