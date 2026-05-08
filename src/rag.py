import logging

from src.llm import LLMClient
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, vectorstore: VectorStore, llm: LLMClient, top_k: int = 4):
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

    def query(self, question: str) -> dict:
        logger.info("收到查询: %s", question[:50])

        results = self.vectorstore.query(question, top_k=self.top_k)
        if not results:
            return {
                "answer": "未找到相关文档内容，请先上传文档并完成索引。",
                "sources": [],
            }

        context = "\n---\n".join(r["text"] for r in results)
        answer = self.llm.chat(question, context)

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

    def update_llm(self, api_base: str, api_key: str, model_name: str, provider: str | None = None) -> None:
        self.llm.update_config(api_base, api_key, model_name, provider)

    def test_llm_connection(self) -> tuple[bool, str]:
        return self.llm.test_connection()
