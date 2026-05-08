import logging

from anthropic import Anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是一个文档问答助手。基于以下参考资料回答问题。"
    "如果资料中没有相关信息，请明确说明，不要编造答案。"
    "回答时请引用来源文件名。"
)


class LLMClient:
    def __init__(self, api_base: str, api_key: str, model_name: str, provider: str = "openai"):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self._init_client()

    def _init_client(self) -> None:
        key = self.api_key or "sk-placeholder"
        if self.provider == "anthropic":
            self.client = Anthropic(base_url=self.api_base, api_key=key)
        else:
            self.client = OpenAI(base_url=self.api_base, api_key=key)

    def update_config(self, api_base: str, api_key: str, model_name: str, provider: str | None = None) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        if provider is not None:
            self.provider = provider
        self._init_client()
        logger.info("LLM 配置已更新: provider=%s, base=%s, model=%s", self.provider, api_base, model_name)

    def chat(self, query: str, context: str) -> str:
        user_message = f"参考资料：\n{context}\n\n问题：{query}"
        try:
            if self.provider == "anthropic":
                return self._chat_anthropic(user_message)
            return self._chat_openai(user_message)
        except Exception as e:
            logger.error("LLM 调用失败: %s", e)
            return f"LLM 调用失败: {e}"

    def _chat_openai(self, user_message: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, user_message: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        texts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                texts.append(block.text)
        return "\n".join(texts) if texts else "模型未返回有效内容"

    def test_connection(self) -> tuple[bool, str]:
        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "hi"}],
                )
                reply = next(
                    (b.text for b in response.content if hasattr(b, "text")),
                    "无文本内容",
                )
                return True, f"连接成功，模型回复: {reply}"
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
            )
            return True, f"连接成功，模型回复: {response.choices[0].message.content}"
        except Exception as e:
            return False, f"连接失败: {e}"
