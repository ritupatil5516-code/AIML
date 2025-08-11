from pydantic import BaseModel
import os
class AppConfig(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    max_context_docs: int = int(os.getenv("MAX_CONTEXT_DOCS", "12"))
    tool_loop_limit: int = int(os.getenv("TOOL_LOOP_LIMIT", "4"))
cfg = AppConfig()
