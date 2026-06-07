""" 
Local sentence-transformers embeddings via langchain-huggingface. 
Inherits from LangChain's Embeddings base so FAISS accepts it direclty. 
"""


import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from app.config import get_settings
from app.utils.retry import embed_retry

logger = logging.getLogger(__name__)


class ResilientEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings wrapper around sentence-transformers.
    Runs fully locally — no API key required.
    """

    def __init__(self) -> None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        self._model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model ready.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_docs(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_q(text)


    def _embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    @embed_retry
    def _embed_q(self, text: str) -> list[float]:
        return self._model.embed_query(text)


@lru_cache(maxsize=1)
def get_embeddings() -> ResilientEmbeddings:
    return ResilientEmbeddings()