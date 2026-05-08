import json
import logging
from pathlib import Path

import chromadb

from src.embedder import LocalEmbedder
from src.loader import file_hash, load_and_chunk

logger = logging.getLogger(__name__)

_HASH_FILE = ".indexed_files.json"


class VectorStore:
    def __init__(
        self,
        embedder: LocalEmbedder,
        persist_directory: str = "./vectordb",
        collection_name: str = "documents",
    ):
        self.embedder = embedder
        self.persist_directory = persist_directory
        self.hash_file = Path(persist_directory) / _HASH_FILE

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.indexed_files = self._load_hash()
        logger.info("向量库初始化完成，已有 %d 条记录", self.collection.count())

    def _load_hash(self) -> dict:
        if self.hash_file.exists():
            with open(self.hash_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hash(self) -> None:
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, "w", encoding="utf-8") as f:
            json.dump(self.indexed_files, f, ensure_ascii=False, indent=2)

    def build_index(
        self,
        file_paths: list[str],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ) -> int:
        new_count = 0
        for path in file_paths:
            h = file_hash(path)
            if self.indexed_files.get(path) == h:
                logger.info("文件已索引，跳过: %s", path)
                continue

            chunks = load_and_chunk(path, chunk_size, chunk_overlap, separators)
            if not chunks:
                continue

            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_texts(texts)
            ids = [f"{Path(path).stem}_{c['metadata'].get('chunk_id', i)}" for i, c in enumerate(chunks)]
            metadatas = [{**c["metadata"], "file_hash": h} for c in chunks]

            self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
            self.indexed_files[path] = h
            new_count += len(chunks)
            logger.info("索引完成: %s (%d 分块)", path, len(chunks))

        self._save_hash()
        logger.info("索引构建完成，新增 %d 条记录，总计 %d 条", new_count, self.collection.count())
        return new_count

    def query(self, query_text: str, top_k: int = 4) -> list[dict]:
        query_embedding = self.embedder.embed_texts([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"][0]:
            return []

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def remove_file(self, file_path: str) -> int:
        source = str(file_path)
        results = self.collection.get(where={"source": source})
        if not results["ids"]:
            return 0
        self.collection.delete(ids=results["ids"])
        self.indexed_files.pop(file_path, None)
        self._save_hash()
        logger.info("已删除文件 %s 的 %d 条记录", file_path, len(results["ids"]))
        return len(results["ids"])

    def list_indexed_files(self) -> list[dict]:
        return [
            {"path": path, "hash": h}
            for path, h in self.indexed_files.items()
        ]

    def get_chunk_count(self, file_path: str) -> int:
        source = str(file_path)
        results = self.collection.get(where={"source": source})
        return len(results["ids"]) if results["ids"] else 0
