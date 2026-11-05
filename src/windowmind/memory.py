from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings

class WindowMemory:
    """In-process vector memory for the current document session."""

    def __init__(self, model: str, base_url: str):
        embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        self.store = InMemoryVectorStore(embedding=embeddings)

    def index(self, windows: list[dict]) -> None:
        docs = [
            Document(
                page_content=w["text"],
                metadata={
                    "window_id": w["window_id"],
                    "start_char": w["start_char"],
                    "end_char": w["end_char"],
                    "summary": w.get("summary", ""),
                },
            )
            for w in windows
        ]
        if docs:
            self.store.add_documents(docs)

    def retrieve(self, query: str, k: int = 4) -> list[Document]:
        return self.store.similarity_search(query, k=k)
