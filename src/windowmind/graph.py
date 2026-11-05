from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from .config import Settings
from .memory import WindowMemory
from .prompts import WINDOW_SUMMARY_PROMPT, ANSWER_PROMPT
from .schemas import AgentState
from .windowing import SlidingWindowEngine

class WindowMindAgent:
    """A deterministic LangGraph pipeline for document-window analysis."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.llm = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )
        self.window_engine = SlidingWindowEngine(settings.window_size, settings.window_overlap)
        self.memory = WindowMemory(settings.ollama_embed_model, settings.ollama_base_url)
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("create_windows", self._create_windows)
        builder.add_node("summarize_windows", self._summarize_windows)
        builder.add_node("index_memory", self._index_memory)
        builder.add_node("retrieve_context", self._retrieve_context)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_edge(START, "create_windows")
        builder.add_edge("create_windows", "summarize_windows")
        builder.add_edge("summarize_windows", "index_memory")
        builder.add_edge("index_memory", "retrieve_context")
        builder.add_edge("retrieve_context", "generate_answer")
        builder.add_edge("generate_answer", END)
        return builder.compile()

    def _create_windows(self, state: AgentState) -> dict:
        windows, stats = self.window_engine.split(state["document_text"])
        return {"windows": [w.model_dump() for w in windows], "stats": stats.model_dump()}

    def _summarize_windows(self, state: AgentState) -> dict:
        summaries = []
        for window in state["windows"]:
            chain = WINDOW_SUMMARY_PROMPT | self.llm
            result = chain.invoke({
                "window_id": window["window_id"],
                "start_char": window["start_char"],
                "end_char": window["end_char"],
                "text": window["text"],
            })
            summary = str(result.content).strip()
            window["summary"] = summary
            summaries.append(summary)
        return {"windows": state["windows"], "summaries": summaries}

    def _index_memory(self, state: AgentState) -> dict:
        self.memory.index(state["windows"])
        return {}

    def _retrieve_context(self, state: AgentState) -> dict:
        matches = self.memory.retrieve(state["question"], k=self.settings.retrieval_k)
        blocks = []
        for doc in matches:
            window_id = doc.metadata["window_id"]
            blocks.append(f"[Window {window_id}]\n{doc.page_content}")
        context = "\n\n---\n\n".join(blocks)
        context = context[: self.settings.max_context_chars]
        return {"retrieved_context": context}

    def _generate_answer(self, state: AgentState) -> dict:
        chain = ANSWER_PROMPT | self.llm
        result = chain.invoke({
            "question": state["question"],
            "context": state.get("retrieved_context", ""),
        })
        return {"answer": str(result.content).strip()}

    def run(self, document_text: str, question: str) -> dict:
        return self.graph.invoke({"document_text": document_text, "question": question})
