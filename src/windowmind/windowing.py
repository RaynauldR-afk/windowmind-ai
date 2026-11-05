from time import perf_counter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .schemas import WindowRecord, WindowStats

class SlidingWindowEngine:
    """Create overlapping character windows while preserving natural text boundaries.

    Character windows are used here for transparent, deterministic behavior.
    They are not exact tokenizer-token windows; model tokenization varies by model.
    """

    def __init__(self, window_size: int = 1200, overlap: int = 200):
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be >= 0 and smaller than window_size")
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=window_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )

    def split(self, text: str) -> tuple[list[WindowRecord], WindowStats]:
        if not text or not text.strip():
            raise ValueError("Document cannot be empty")
        started = perf_counter()
        docs = self.splitter.create_documents([text])
        windows = []
        for index, doc in enumerate(docs):
            start = int(doc.metadata.get("start_index", 0))
            content = doc.page_content
            windows.append(WindowRecord(
                window_id=index,
                start_char=start,
                end_char=start + len(content),
                text=content,
            ))
        elapsed = perf_counter() - started
        stats = WindowStats(
            total_characters=len(text),
            total_windows=len(windows),
            window_size=self.window_size,
            overlap=self.overlap,
            stride=self.stride,
            average_window_chars=round(sum(len(w.text) for w in windows) / len(windows), 2),
            processing_seconds=round(elapsed, 6),
        )
        return windows, stats
