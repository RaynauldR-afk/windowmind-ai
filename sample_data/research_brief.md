# Enterprise AI Adoption: Research Brief

Enterprise AI adoption requires more than selecting a capable language model. Organizations must
build reliable data pipelines, establish governance, measure model quality, and design systems
that remain useful as information grows.

Retrieval-augmented generation (RAG) connects language models to external knowledge sources.
A typical RAG pipeline ingests documents, divides them into chunks, creates vector embeddings,
stores those vectors, retrieves relevant passages, and supplies the passages to a language model.
The quality of chunking affects retrieval because chunks that are too small may lose context,
while chunks that are too large may contain several unrelated topics.

Sliding-window chunking creates overlapping segments. For example, a window size of 1,000
characters and overlap of 150 characters produces a stride of 850 characters. The overlap helps
preserve context near boundaries, but excessive overlap increases storage and embedding costs.

Context windows are also important in conversational agents. A conversation may exceed the
model's context budget. A context manager can retain the most recent turns, summarize older
turns, and retrieve relevant historical facts when a new question arrives. This is often called
a hybrid memory strategy because it combines recency, summarization, and retrieval.

Evaluation should measure more than response quality. Useful operational metrics include
document length, number of generated windows, average window size, overlap ratio, retrieval
latency, end-to-end latency, memory usage, and the proportion of answers supported by retrieved
evidence.

Production systems should treat model output as untrusted. They should validate inputs, set
timeouts, handle unavailable model servers, log failures, and test edge cases. Embedding models
and chat models serve different purposes: chat models generate and reason over text, while
embedding models map text into vectors for similarity search.

LangGraph can represent the pipeline as explicit state transitions. Each node performs one
responsibility, such as window creation, summarization, indexing, retrieval, or answer generation.
This makes the workflow observable, testable, and easier to extend.
