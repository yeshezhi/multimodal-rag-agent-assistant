from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    files_processed: int
    chunks_added: int
    knowledge_base_chunks: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    top_k: int | None = Field(default=None, ge=1, le=15)


class Citation(BaseModel):
    chunk_id: str
    source: str
    location: str
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]


class KnowledgeBaseStatus(BaseModel):
    chunks: int
    vector_dimension: int | None


class DocumentSummary(BaseModel):
    source_name: str
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class ImageIngestResponse(BaseModel):
    files_processed: int
    image_index_size: int


class ImageSearchResult(BaseModel):
    image_id: str
    source_name: str
    score: float


class ImageSearchResponse(BaseModel):
    results: list[ImageSearchResult]


class ImageKnowledgeBaseStatus(BaseModel):
    images: int
    vector_dimension: int | None


class MultimodalImageResponse(BaseModel):
    analysis: str
    matches: list[ImageSearchResult]


class MultimodalRagResponse(BaseModel):
    answer: str
    citations: list[Citation]
    matches: list[ImageSearchResult]


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)


class AgentResponse(BaseModel):
    route: str
    answer: str


class QueryEvent(BaseModel):
    timestamp: str
    route: str
    question: str
    latency_ms: float
    outcome: str
    candidate_count: int = 0
    citations: list[str] = Field(default_factory=list)


class ObservabilitySummary(BaseModel):
    total_queries: int
    answered_queries: int
    refused_queries: int
    average_latency_ms: float | None
    recent: list[QueryEvent]


class EvaluationSummary(BaseModel):
    available: bool
    generated_at: str | None = None
    top_k: int | None = None
    total_cases: int | None = None
    passed: int | None = None
    pass_at_k: float | None = None
    mrr: float | None = None
