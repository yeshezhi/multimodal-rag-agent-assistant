from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse

from .config import get_settings
from .agent import EnterpriseAgent
from .documents import parse_document
from .observability import ObservabilityStore, load_evaluation_report
from .rag import RagService
from .schemas import (
    ChatRequest,
    ChatResponse,
    AgentRequest,
    AgentResponse,
    DocumentListResponse,
    DocumentSummary,
    EvaluationSummary,
    ImageIngestResponse,
    ImageKnowledgeBaseStatus,
    ImageSearchResponse,
    ImageSearchResult,
    IngestResponse,
    KnowledgeBaseStatus,
    MultimodalImageResponse,
    MultimodalRagResponse,
    ObservabilitySummary,
)


app = FastAPI(title="企业知识库 RAG", version="0.1.0")
STATIC_DIR = Path(__file__).parent / "static"
UNIVERSITY_SATELLITE_DIR = Path("/home/cjy/project/Sample4Geo_copy/dataset/U1652/University-Release/train/satellite")


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    files_schema = schema["components"]["schemas"][
        "Body_ingest_documents_api_v1_documents_post"
    ]["properties"]["files"]
    files_schema.pop("format", None)
    files_schema["items"].pop("contentMediaType", None)
    files_schema["items"]["format"] = "binary"
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/v1/university-satellite/{location_id}", include_in_schema=False)
def university_satellite_image(location_id: str) -> FileResponse:
    if not location_id.isdigit() or len(location_id) != 4:
        raise HTTPException(status_code=404, detail="Image not found.")
    path = UNIVERSITY_SATELLITE_DIR / location_id / f"{location_id}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, media_type="image/jpeg")


@lru_cache
def get_service() -> RagService:
    return RagService(get_settings())


@lru_cache
def get_agent() -> EnterpriseAgent:
    return EnterpriseAgent(get_service())


@lru_cache
def get_observability() -> ObservabilityStore:
    return ObservabilityStore(get_settings().rag_data_dir)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/api/v1/documents",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {"properties": {"files": {"items": {"format": "binary"}}}}
                }
            }
        }
    },
)
async def ingest_documents(
    files: Annotated[
        list[UploadFile],
        File(
            description="上传 PDF、DOCX、Markdown 或 TXT 文件。",
            json_schema_extra={"format": "binary"},
        ),
    ],
) -> IngestResponse:
    settings = get_settings()
    documents = []
    for file in files:
        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{file.filename} 超过上传限制。")
        try:
            documents.extend(parse_document(file.filename or "unnamed", content))
        except ValueError as error:
            raise HTTPException(status_code=415, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=422, detail=f"无法解析 {file.filename}：{error}") from error

    service = get_service()
    try:
        chunks_added = service.ingest(documents)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    total, _ = service.status()
    return IngestResponse(files_processed=len(files), chunks_added=chunks_added, knowledge_base_chunks=total)


@app.post("/api/v1/images", response_model=ImageIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_images(files: Annotated[list[UploadFile], File(...)]) -> ImageIngestResponse:
    settings = get_settings()
    accepted_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    images: list[tuple[str, bytes]] = []
    for file in files:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in accepted_suffixes:
            raise HTTPException(status_code=415, detail="Only JPG, JPEG, PNG and WEBP images are supported.")
        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"{file.filename} exceeds upload limit.")
        images.append((file.filename or "unnamed-image", content))
    try:
        processed = get_service().ingest_images(images)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    count, _ = get_service().image_status()
    return ImageIngestResponse(files_processed=processed, image_index_size=count)


@app.post("/api/v1/image-search", response_model=ImageSearchResponse)
async def image_search(file: Annotated[UploadFile, File(...)], top_k: int = 5) -> ImageSearchResponse:
    if not 1 <= top_k <= 15:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 15.")
    try:
        results = get_service().search_images(await file.read(), top_k)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ImageSearchResponse(results=[ImageSearchResult(**result.__dict__) for result in results])


@app.post("/api/v1/multimodal-image-query", response_model=MultimodalImageResponse)
async def multimodal_image_query(
    file: Annotated[UploadFile, File(...)],
    prompt: Annotated[str, Form()] = "请简要描述这张图片中可见的内容。",
    top_k: Annotated[int, Form()] = 5,
) -> MultimodalImageResponse:
    if not 1 <= top_k <= 15:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 15.")
    try:
        analysis, matches = get_service().analyze_image(await file.read(), prompt, top_k)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return MultimodalImageResponse(
        analysis=analysis,
        matches=[ImageSearchResult(**match.__dict__) for match in matches],
    )


@app.post("/api/v1/multimodal-rag-query", response_model=MultimodalRagResponse)
async def multimodal_rag_query(
    file: Annotated[UploadFile, File(...)],
    question: Annotated[str, Form()],
    top_k: Annotated[int, Form()] = 5,
) -> MultimodalRagResponse:
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty.")
    if not 1 <= top_k <= 15:
        raise HTTPException(status_code=422, detail="top_k must be between 1 and 15.")
    try:
        answer, citations, matches = get_service().multimodal_rag(
            await file.read(), question.strip(), top_k
        )
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return MultimodalRagResponse(
        answer=answer,
        citations=[get_service()._citation(item) for item in citations],
        matches=[ImageSearchResult(**item.__dict__) for item in matches],
    )


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started_at = perf_counter()
    try:
        response, trace = get_service().chat_with_trace(request.question, request.top_k)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    get_observability().record(
        {
            "route": "knowledge_base",
            "question": request.question.strip()[:300],
            "latency_ms": round((perf_counter() - started_at) * 1000, 1),
            "outcome": "answered" if response.citations else "refused",
            "candidate_count": trace.get("candidate_count", 0),
            "citations": [item.source for item in response.citations],
        }
    )
    return response


@app.post("/api/v1/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    started_at = perf_counter()
    result = get_agent().invoke(request.question)
    response = AgentResponse(route=result["route"], answer=result["answer"])
    get_observability().record(
        {
            "route": f"agent:{response.route}",
            "question": request.question.strip()[:300],
            "latency_ms": round((perf_counter() - started_at) * 1000, 1),
            "outcome": "refused" if "无法确定" in response.answer else "answered",
            "candidate_count": 0,
            "citations": [],
        }
    )
    return response


@app.post("/api/v1/agent/image", response_model=MultimodalRagResponse)
async def image_agent(file: Annotated[UploadFile, File(...)], question: Annotated[str, Form()]) -> MultimodalRagResponse:
    result = get_agent().invoke_image(await file.read(), question)
    return MultimodalRagResponse(answer=result["answer"], citations=[get_service()._citation(item) for item in result["citations"]], matches=[ImageSearchResult(**item.__dict__) for item in result["matches"]])


@app.get("/api/v1/knowledge-base", response_model=KnowledgeBaseStatus)
def knowledge_base_status() -> KnowledgeBaseStatus:
    chunks, dimension = get_service().status()
    return KnowledgeBaseStatus(chunks=chunks, vector_dimension=dimension)


@app.get("/api/v1/image-knowledge-base", response_model=ImageKnowledgeBaseStatus)
def image_knowledge_base_status() -> ImageKnowledgeBaseStatus:
    images, dimension = get_service().image_status()
    return ImageKnowledgeBaseStatus(images=images, vector_dimension=dimension)


@app.get("/api/v1/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    documents = [
        DocumentSummary(source_name=source_name, chunks=chunks)
        for source_name, chunks in get_service().list_documents()
    ]
    return DocumentListResponse(documents=documents)


@app.delete("/api/v1/documents/{source_name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(source_name: str) -> None:
    deleted = get_service().delete_document(source_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="未找到指定文档。")


@app.delete("/api/v1/knowledge-base", status_code=status.HTTP_204_NO_CONTENT)
def clear_knowledge_base() -> None:
    get_service().store.clear()


@app.get("/api/v1/observability/summary", response_model=ObservabilitySummary)
def observability_summary() -> ObservabilitySummary:
    return ObservabilitySummary(**get_observability().summary())


@app.get("/api/v1/evaluation/summary", response_model=EvaluationSummary)
def evaluation_summary() -> EvaluationSummary:
    report = load_evaluation_report(get_settings().rag_data_dir)
    if report is None:
        return EvaluationSummary(available=False)
    return EvaluationSummary(available=True, **report)
