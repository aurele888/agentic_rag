"""FastAPI dependencies for dependency injection."""
from sqlalchemy.orm import Session
from .database import SessionLocal, engine
from .services.document_service import DocumentService
from .services.rag_service import RAGService
from .services.llm_service import LlmService
from .services.query_router_service import QueryRouterService
from fastapi import Depends

def get_db():
    """Database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_llm_service() -> LlmService:
    """LLM service dependency."""
    return LlmService()

def get_query_router_service(
        llm_service: LlmService = Depends(get_llm_service)
        ) -> QueryRouterService:
    return QueryRouterService(llm_service=llm_service)


def get_document_service(
        llm_service: LlmService = Depends(get_llm_service)
) -> DocumentService:
    """Docuemt service dependency."""
    return DocumentService(llm_service=llm_service)


def get_rag_service(
        query_router_service: QueryRouterService = Depends(get_query_router_service)
) -> RAGService:
    """RAG service dependency."""
    return RAGService(query_router_service=query_router_service)


