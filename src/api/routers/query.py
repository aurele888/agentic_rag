"""RAG query endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage, AIMessage
import uuid

from ..dependencies import get_db, get_rag_service, get_llm_service
from ..models import QueryRequest, QueryResponse, SourceInfo
from ..services.rag_service import RAGService
from ..services.llm_service import LlmService

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_pdfs(
    request: QueryRequest,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
    llm_service: LlmService = Depends(get_llm_service)
):
    """Query across PDFs with source attribution."""
    import logging
    logger = logging.getLogger(__name__)

    logger.info(f"📥 Received query request: question='{request.question[:50]}...', model={request.model}")

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"🔑 Session ID: {session_id}")

    # USE REDIS CACHING FOR FASTER OPERATION
    messages = rag_service.get_session_messages(session_id, db, 6).all()
   
    prune_chat_history = [
            AIMessage(content=messages[r_index].content)
            if messages[r_index].role == "assistant" else HumanMessage(content = messages[r_index].content)
            for r_index in range(-1, -len(messages)-1, -1) if messages[r_index].role != "system"
        ]
    logger.info(f"Current context: {prune_chat_history}")
    
    # Save user message --- USE REDIS CACHING FOR FAST OPERATION
    rag_service.save_message(
        session_id=session_id,
        role="user",
        content=request.question,
        sources=None,
        db=db
    )
    logger.info("💾 User message saved")


    # Query RAG
    logger.info("🚀 Starting RAG query...")
    try:
        # Create the LLM service instance 
        llm_service.set_ollama_model(request.model)
        answer, sources, reasoning_steps = rag_service.query_multi_pdf(
            question=request.question,
            llm_service=llm_service,
            chat_context = prune_chat_history,
            pdf_ids=request.pdf_ids,
            db=db
        )

        logger.info(f"✅ RAG query complete: answer_length={len(answer)}, sources_count={len(sources)}, reasoning_steps={len(reasoning_steps)}")
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower() and "404" in error_msg:
            logger.error(f"❌ Model not found: {request.model}")
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' not found. Please select a different model from the dropdown or install it with: ollama pull {request.model}"
            )
        logger.error(f"❌ Query failed: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Query failed: {error_msg}")

    # Save assistant message
    message = rag_service.save_message(
        session_id=session_id,
        role="assistant",
        content=answer,
        sources=sources,
        db=db
    )
    logger.info(f"💾 Assistant message saved with ID: {message.message_id}")

    response = QueryResponse(
        answer=answer,
        sources=[SourceInfo(**s) for s in sources],
        metadata={
            "model_used": request.model,
            "chunks_retrieved": len(sources),
            "pdfs_queried": len(set(s["pdf_id"] for s in sources)),
            "reasoning_steps": reasoning_steps
        },
        session_id=session_id,
        message_id=message.message_id
    )

    logger.info(f"📤 Returning response: answer_length={len(response.answer)}, sources={len(response.sources)}")
    logger.info(f"📊 First 200 chars of answer: {response.answer[:200]}")

    return response

@router.get("/sessions/{session_id}/messages")
def get_session_messages(
    session_id: str,
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get chat history for a session."""
    messages = rag_service.get_session_messages(session_id, db)
    return [
        {
            "message_id": msg.message_id,
            "role": msg.role,
            "content": msg.content,
            "sources": msg.sources,
            "timestamp": msg.timestamp
        }
        for msg in messages
    ]
