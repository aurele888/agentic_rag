"""RAG query service."""

from typing import List, Dict, Tuple, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import logging, os, copy

from langchain_ollama import ChatOllama
import ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from types import SimpleNamespace
from fastapi import Depends


from ..config import settings
import concurrent.futures
from langchain_core.prompts import (
    ChatPromptTemplate, 
    PromptTemplate, 
    HumanMessagePromptTemplate)

from ..database import DocumentMetadata, ChatSession, ChatMessage
from ..config import settings
from  ...core.prompt import PromptManager
from ..services.query_router_service import QueryRouterService
from ..services.llm_service import LlmService


logger =logging.getLogger(__name__)


class RAGService:
    """Service for RAG operations."""

    prompt_manager : PromptManager = PromptManager()

    def __init__(self, query_router_service: QueryRouterService):
        """Initialize RAG service."""
        self.persist_directory = settings.VECTOR_DB_DIR
        self.llm_service = None
        self.query_router_service = query_router_service
    
           
    def retrieve_doc(self, pdf_id, pdf_meta_data, QUERY_PROMPT, question):
        """
        Retrieve relevant chunks from a single PDFs documents with source attribution.
        return pairs of retrieved chunks and query, corresponding source, and all retrived chucks
        """

        # Retrieve from current PDF collections
        curr_retrieved_docs = []
        curr_querry_docs_pairs =[]
        curr_sources = []

        vector_db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=OllamaEmbeddings(model=settings.EMBEDDING_MODEL),
            collection_name=pdf_meta_data.collection_name
        )

        # Setup Multi querry retriever
        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(search_kwargs={"k": 3}),
            self.llm_service.llm,
            prompt=QUERY_PROMPT
        )
        try:
            if settings.USE_HYBRID_SEARCH:
                logger.info(f"Initialising the hybrid search strategy ...")
                
                # Setup BM25 Retriever (Keyword search)
                bm25_retriever = BM25Retriever.from_documents(
                    [Document(page_content=doc, metadata=metadata) for doc, metadata 
                        in zip(vector_db.get()['documents'], vector_db.get()['metadatas'])])
                bm25_retriever.k = 3 

                # Weighted fusion of the retrievers a.k.a hybrid search
                ensemble_retriever = EnsembleRetriever(
                            retrievers=[bm25_retriever, retriever],
                            weights=[settings.KEYWORDS_SIMILARITY_WEIGHT, settings.SEMANTIC_SIMILARITY_WEIGHT]
                        )
                docs = ensemble_retriever.invoke(question)
            else:
                docs = retriever.invoke(question)

            logger.info(f"Retrieved {len(docs)} documents chunks from {pdf_meta_data.name}")
            
            # Ensuring metadata is collected for referencing
            for doc in docs:
                if "pdf_name" not in doc.metadata:
                    doc.metadata["pdf_name"] = pdf_meta_data.name
                if "pdf_id" not in doc.metadata:
                    doc.metadata["pdf_id"] = pdf_id
                if settings.USE_RE_RANKER:
                    # Temporaly separately  store querry_doc_pairs and sources to match the re-ranker input format
                    curr_querry_docs_pairs.append((question, doc.page_content))
                    curr_sources.append(doc.metadata)
            curr_retrieved_docs.extend(docs)
            
        except Exception as e:
            logger.warning(f"Error retrieving from {pdf_meta_data.name}: {e}")

        return curr_querry_docs_pairs, curr_sources, curr_retrieved_docs
    

    def retrieve_all_doc(
        self,
        question: str,
        pdfs: List[SimpleNamespace],
        router_file_match: List,
    ): 
        """
        Retrieve relevant chunks across multiple PDFs documents with source attribution.
        return retrieved PDFs chunk and a formatted bundel of chunk and sources
        """
        
        logger.info(f"Processing question across {len(pdfs)} PDFs: {question}")
        logger.info(f"Router match consists of : {len(router_file_match)} PDFs")
        # Query prompt for query expansion. Chat history is appended for query contextualization
        template = RAGService.prompt_manager.MULTI_QUERY_PROMPT_TEMPLATE
        
        QUERY_PROMPT = PromptTemplate.from_template(template)
        
        # Retrieve from ALL PDF collections
        all_retrieved_docs = []
        all_querry_docs_pairs = []
        all_sources = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                self.retrieve_doc, 
                pdf_meta_data.pdf_id, 
                pdf_meta_data,
                QUERY_PROMPT, 
                question) : pdf_meta_data.pdf_id for pdf_meta_data in pdfs if pdf_meta_data.pdf_id+"_"+pdf_meta_data.name in router_file_match
                }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results = future.result()
                    all_querry_docs_pairs.extend(results[0])
                    all_sources.extend(results[1])
                    all_retrieved_docs.extend(results[2])
                except Exception as error_msg:
                    logger.error(f"Unexpected error occurs with {futures[future]} which is cause by :{error_msg}")

        logger.info(f"Total documents chunks retrieved: {len(all_retrieved_docs)}")

        # Initialize context (doc + source) formating variable
        context_parts = []

        # Elo Re-rankers to re-rank the retrieved documents (work best on GPU, very slow on CPU)
        if settings.USE_RE_RANKER:
            re_rank_documents = []
            self.llm_service.set_cross_encoder_model()
            logger.info(f"Re-ranking the {len(all_querry_docs_pairs)} retreived documents !")
            if self.llm_service.cross_encoder_model:
                re_rank_documents = self.llm_service.cross_encoder_model.get_ranked_document(all_querry_docs_pairs, all_sources)
            for score, doc, source in re_rank_documents:
                if score >= settings.RE_RANKER_THRESHOLD:
                    context_parts.append(f"[Source: {source.get('pdf_name', 'Unknown')}]\n{doc}\n")
        else:
            # Keep Top-k chunks
            for doc in all_retrieved_docs if len(all_retrieved_docs) < settings.TOP_K_CHUNK else all_retrieved_docs[:settings.TOP_K_CHUNK]: 
                source = doc.metadata.get("pdf_name", "Unknown")
                context_parts.append(f"[Source: {source}\n{doc.page_content}\n")
        return  "\n---\n".join(context_parts), all_retrieved_docs
    

    # TO-DO : Record the reasonning process
    def query_multi_pdf(
        self,
        question: str,
        llm_service: LlmService,
        chat_context: List,
        pdf_ids: Optional[List[str]],
        db: Session
    ) -> Tuple[str, List[Dict], List[str]]:
        """
        Process the user query by acessing all available documents
        return assistant/AI response asynchonously
        """

        reasoning_steps = []
        self.llm_service = llm_service
        self.prune_chat_context = chat_context

        # TO-DO : Scale this process to handle million of documents
        query = db.query(DocumentMetadata)
        if pdf_ids:
            query = query.filter(DocumentMetadata.pdf_id.in_(pdf_ids))
        pdfs = query.all()
        if not pdfs:
            return "No PDFs found to query.", [], []
        
        pdfs = [SimpleNamespace(**{"pdf_id": pdf.pdf_id, 
                  "name":pdf.name, 
                  "collection_name": pdf.collection_name
                  }) for pdf in pdfs]
        
        reasoning_steps.append(f"🤖 Using model: {self.llm_service.llm.name}")
        reasoning_steps.append(f"📚 Searching across {len(pdfs)} PDF(s): {', '.join([p.name for p in pdfs])}")
        
        # In the case a pdf is explicitly attached directly use it to answer the question
        # TO-DO write the ifelse statement here
        
        formatted_context, all_retrieved_docs = None, None
        try:
            # Routing the user query
            router_response = self.query_router_service.route(question, db)

            if 'internal_docs' in router_response['route']:
                
                match_files_path = [os.path.basename(path) for path in router_response['matches']]
                logger.info(f"Router has matched '{question}' to these PDFs: {match_files_path}")
                reasoning_steps.append(f" 📚 Narrowing down the search to : {match_files_path}")
                logger.info(f"'{question}' is clasified as a retrival query.")
                formatted_context, all_retrieved_docs = self.retrieve_all_doc(question, pdfs, match_files_path)
                
                # RAG prompt with source awareness
                router_category_template = RAGService.prompt_manager.ROUTER_RETRIEVAL_PROMPT_TEMPLATE
                logger.info("Generating response with source attribution!")
            else:
                reasoning_steps.append(f" 📚 No internal document is relevant to the query")
                logger.info(f"'{question}' is clasified as a general knowledge query.")
                router_category_template = RAGService.prompt_manager.ROUTER_CASUAL_PROMPT_TEMPLATE
                logger.info("Generating simple response!")

            
            self.prune_chat_context.append(HumanMessagePromptTemplate.from_template(router_category_template))
            chat_context.append(HumanMessage(content=question))
            decision_chain = ChatPromptTemplate.from_messages(self.prune_chat_context) | self.llm_service.llm

            response_chain = (
                {"context": lambda x: formatted_context, "question": lambda x: x["question"]} 
                | decision_chain
                | StrOutputParser()
                )
            
            # Returning the assistance response
            # response = ""
            # for chunk in response_chain.stream({"question": question}):
            #         response+= chunk
            #         yield chunk

            response = response_chain.invoke({"question": question})
        except Exception as e_message:

            logger.error(f"Unexpected error ocuurs : {e_message}")
        
        source_details = [
                    {
                        "pdf_name": doc.metadata.get("pdf_name"),
                        "pdf_id": doc.metadata.get("pdf_id"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "chunk_value": doc.page_content
                    }
                    for doc in all_retrieved_docs
                ] if all_retrieved_docs else []
        
        # self.response = response
        # self.sources = source_details
        return response, source_details, []
        

    def query_multi_pdf_old(
        self,
        question: str,
        model: str,
        pdf_ids: Optional[List[str]],
        db: Session
    ) -> Tuple[str, List[Dict], List[str]]:
        """Query across multiple PDFs with source attribution.

        Args:
            question: User question
            model: LLM model to use
            pdf_ids: List of PDF IDs to query (None = all PDFs)
            db: Database session

        Returns:
            Tuple of (answer, sources, reasoning_steps)
        """
        reasoning_steps = []

        # Get PDF metadata
        query = db.query(DocumentMetadata)
        if pdf_ids:
            query = query.filter(DocumentMetadata.pdf_id.in_(pdf_ids))
        pdfs = query.all()

        if not pdfs:
            return "No PDFs found to query.", [], []

        reasoning_steps.append(f"📚 Searching across {len(pdfs)} PDF(s): {', '.join([p.name for p in pdfs])}")

        # Initialize LLM
        llm = ChatOllama(model=model)
        reasoning_steps.append(f"🤖 Using model: {model}")

        # Query prompt for multi-query retriever
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}"""
        )

        reasoning_steps.append("🔍 Generating alternative search queries...")

        # Retrieve from all collections
        all_docs = []
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        for pdf in pdfs:
            vector_db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings,
                collection_name=pdf.collection_name
            )

            retriever = MultiQueryRetriever.from_llm(
                vector_db.as_retriever(search_kwargs={"k": 3}),
                llm,
                prompt=QUERY_PROMPT
            )

            try:
                reasoning_steps.append(f"📄 Retrieving from: {pdf.name}")
                # Use invoke instead of deprecated get_relevant_documents
                docs = retriever.invoke(question)
                # Ensure metadata is present
                for doc in docs:
                    if "pdf_name" not in doc.metadata:
                        doc.metadata["pdf_name"] = pdf.name
                    if "pdf_id" not in doc.metadata:
                        doc.metadata["pdf_id"] = pdf.pdf_id
                all_docs.extend(docs)
                reasoning_steps.append(f"✅ Found {len(docs)} relevant chunks in {pdf.name}")
            except Exception as e:
                reasoning_steps.append(f"⚠️ Error retrieving from {pdf.name}: {str(e)}")
                print(f"Error retrieving from {pdf.name}: {e}")

        reasoning_steps.append(f"📊 Total chunks retrieved: {len(all_docs)}")

        # Format context with source labels
        context_parts = []
        for doc in all_docs[:10]:
            source = doc.metadata.get("pdf_name", "Unknown")
            context_parts.append(f"[Source: {source}]\n{doc.page_content}\n")

        formatted_context = "\n---\n".join(context_parts)
        reasoning_steps.append(f"🔗 Using top {min(len(all_docs), 10)} chunks for context")

        # RAG prompt template with chain-of-thought
        template = """Answer the question based ONLY on the following context from multiple PDF documents.
        Each section is marked with its source document.

        Use chain-of-thought reasoning:
        1. First, identify which parts of the context are relevant to the question
        2. Analyze the information from each source document
        3. Synthesize the information to form a comprehensive answer
        4. Ensure you cite the source document name for each piece of information
        5. If information comes from multiple sources, mention all relevant sources
        6. If sources contradict, note the discrepancy and cite both sources

        Context:
        {context}

        Question: {question}

        Think step-by-step and provide your answer with source citations:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": lambda x: formatted_context, "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

        reasoning_steps.append("💭 Generating answer with source citations...")

        # Check if model supports thinking (e.g., qwen3, deepseek-r1)
        thinking_models = ['qwen3', 'deepseek-r1', 'qwen', 'deepseek']
        supports_thinking = any(tm in model.lower() for tm in thinking_models)

        if supports_thinking:
            reasoning_steps.append("🧠 Using thinking-enabled model with chain-of-thought reasoning...")
            try:
                # Enhanced system message for chain-of-thought reasoning
                cot_system_message = f"""You are an expert AI assistant that uses chain-of-thought reasoning.

                        Answer the question based ONLY on the provided context from PDF documents.

                        CHAIN-OF-THOUGHT PROCESS:
                        1. **Read and understand** the question carefully
                        2. **Scan the context** to identify all relevant information
                        3. **Break down** the information by source document
                        4. **Analyze** how each piece relates to the question
                        5. **Synthesize** a comprehensive answer
                        6. **Cite sources** explicitly for every claim

                        Context from PDF documents:
                        {formatted_context}

                        Think through each step carefully, showing your reasoning process."""

                # Use Ollama client directly for thinking-capable models
                ollama_response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": cot_system_message},
                        {"role": "user", "content": f"Question: {question}\n\nThink step-by-step and provide a detailed answer with source citations."}
                    ],
                    think=True,
                    stream=False
                )

                # Add thinking process to reasoning steps
                if hasattr(ollama_response.message, 'thinking') and ollama_response.message.thinking:
                    thinking_text = ollama_response.message.thinking
                    # Show more of the thinking process (500 chars instead of 200)
                    reasoning_steps.append(f"💡 Model's chain-of-thought:\n{thinking_text[:500]}{'...' if len(thinking_text) > 500 else ''}")

                response = ollama_response.message.content
            except Exception as e:
                print(f"Error using thinking mode, falling back to standard: {e}")
                response = chain.invoke(question)
        else:
            response = chain.invoke(question)

        # Extract source information
        sources = [
            {
                "pdf_name": doc.metadata.get("pdf_name"),
                "pdf_id": doc.metadata.get("pdf_id"),
                "chunk_index": doc.metadata.get("chunk_index", 0)
            }
            for doc in all_docs[:10]
        ]

        reasoning_steps.append("✨ Answer generated successfully!")

        return response, sources, reasoning_steps

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]],
        db: Session
    ) -> ChatMessage:
        """Save chat message to database.

        Args:
            session_id: Chat session identifier
            role: Message role (user or assistant)
            content: Message content
            sources: Source documents (for assistant messages)
            db: Database session

        Returns:
            Saved chat message
        """
        # Ensure session exists
        session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if not session:
            session = ChatSession(
                session_id=session_id,
                created_at=datetime.now(),
                last_active=datetime.now()
            )
            db.add(session)
        else:
            session.last_active = datetime.now()

        # Save message
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources,
            timestamp=datetime.now()
        )
        db.add(message)
        db.commit()
        db.refresh(message)

        return message



    def get_session_messages(self, session_id: str, db: Session, limit_count: int =-1) -> List[ChatMessage]:
        """Get all messages for a session.

        Args:
            session_id: Chat session identifier
            db: Database session

        Returns:
            List of chat messages
        """
        if limit_count == -1:
            return db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp).all()
        else:
            return db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.timestamp.desc()).limit(limit_count)
        
    
    

