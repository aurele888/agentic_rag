import sqlite3
import logging
import json
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import  RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import numpy as np
from keybert import KeyBERT
import sqlalchemy
from datetime import datetime
from functools import lru_cache
from ..services.llm_service import LlmService
from ...core.prompt import PromptManager
from ...core.utils import detect_device
from ..config import settings


logger = logging.getLogger(__name__)


# TO-DO : Add to ingest_document() method,  a logic for the file type (.pdf, .xlsx, or other free forms textual documents)

class QueryRouterService:
    """
    A simple query router that decides if a given query requires specific internal document 
    or a general knowledge to be answered.
    """

    prompt_manager : PromptManager = PromptManager()

    # MOVE THE query router TABLE INIT TO the Digection part of the code
    def __init__(self, llm_service: LlmService):
        self.db_path = settings.DATABASE_URL
        self.model_name = settings.DEFAULT_QUERY_ROUTER_MODEL
        self.max_workers = settings.DEFAULT_QUERY_ROUTER_MAX_WORKERS

        # Default: BM25 scores are negative; -1.0 is a solid match
        self.threshold = settings.DEFAULT_QUERY_ROUTER_THRESHOLD  
        
        # TO-DO Run the loading asyncronously 
        # Initialize with structured output for fast parsing
        self.query_gen_model = llm_service.set_ollama_model(
            model_name = settings.DEFAULT_QUERY_ROUTER_MODEL,
            temperature = settings.DEFAULT_QUERY_ROUTER_TEMPERATURE,
            top_p = settings.DEFAULT_QUERY_ROUTER_TOP_P,
            top_k =  settings.DEFAULT_QUERY_ROUTER_TOP_K,
        )
        
        self.sentence_model = llm_service.set_sentence_tranformer_model()
        logger.info("Successfully initialised the Query Router class and its models!")

    
    def tune_threshold(self):
        """Analyzes historical logs to find the optimal BM25 threshold."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get scores for queries that were routed to internal_docs
            cursor.execute("SELECT score FROM query_logs WHERE route = 'internal_docs' AND score < 0")
            scores = [row[0] for row in cursor.fetchall()]
        
        if not scores:
            logger.warning("Not enough data to tune. Keeping default threshold.")
            return self.threshold

        # Calculate the 75th percentile (or mean) of successful scores
        # We want a threshold that captures most successful hits but filters noise
        new_threshold = np.percentile(scores, 75) 
        self.threshold = new_threshold
        logger.info(f"Threshold tuned to: {self.threshold:.4f}")
        return self.threshold


    @lru_cache(maxsize=1000)
    def _extract_keywords_cached(self, query):
        """Skip the LLM step for identical queries via in-memory caching."""
        return self._extract_query_keywords(query)

    def _extract_query_keywords(self, query):
        """ Extract keywords form the user query"""
        DEFAULT_QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=QueryRouterService.prompt_manager.ROUTER_PROMPT_TEMPLATE,
        )
        query_chain = (
            {'question':  RunnablePassthrough()}
            | DEFAULT_QUERY_PROMPT
            | self.query_gen_model
            | StrOutputParser()
        )
        response = query_chain.invoke(query)
        logger.info(f"Generated queries:\n {response}")
        kw_model = KeyBERT(model=self.sentence_model)
        keywords = kw_model.extract_keywords(response, stop_words=None)
        return list({k for k,score in keywords})
    

    def route(self, query: str, db: Session):
        """Routes using BM25 ranking for sub-millisecond keyword lookup via SQLAlchemy."""
    
        q_terms = self._extract_keywords_cached(query)
        search_query = " OR ".join(q_terms)
        matches = []
        best_score = 0.0
        route = "general_knowledge"

        # Use a single session to handle both reading and writing
        with db as session:
            try:
                if q_terms:
                    # 1. Execute the BM25 Search
                    search_stmt = sqlalchemy.text('''
                        SELECT filename, bm25(document_index) as score 
                        FROM document_index 
                        WHERE document_index MATCH :search_query 
                        ORDER BY score ASC LIMIT 3
                    ''')
                    
                    results = session.execute(search_stmt, {"search_query": search_query}).all()
                    # SQLAlchemy returns Row objects which can be accessed like tuples or via keys
                    if results:
                        matches = [row.filename for row in results]
                        best_score = results[0].score

                # Use a relevance threshold: only route to docs if the score is strong (e.g. < -0.5)
                logger.info(f"matches:{matches} --- best_score:{best_score}")
                if matches and best_score < self.threshold:
                    route = "internal_docs"
                    
                # 2. Log analytics with the BM25 score inside an atomic transaction
                log_stmt = sqlalchemy.text('''
                    INSERT INTO query_logs (query, keywords, route, filename, score, timestamp) 
                    VALUES (:query, :keywords, :route, :filename, :score, :timestamp)
                ''')
                
                session.execute(log_stmt, {
                    "query": query,
                    "keywords": json.dumps(q_terms),
                    "route": route,
                    "filename": json.dumps(matches),
                    "score": best_score,
                    "timestamp": datetime.now()
                })
                session.commit()
                session.refresh()

            except SQLAlchemyError as se:
                # Crucial: Roll back database state to prevent connection pool corruption
                session.rollback()
                logger.error(f"Database error during search routing or logging: {se}", exc_info=True)
                # Fallback gracefully to general knowledge so your application keeps running
                route = "general_knowledge"
                matches = []
                best_score = 0.0
                
            except Exception as e:
                # Fallback block for serialization errors (json.dumps) or missing properties
                session.rollback()
                logger.error(f"Unexpected application error during routing workflow: {e}", exc_info=True)
                route = "general_knowledge"
                matches = []
                best_score = 0.0
        return {"route": route, "matches": matches, "relevance_score": best_score}


    # def route_old (self, query: str):
    #     """Routes using BM25 ranking for sub-millisecond keyword lookup."""
      
    #     q_terms = self._extract_keywords_cached(query)
    #     search_query = " OR ".join(q_terms)
    #     matches = []
    #     best_score = 0.0
    
    #     if q_terms:
    #         with sqlite3.connect(self.db_path) as conn:
    #             cursor = conn.cursor()
    #             cursor.execute('''
    #                 SELECT filename, bm25(document_index) as score 
    #                 FROM document_index 
    #                 WHERE document_index MATCH ? 
    #                 ORDER BY score ASC LIMIT 5
    #             ''', (search_query,))
    #             results = cursor.fetchall()
    #             if results:
    #                 matches = [r[0] for r in results]
    #                 best_score = results[0][1]

    #     # Use a relevance threshold: only route to docs if the score is strong (e.g. < -0.5)
    #     logger.info(f"matches:{matches} --- best_score:{best_score}")
    #     route = "internal_docs" if (matches and best_score < self.threshold) else "general_knowledge"
        
    #     # Log analytics with the BM25 score
    #     with sqlite3.connect(self.db_path) as conn:
    #         conn.execute("INSERT INTO query_logs (query, keywords, route, filename, score) VALUES (?, ?, ?, ?, ?)",
    #                     (query, json.dumps(q_terms), route, json.dumps(matches), best_score))
    #         conn.commit()
    #     return {"route": route, "matches": matches, "relevance_score": best_score}
    


if __name__== "__main__":
  
    q_router = QueryRouterService()
    q_router.batch_ingest("/Users/aurele/Desktop/test/")
    logger.info(q_router.route("What is instance normalisation ?"))
    
