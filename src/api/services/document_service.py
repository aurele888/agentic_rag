"""PDF processing service."""
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Optional
from fastapi import UploadFile
from sqlalchemy.orm import Session
from typing import Dict, List
import sqlalchemy, logging, json

# TO-DO: To be changed to alchemy
import sqlite3
from docling.document_converter import DocumentConverter
from langchain_community.document_loaders import PyPDFLoader

from ...core.documents import DocumentProcessor, SupportedFileType
from ...core.embeddings import VectorStore
from ..database import DocumentMetadata, engine
from ..services.llm_service import LlmService

from ..config import settings

import nltk
from keybert import KeyBERT
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction import text


logger = logging.getLogger(__name__)


class DocumentService:
    """Service for PDF operations."""

    def __init__(self, llm_service: LlmService):
        """Initialize PDF service."""
        self.sentence_model = llm_service.set_sentence_tranformer_model()
        self.doc_processor = DocumentProcessor(chunk_size=1500, chunk_overlap=80)
        self.vector_store = VectorStore(
            embedding_model= settings.EMBEDDING_MODEL,
            persist_directory=settings.VECTOR_DB_DIR
        )
        self.storage_dir = Path(settings.PDF_STORAGE_DIR)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.generic_words = stopwords.words('english')
        self.generic_words.extend(list(text.ENGLISH_STOP_WORDS))
        self._init_tables()


    def _init_tables(self):
        """Initializes the BM25-enabled virtual table and log table using SQLAlchemy."""
        with engine.begin() as conn:
            conn.execute(sqlalchemy.text('''
                CREATE VIRTUAL TABLE IF NOT EXISTS document_index 
                USING fts5(filename, keywords, tokenize='porter')
            '''))


    # TO-DO : Add a logic for the file type (.pdf, .xlsx, or other free forms textual documents)
    def extract_doc_keywords(self, file_path: str, db: Session) -> str:
        """
            Extracts keywords and upserts into the FTS5 virtual table.
        """

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"Ingesting document {file_path}")
        if len(pages) <=10:
            text = " ".join([p.page_content for p in pages if p])
        else:
            text = " ".join([p.page_content for p in (pages[:7] + pages[-3:]) if p])
        logger.info(f"Doc length : {len(text)}")
        kw_model = KeyBERT(model=self.sentence_model)
        keywords = set()
        left_id = 0
        logger.info("Extracting keywords from the document ...")
        for right_id in range(512 ,len(text), 432):          
            curr_keywords = kw_model.extract_keywords(
                text[left_id:right_id].strip().lower(), 
                stop_words= self.generic_words)
            
            logger.info(curr_keywords)
            [keywords.add(k) for k,score in curr_keywords if score >= 0.35]
            left_id = right_id - 80

        logger.info(f"I extracted {len(keywords)} keywords which are : {keywords}")
        keywords = json.dumps(list(keywords)).replace("[", "").replace("]", "").replace('"', "").replace(",", "").strip()
        logger.info(f"Parse keywords -- {keywords}")
        with db as session:
            # FTS5 tables don't support 'UNIQUE' constraints; we manually delete then insert
            session.execute(sqlalchemy.text("DELETE FROM document_index WHERE filename = :filename"),
                {"filename": file_path})
            session.execute(sqlalchemy.text("INSERT INTO document_index (filename, keywords) VALUES (:filename, :keywords)"),
                {"filename": file_path, "keywords": keywords})
            session.commit()
        return keywords
    

    def adaptive_excel_ingest(file_path: str, db_path: str) -> Dict[str, Dict[str, str]]:
        """
        Analyzes an Excel file during ingestion and dynamically routes it to either
        the SQL database or the text vector store based on its layout and context density.
        """

        xl = pd.ExcelFile(file_path)
        base_name = os.path.basename(file_path).split('.')[0].lower().replace(" ", "_")
        excel_data = {}
        

        # Use Doclin to convert
        converter = DocumentConverter()
        result = converter.convert("Master_Drawing_Register.xlsx")

        sheet_name_list = [sheet_name for sheet_name in xl.sheet_names]
        for sheet_name, table in zip(sheet_name_list, result.document.tables):
            df = table.export_to_dataframe()
            if df.empty:
                continue

            excel_data[str(sheet_name)] = {}
            # Clean header formatting
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

            # --- PHASE 1: STRUCTURAL ANALYSIS EVALUATION ---
            total_cells = df.size
            string_cells = df.select_dtypes(include=['object'])

            # Calculate the average character length of string values in the sheet
            if not string_cells.empty:
                avg_char_length = string_cells.map(lambda x: len(str(x))).mean().mean()
            else:
                avg_char_length = 0

            # Calculate the percentage of cells that are numeric data
            numeric_cells_count = df.select_dtypes(include=['number']).size
            numeric_ratio = numeric_cells_count / total_cells if total_cells > 0 else 0

            logger.info(f"Sheet '{sheet_name}' -> Avg Length: {avg_char_length:.1f} chars | Numeric Ratio: {numeric_ratio:.2%}")
            is_text_heavy = avg_char_length > 40 or (numeric_ratio < 0.05 and avg_char_length > 20)
            table_name = f"{base_name}_{sheet_name.lower()}"

            
            # Check if the dataframe columns are MultiIndex (nested subcolumns)
            if isinstance(df.columns, pd.MultiIndex):
                logger.info("Dealing with multiple columns ....")
                # Flatten the levels by joining them with an underscore, ignoring empty header levels
                df.columns = [
                    "_".join([str(level_val).strip() for level_val in col_tuple if pd.notna(level_val) and str(level_val).strip() != ""])
                    for col_tuple in df.columns.values
                ]

            else:
                # If it's a standard single-row header, just ensure column names are strings
                df.columns = [str(col).strip() for col in df.columns]

            col_summary = ", ".join(list(df.columns))
            meta_data = table_name.replace("_", " ")+ " " + col_summary +" "+ col_summary.replace("_", " ").replace("\n"," ")

            # Text heavy logic 
            if is_text_heavy:
                excel_data["type"] = "text_heavy"
                logger.info(f"Routing '{sheet_name}' to Vector Database (Text-Heavy Detect)...")

                # Convert rows into a natural sentence document structure for your vector pipeline
                text_documents_list = []
                for idx, row in df.iterrows():
                    # Convert the row array into a narrative string chunk that vector search loves
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if str(val).strip()])
                    text_documents_list.append(row_text)

                # Log placeholder to your central metadata catalog from Step 2
                excel_data[str(sheet_name)]["type"] = "text_heavy"
                excel_data[str(sheet_name)]["data"] = text_documents_list
                excel_data[str(sheet_name)]["meta_data"] = meta_data

            else:
                logger.info(f"Routing '{sheet_name}' to SQL Database (Structured Table Detect)...")
                    
                # Clean up column names to make them SQL-friendly (remove spaces/special characters if desired)
                df.columns = [f"{col}_{i}" if df.columns.duplicated()[i] else col for i, col in enumerate(df.columns)]
                df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
                df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
                df.dropna(axis=1, how='all', inplace=True)

                table_name = f"{base_name}_{sheet_name.lower()}"
                logger.info(df.columns)
                
                # TO-DO: Change this to sqlalchemy
                conn = sqlite3.connect(db_path)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                conn.close()

                col_summary = ", ".join(list(df.columns))
                logger.info(col_summary)
                excel_data[str(sheet_name)]["type"] = "num_heavy"
                excel_data[str(sheet_name)]["data"] = meta_data
                excel_data[str(sheet_name)]["meta_data"] = meta_data

        return excel_data


    def delete_from_db(self, file_path: str, db: Session) -> bool:
        """
            Removes a document and its keywords from the BM25 index and the documents table using SQLAlchemy.
        """
        if not file_path:
            return False
        try:
            with db as session:
                doc_index_result = session.execute(
                    sqlalchemy.text("DELETE FROM document_index WHERE filename = :filename"),
                    {"filename": file_path}
                )
                if doc_index_result.rowcount == 0:
                    logger.info(f"Document not found in index: {file_path}")

                doc_result = session.execute(
                    sqlalchemy.text("DELETE FROM documents WHERE file_path = :file_path"),
                    {"file_path": file_path}
                )
                if doc_result.rowcount == 0:
                    logger.info(f"Document not found in documents table: {file_path}")
                
                if not doc_index_result or not doc_result:
                    return False
                session.commit() 

            logger.info(f"Successfully removed from index: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False

    
    async def upload_and_process(
        self,
        file: UploadFile,
        db: Session
    ) -> DocumentMetadata:
        """Upload and process a PDF file.

        Args:
            file: Uploaded PDF file
            db: Database session

        Returns:
            PDFMetadata: Metadata for the processed PDF
        """

        # Generate unique ID
        pdf_id = self._generate_pdf_id(file.filename)
        # Save file
        file_path = self.storage_dir / f"{pdf_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process Document
        current_document_type = [file.name for file in SupportedFileType if file.value == Path(file_path.suffix)]
        documents = self.doc_processor.load_document(file_path)
        chunks = self.doc_processor.split_documents(documents)

    
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "pdf_id": pdf_id,
                "pdf_name": file.filename,
                "chunk_index": i,
                "source_file": file.filename
            })

        # Create vector DB collection
        collection_name = f"pdf_{abs(hash(file.filename + pdf_id))}"
        vector_db = self.vector_store.create_vector_db(
            documents=chunks,
            collection_name=collection_name
        )
        keywords = self.extract_doc_keywords(file_path, db)

        # Store metadata in database
        pdf_metadata = DocumentMetadata(
            pdf_id=pdf_id,
            name=file.filename,
            collection_name=collection_name,
            upload_timestamp=datetime.now(),
            doc_count=len(chunks),
            page_count=len(documents),
            is_sample=False,
            file_path=str(file_path),
            doc_type = current_document_type[0],
            keywords = keywords,
        )

        db.add(pdf_metadata)
        db.commit()
        db.refresh(pdf_metadata)
        return pdf_metadata
    

    def list_pdfs(self, db: Session) -> List[DocumentMetadata]:
        """List all PDFs.

        Args:
            db: Database session

        Returns:
            List of PDF metadata
        """
        return db.query(DocumentMetadata).all()

    def get_pdf(self, pdf_id: str, db: Session) -> Optional[DocumentMetadata]:
        """Get single PDF metadata.

        Args:
            pdf_id: PDF identifier
            db: Database session

        Returns:
            PDF metadata or None
        """
        return db.query(DocumentMetadata).filter(DocumentMetadata.pdf_id == pdf_id).first()

    def delete_pdf(self, pdf_id: str, db: Session) -> bool:
        """Delete PDF and its collection.

        Args:
            pdf_id: PDF identifier
            db: Database session

        Returns:
            True if deleted successfully, False otherwise
        """
        pdf = self.get_pdf(pdf_id, db)
        if not pdf:
            return False

        # Delete vector collection
        from langchain_community.vectorstores import Chroma
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(model = settings.EMBEDDING_MODEL)
        vector_db = Chroma(
            persist_directory=settings.VECTOR_DB_DIR,
            embedding_function=embeddings,
            collection_name=pdf.collection_name
        )
        vector_db.delete_collection()

        # Delete file if it exists
        if pdf.file_path and os.path.exists(pdf.file_path):
            os.remove(pdf.file_path)

        # Delete metadata from database
        self.delete_from_db(pdf.file_path, db)
        # db.delete(pdf)
        # db.commit()

        return True


    def _generate_pdf_id(self, filename: str) -> str:
        """Generate unique PDF ID.

        Args:
            filename: Original filename

        Returns:
            Unique PDF identifier
        """
        timestamp = datetime.now().isoformat()
        return f"pdf_{abs(hash(filename + timestamp))}"
