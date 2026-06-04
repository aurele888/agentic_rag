"""Document processing functionality."""
import logging
from pathlib import Path
from typing import List
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredCSVLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    )
from enum import Enum
from langchain_text_splitters import RecursiveCharacterTextSplitter


logger = logging.getLogger(__name__)


class SupportedFileType (str, Enum):
    pdf = ".pdf"
    docx = ".docx"
    pptx = ".pptx"
    csv = ".csv"
    txt = ".txt"
    html = ".html"


class DocumentProcessor:
    """Handles PDF document loading and processing."""
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 80):
        # 7500 --- 100
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    
    def load_document(self, file_path: Path) -> List:
        
        if not file_path:
            return []
        
        extension = Path(file_path).suffix
        try:
            if extension == SupportedFileType.pdf:
                logger.info(f"Loading PDF from {file_path}")
                loader = UnstructuredPDFLoader(str(file_path))
            elif extension == SupportedFileType.docx:
                logger.info(f"Loading DOCX from {file_path}")
                loader = Docx2txtLoader(str(file_path))
            elif extension == SupportedFileType.pptx:
                logger.info(f"Loading pptx from {file_path}")
                loader = UnstructuredPowerPointLoader(str(file_path))
            elif extension == SupportedFileType.csv:
                logger.info(f"Loading csv from {file_path}")
                loader = UnstructuredCSVLoader(str(file_path))
            elif extension == SupportedFileType.txt:
                logger.info(f"Loading txt from {file_path}")
                loader = TextLoader(str(file_path))
            elif extension == SupportedFileType.html:
                logger.info(f"Loading html from {file_path}")
                loader = UnstructuredHTMLLoader(str(file_path))
            else:
                raise ValueError("Unsurpported file type.")
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading the document: {e}")
            raise


    def split_documents(self, documents: List) -> List:
        """Split documents into chunks."""
        try:
            logger.info("Splitting documents into chunks")
            return self.splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise 

    