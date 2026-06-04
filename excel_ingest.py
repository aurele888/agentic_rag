
import os
import pandas as pd
import sqlite3
import logging
from keybert import KeyBERT
import json
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import torch
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from  hub_ingest import HubDigest
from pathlib import Path
import numpy as np
import re
from docling.document_converter import DocumentConverter



PERSIST_DIRECTORY = os.path.join("data", "vectors")
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)
logger = logging.getLogger(__name__)


# --- Example Pipeline Ingestion Logs ---
# When you ingest Excel sheets:
# register_file_metadata("excel", "q4_financials", "Contains columns: region, item_sold, revenue, cost, net_profit. Covers winter 2025 sales performance.")
# register_file_metadata("excel", "employee_directory", "Contains columns: employee_id, full_name, department, email, salary. Corporate staffing list.")

# # When you ingest PDFs to your Vector Database:
# register_file_metadata("pdf", "vector_collection_hr_policies", "The full corporate compliance handbook, parental leave terms, and remote work safety guidelines.")


# Set index database  name ---- rag_structured data
db_path = os.path.join(PERSIST_DIRECTORY, "documents.db") 

# Set the keyword extractor model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
sentence_model.to(torch.device("cuda"))
sentence_model.compile(dynamic=True)
max_workers = 4



def batch_ingest(folder_path):
    """Parallel ingestion using the pool."""
    if not folder_path:
        return
    
    _init_tables()
    files = [(os.path.join(folder_path, f), Path(f).suffix.replace(".", "")) for f in os.listdir(folder_path) if os.path.isfile(f)]
    files_dict = dict(files)
    logger.info(files_dict.values())
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        list(executor.map(ingest_document, files_dict.keys(), files_dict.values()))


def _init_tables():
        
        """Initializes the BM25-enabled virtual table and log table."""
        with sqlite3.connect(db_path) as conn:
            # FTS5 virtual table for BM25 search. 'porter' handles stems like 'running' -> 'run'
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS document_index 
                USING fts5(filename, keywords, tokenize='porter')
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    keywords TEXT,
                    route TEXT,
                    filename TEXT,
                    score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()


def ingest_document(file_path: str, file_type: str = None) -> str:

    """Extracts keywords and upserts into the FTS5 virtual table."""
    
    keywords = None

    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        logger.info(f"Ingesting document {file_path}")
        if len(pages) <=10:
            text = " ".join([p.page_content for p in pages if p])
        else:
            text = " ".join([p.page_content for p in (pages[:7] + pages[-3:]) if p])
        logger.info(f"Doc length : {len(text)}")
        kw_model = KeyBERT(model=sentence_model)
        keywords = set()
        left_id = 0
        logger.info("Extracting keywords from the document ...")
        
        # This code snippet is simulating an overlapping string of 80 characters
        for right_id in range(512 ,len(text), 432):          
            curr_keywords = kw_model.extract_keywords(text[left_id:right_id].strip().lower())
            logger.info(curr_keywords)
            [keywords.add(k) for k,score in curr_keywords if score >= 0.35]
            left_id = right_id - 80

        logger.info(f"I extracted {len(keywords)} keywords which are : {keywords}")
        keywords = json.dumps(list(keywords)).replace("[", "").replace("]", "").replace('"', "").replace(",", "").strip()
        logger.info(f"Parse keywords -- {keywords}")

    elif file_type == "xlsx" or file_type == "csv":

        excel_data = adaptive_excel_ingest(file_path)
        for sheet_name in excel_data:
            if excel_data[sheet_name]["type"] == "txt_heavy":

                logger.info(f"Processing {sheet_name} as an excel file with heavy text")
                text = excel_data[sheet_name]["data"]
                        
                logger.info("Extracting keywords from the document ...")
                # This part of the code can be transformed to a method as we call more than once
                kw_model = KeyBERT(model=sentence_model)
                keywords = set()
                left_id = 0
                
                # This code snippet is simulating an overlapping string of 80 characters
                for right_id in range(512 ,len(text), 432):          
                    curr_keywords = kw_model.extract_keywords(text[left_id:right_id].strip().lower())
                    logger.info(curr_keywords)
                    [keywords.add(k) for k,score in curr_keywords if score >= 0.35]
                    left_id = right_id - 80

                logger.info(f"I extracted {len(keywords)} keywords which are : {keywords}")
                keywords = json.dumps(list(keywords)).replace("[", "").replace("]", "").replace('"', "").replace(",", "").strip()
                logger.info(f"Parse keywords -- {keywords}")

            elif excel_data[sheet_name]["type"] == "num_heavy":
                logger.info(f"Processing {sheet_name} as an excel file with heavy numeric")
                keywords = set([excel_data[sheet_name]["data"]])
                keywords = json.dumps(list(keywords)).replace("[", "").replace("]", "").replace('"', "").replace(",", "").strip()
                logger.info(f"Parse keywords -- {keywords}")
    
    if keywords:
        with sqlite3.connect(db_path) as conn:
            # FTS5 tables don't support 'UNIQUE' constraints; we manually delete then insert
            conn.execute("DELETE FROM document_index WHERE filename = ?", (file_path,))
            conn.execute("INSERT INTO document_index (filename, keywords) VALUES (?, ?)", 
                            (file_path, keywords))
            conn.commit()

    return file_path


def adaptive_excel_ingest(file_path: str) -> Dict[str, Dict[str, str]]:
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

            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()

            col_summary = ", ".join(list(df.columns))
            logger.info(col_summary)
            excel_data[str(sheet_name)]["type"] = "num_heavy"
            excel_data[str(sheet_name)]["data"] = meta_data
            excel_data[str(sheet_name)]["meta_data"] = meta_data

    return excel_data



def adaptive_excel_ingest_old(file_path: str) -> Dict[str, Dict[str, str]]:
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
        meta_data =  table_name.replace("_", " ")



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
                
            # Clean up column names to make them SQL-friendly (remove spaces/special characters if desired)
            df.columns = [f"{col}_{i}" if df.columns.duplicated()[i] else col for i, col in enumerate(df.columns)]
            df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
            df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
            df.dropna(axis=1, how='all', inplace=True)

            table_name = f"{base_name}_{sheet_name.lower()}"
            logger.info(df.columns)

            conn = sqlite3.connect(db_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            conn.close()

            col_summary = ", ".join(list(df.columns))
            logger.info(col_summary)
            excel_data[str(sheet_name)]["type"] = "num_heavy"
            excel_data[str(sheet_name)]["data"] = col_summary
            excel_data[str(sheet_name)]["meta_data"] = meta_data

    return excel_data
   

# Working logic
def process_via_doclin():
    converter = DocumentConverter()
    result = converter.convert("Master_Drawing_Register.xlsx")

    # 2. Extract and flatten tables
    dfs = {}
    for i, table in enumerate(result.document.tables):
        df = table.export_to_dataframe()
        
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
            
        # Clean up column names to make them SQL-friendly (remove spaces/special characters if desired)
        df.columns = [f"{col}_{i}" if df.columns.duplicated()[i] else col for i, col in enumerate(df.columns)]
        df.columns = [col.replace(" ", "_").replace("-", "_") for col in df.columns]
        df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all")
        df.dropna(axis=1, how='all', inplace=True)

        table_name = f"nested_table_{i}"
        dfs[table_name] = df
        logger.info(df.columns)

        conn = sqlite3.connect(db_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()



if __name__ == "__main__":
    batch_ingest("./")
   

