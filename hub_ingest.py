"""
    BASIC CLASS TO MIRROR COSMOS DRIVE
"""

from typing import List, Dict, Union, Tuple
import requests
import logging
import os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import sqlite3
import networkx as nx
from enum import Enum
from pathlib import Path
from threading import Thread
import subprocess
import pandas as pd
warnings.filterwarnings('ignore', category=UserWarning)


PERSIST_DIRECTORY = os.path.join("data", "vectors")
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)


logger = logging.getLogger(__name__)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log"), mode='w'), 
        logging.StreamHandler(sys.stdout)         
    ], 
)

class LayerType(str, Enum):
    O = "ORGANIZATION"
    U = "USER"
    P = "PROJECT"

class LayerRootFolder(str, Enum):
    O = "Organization"
    U = "My Drive"
    P = "Projects"


class HubDigest:

    API_KEY = "apikey_OAo7UuyFAPx10D2X65jv8rTTuxzxqyyL"
    API_BASE_TIER1_URL = "https://cosmos-cmp-api-dev.varadise.cloud/api/v1"
    API_BASE_URL = "https://cosmos-dev-api.varadise.cloud/drive-service/api/v1"
    ORG_URL = os.path.join(API_BASE_TIER1_URL,"organizations")
    PROJ_URL = os.path.join(API_BASE_TIER1_URL,"projects")
    PROJ_CONTENT_URL = os.path.join(API_BASE_URL,"drive/PROJECT/id/contents")
    ORG_CONTENT_URL = os.path.join(API_BASE_URL,"drive/ORGANIZATION/id/contents")
    DRIVE_CONTENT_URL = os.path.join(API_BASE_URL,"drive/USER/id/contents")
    FOLDER_CONTENT_URL = os.path.join(API_BASE_URL,"folders/id/contents")
    DOC_URL = os.path.join(API_BASE_URL,"documents/id/download")
    API_BASE_URL = "https://cosmos-dev-api.varadise.cloud/drive-service/api/v1"
    TREE_URL = os.path.join(API_BASE_URL,"drive/layer/id/folder-tree")

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    def __init__(self):
        
        self.db_path = os.path.join(PERSIST_DIRECTORY, "documents.db")
        self._init_tables()


    def _init_tables(self):

        """Initializes the BM25-enabled virtual table and log table."""

        # with sqlite3.connect(self.db_path) as conn:
        #     conn.execute('''
        #         CREATE TABLE IF NOT EXISTS document_logs (
        #             id INTEGER PRIMARY KEY AUTOINCREMENT,
        #             doc_id TEXT,
        #             doc_type TEXT,
        #             url TEXT,
        #             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        #         )
        #     ''')
        #     conn.commit()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS hub_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    doc_type TEXT,
                    doc_source TEXT,
                    tag TEXT,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()


    def save_document(
            self, 
            doc_info:Dict,
            ) -> List:
        """
            Save the documents meta-data in a Database
        """
        if not doc_info:
            return
        if not ("doc_id" in doc_info):
            return
        if not ("doc_type" in doc_info):
            return
        if not ("doc_source" in doc_info):
            return
        if not ("category" in doc_info):
            return
        if not ("tag" in doc_info):
            return
        existing_doc_ids = {}
        process_doc_info = [
            {
                "doc_id": doc_info["doc_id"][j],
                "doc_type": doc_info["doc_type"][j],
                "doc_source": doc_info["doc_source"][j],
                "tag": doc_info["tag"][j],
                "category": doc_info["category"][j],
                } for j in range(len(doc_info["doc_id"])) ]
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT doc_id FROM hub_documents")
                existing_doc_ids = {row[0] for row in cursor.fetchall()}

                filtered_doc_info = [d for d in process_doc_info if d["doc_id"] not in existing_doc_ids]
                if not filtered_doc_info:
                    return existing_doc_ids
                
                query = "INSERT INTO hub_documents (doc_id, doc_type, doc_source, tag, category) VALUES (:doc_id, :doc_type, :doc_source, :tag, :category)"
                conn.executemany(query, filtered_doc_info)

                conn.commit()
                logger.info(f"Successfully inserted {len(process_doc_info)} rows.")
                
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
        finally:
            conn.close()
            return existing_doc_ids


    def get_layer_level_tree(
            self, 
            layer_id:str, 
            layer_type:LayerType = LayerType.P,
            ):
        """
        
        """
        if layer_type == LayerType.P:
            url = HubDigest.TREE_URL.replace("layer", LayerType.P).replace("id", layer_id)
        elif layer_type == LayerType.O:
            url = HubDigest.TREE_URL.replace("layer", LayerType.O).replace("id", layer_id)
        elif layer_type == LayerType.U:
            url = HubDigest.TREE_URL.replace("layer", LayerType.U).replace("id", layer_id)
        else:
            raise Exception("Unknown layer")
        tree_data = self._fetch_data_by_url(url, None)
        return tree_data


    def remove_document(
            self, 
            doc_id: str,
            ) -> bool:
        """
            Removes a document and its keywords from the BM25 index.
        """
        if not doc_id:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT doc_id FROM document_logs WHERE doc_id = ?", (doc_id,))
                exists = cursor.fetchone()
                if not exists:
                    logger.info(f"Document not found in index: {doc_id}")
                    return False
                conn.execute("DELETE FROM document_logs WHERE doc_id = ?", (doc_id,))
                conn.commit()
            logger.info(f"Successfully removed from index: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error removing document {doc_id}: {e}")
            return False
        

    def get_org_doc_urls(
            self, 
            org_id:str,
            ) -> Tuple[Dict[str, List[str]], List]:
        """
        
        """
        url_dict = {}
        # The organization level layer map directory to a root directory  
        doc_meta_dict = hub_diggest.get_doc_from_folders([org_id], 'O')
        for folder_id in doc_meta_dict:
            doc_id_list = [{"id": doc_meta["id"], 
                            "name": doc_meta["name"],
                            "fileType": doc_meta["fileType"], 
                            "folderId": doc_meta["folderId"]} for doc_meta in doc_meta_dict[folder_id]]
            url_dict[folder_id] = hub_diggest.get_url_from_doc_ids(doc_id_list)
        
        return  url_dict, []


    def get_user_doc_urls(
            self, 
            user_id:str,
            ) -> Tuple[Dict[str, List[str]], List]:
        """
        
        """
        url_dict = {}
        # The drive/users level layer map directory to a root directory  
        doc_meta_dict = hub_diggest.get_doc_from_folders([user_id], 'U')
        for folder_id in doc_meta_dict:
            doc_id_list = [{"id": doc_meta["id"],
                            "name": doc_meta["name"],
                            "folderId": doc_meta["folderId"], 
                            "fileType": doc_meta["fileType"]} for doc_meta in doc_meta_dict[folder_id]]
            url_dict[folder_id] = hub_diggest.get_url_from_doc_ids(doc_id_list)
        
        return  url_dict, []
      
    
    def get_all_proj_doc_urls(
            self, 
            org_id:str,
            ) -> Tuple[Dict[str, List[str]], List]:
        """
         
        """
        url_dict = {}
        # The project level layer does not map directory to a root directory 
        proj_list_from_org = hub_diggest.get_proj_by_org_id(org_id)
        # Get only ids from meta-data
        proj_ids_from_org = [proj["id"] for proj in proj_list_from_org]
        
        # Get only names from meta-data
        proj_names_from_org = []
        doc_meta_dict =  hub_diggest.get_doc_from_folders(proj_ids_from_org)
        
        for folder_id in doc_meta_dict:
            doc_id_list = [{"id": doc_meta["id"], 
                            "name": doc_meta["name"], 
                            "folderId": doc_meta["folderId"],
                            "fileType": doc_meta["fileType"]} for doc_meta in doc_meta_dict[folder_id]]
            url_dict[folder_id] = hub_diggest.get_url_from_doc_ids(doc_id_list)
            proj_names_from_org.extend([proj["name"] for proj in proj_list_from_org if proj["id"] == folder_id])
        
        return  url_dict, proj_names_from_org
        

    # def get_org_ids(self) -> List[Dict]:
    #     org_data = self._fetch_data_by_url(HubDigest.ORG_URL)
    #     if not org_data:
    #         return []
    #     if not ("data" in org_data):
    #         return []
    #     return [org for org in org_data["data"]]
    

    def get_proj_by_org_id(
            self, 
            org_id:str,
            ) -> List[Dict]:
        """
        
        """
        if not org_id:
            return []
        query_param = {
            "orgIds.value": org_id,
            "orgIds.operator": "=",
        }
        proj_data = self._fetch_data_by_url(HubDigest.PROJ_URL, query_param)
        if not proj_data:
            return []
        if not ("data" in proj_data):
            return []
        return [proj for proj in proj_data["data"]]


    def get_doc_from_folders(
            self, 
            folder_ids:List[str], 
            level="P",
            ) -> Dict[str, List[str]]:
        """
            Traverse the list of folders and sub-folders to get a list of all documents
            Return each folders id with its corresponding files 
        """
        
        folders_doc_dict = {}
        if not folder_ids:
            return folders_doc_dict
        
        folders_doc_list = [[] for k in range(len(folder_ids))]
        with ThreadPoolExecutor() as executor:
            futures = {
                    executor.submit(
                    self._get_doc_from_folder, 
                    folder_doc_list,
                    folder_id, 
                    level,
                    ) : folder_id for folder_id, folder_doc_list in zip(folder_ids, folders_doc_list)
                }
            for future in as_completed(futures):
                try:
                    # set a list per folder id
                    folders_doc_dict[futures[future]] = []
                    results = future.result()
                    # folders_doc_list.extend(results)
                    folders_doc_dict[futures[future]].extend(results)
                except Exception as error_msg:
                    logger.error(f"Unexpected error occurred with {futures[future]} which is cause by :{error_msg}")
        return folders_doc_dict
    
    

    def get_url_from_doc_ids(
            self, 
            doc_meta_list: List[Dict]
            ) -> List[Dict]:
        """
        
        """
        doc_url_list = []
        if not doc_meta_list:
            doc_url_list

        with ThreadPoolExecutor() as executor:
            futures = {
                    executor.submit(
                    self._get_url_from_doc_id, 
                    doc_meta["id"], 
                    ) : (doc_meta["id"], doc_meta["name"],  doc_meta["fileType"], doc_meta["folderId"]) for doc_meta in doc_meta_list
                }
            for future in as_completed(futures):
                try:
                    logger.info(f"File type is : {futures[future][1]}")
                    results = future.result()
                    doc_url_list.append({"url": results,
                                         "id": futures[future][0],
                                         "name": futures[future][1],
                                          "fileType": futures[future][2],
                                          "folderId": futures[future][3]})
                except Exception as error_msg:
                    logger.error(f"Unexpected error occurred with {futures[future]} which is cause by :{error_msg}")
                
        return doc_url_list
    

    def _get_doc_from_folder(
            self, 
            folder_doc_list:List[str],
            folder_id:str, 
            level="P", 
            ) -> List[Dict]:
        """
            Traverse a folder and sub-folder to get a list of all documents
        """
        # Drive level content
        if level == "U":
            CURR_URL =  HubDigest.DRIVE_CONTENT_URL.replace("id", folder_id)
        # Organization level content
        elif level == "O":
            CURR_URL =  HubDigest.ORG_CONTENT_URL.replace("id", folder_id)
        # Project level content
        elif level == "P":
            CURR_URL =  HubDigest.PROJ_CONTENT_URL.replace("id", folder_id)
        # Folder level content
        elif level == "F":
            CURR_URL =  HubDigest.FOLDER_CONTENT_URL.replace("id", folder_id)

        folder_data = self._fetch_data_by_url(
            CURR_URL,
            )
        if not folder_data:
            return []
        
        if "documents" in folder_data:
            if level == "O":
                folder_id = LayerRootFolder.O
            elif level == "P":
                folder_id = LayerRootFolder.P
            elif level == "U":
                folder_id = LayerRootFolder.U

            [folder_data["documents"][j].update({"folderId": folder_id}) for j in range(len(folder_data["documents"]))]
            folder_doc_list.extend(folder_data["documents"])

        if not("folders" in folder_data):
            return folder_doc_list 
        for folder in folder_data["folders"]:
            level="F"
            self._get_doc_from_folder(folder_doc_list, folder["id"], level)
        
        return folder_doc_list
      

    def _get_url_from_doc_id(
            self, 
            doc_id: str,
            ) -> str|None:
        """
        
        """
        if not doc_id:
            return None
        
        doc_url_data = self._fetch_data_by_url(HubDigest.DOC_URL.replace("id", doc_id))
        if not doc_url_data:
            return None
        if not ("url" in doc_url_data):
            return None
        return  doc_url_data["url"]
    

    def _fetch_data_by_url(
            self, 
            url:str, 
            query_param: Dict[str, str]=None,
            ) -> Dict:
        """
            Fetch Data Based on the provided URL and the Query parameters
        """
        try:
            response = requests.get(
                url,
                headers = HubDigest.headers,
                params = query_param,
                timeout = 10,  
            )
            logger.info(f"Request URL: {response.url}")
            response.raise_for_status()
            if not response :
                return {}
            fetch_data = response.json()
            if not fetch_data:
                return {}
            
            logger.info(f"Success! Data retrieved : {fetch_data}")
            return fetch_data 
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return {}
        except requests.exceptions.RequestException as err:
            logger.error(f"An error occurred: {err}")
            return {}


    def parse_folders(
            self, 
            folder_list, 
            parent_id=None, 
            G=None,
            ):

        for folder in folder_list:
            node_id = folder["id"]
            
            # Extract attributes, default to None if missing
            node_attrs = {
                "name": folder.get("name"),
                "color": folder.get("color")
            }
            # Add node with metadata
            G.add_node(node_id, **node_attrs)
            
            # Add edge from parent to current folder
            if parent_id:
                G.add_edge(parent_id, node_id)
                
            # Recursively parse nested folders
            if "folders" in folder:
                self.parse_folders(folder["folders"], parent_id=node_id, G=G)


    def get_folder_path(
            self, 
            graph, 
            target_id, 
            return_names=True,
            ) -> List[str]:
        """
        Finds the path from the root folder to the target ID.
        Returns a list of IDs or a list of folder names.
        """
        if target_id not in graph:
            return target_id
            #return f"ID {target_id} not found in the graph."
        
        path = [target_id]
        current = target_id
        
        # Traverse backwards using predecessors until we hit a root node
        while True:
            predecessors = list(graph.predecessors(current))
            if not predecessors:
                break
            current = predecessors[0]
            path.append(current)
            
        # Reverse to get the path from root -> target
        path.reverse()
        
        if return_names:
            # Convert IDs to their human-readable folder names
            return [graph.nodes[node_id].get("name") for node_id in path]
            
        return path


    def download_file_from_urls(
            self,
            existing_doc_ids: List[str],
            final_doc_dict: Dict,
            ) -> List[Path]:
    
        doc_url_list = []

        with ThreadPoolExecutor() as executor:
        
            futures = {
                    executor.submit(
                    self._download_file_from_url, 
                    final_doc_dict['local_uri'][index],
                    final_doc_dict['remote_url'][index], 
                    ) : (final_doc_dict['local_uri'][index], 
                         final_doc_dict['remote_url'][index],
                         ) for index, doc_id in enumerate(final_doc_dict['doc_id']) if doc_id not in existing_doc_ids
                }
            
            for future in as_completed(futures):
                try:
                    logger.info(f"Sucessfully downloaded and saved file in : {futures[future][0]}")
                    results = future.result()
                    doc_url_list.append(futures[future][0] if results else [])
                except Exception as error_msg:
                    logger.error(f"Unexpected error occurred with {futures[future]} which is cause by :{error_msg}")
        return doc_url_list
    

    def _download_file_from_url(
            self,
            output_file_path:Path, 
            input_file_url:str, 
            timeout: int = 600,
            ) -> Union[subprocess.CompletedProcess[str], None]:
        """ 
            Method to download any files
        """
        self.command = ["wget", "-O", "output_file_path", "input_file_url"]
        result = None
        if not output_file_path:
            return None
        if not input_file_url:
            return None
        
        self.command[2] = output_file_path
        self.command[3] = input_file_url
        try:
            logger.info(f"Running : {self.command}")
            result = subprocess.run(
            self.command,
            timeout=timeout,
            check=True,
            capture_output=True,
            text=True 
            )

        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {e.timeout} seconds.")
            logger.error(f"Captured stdout: {e.stdout}")
            logger.error(f"Captured stderr: {e.stderr}")
        except FileNotFoundError as e:
            logger.error(f"Error: Command '{self.command[0]}' not found - {e}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Process failed with exit code {e.returncode}")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
        return result
    

    def mirror_cosmos_drive(
            self, 
            org_id: str, 
            user_id:str,
            ):
        """
        
        """
        # Initialise the graph
        G = nx.DiGraph()
        # Get doc url (from ORG, PROJ, USER) API
        doc_url_dict = {LayerType.O:[], LayerType.U:[], LayerType.P:[] }
        doc_url_dict[LayerType.O].append(self.get_org_doc_urls(org_id))
        doc_url_dict[LayerType.U].append(self.get_user_doc_urls(user_id))
        doc_url_dict[LayerType.P].append(self.get_all_proj_doc_urls(org_id))
        
        for layer_type in doc_url_dict:
            if not doc_url_dict[layer_type]:
                continue

            final_doc_dict = {
                "doc_id": [], 
                "doc_type": [], 
                "doc_source": [], 
                "category": [], 
                "tag": [], 
                "remote_url": [], 
                "local_uri": []
                }

            for k, sub_layer in enumerate(doc_url_dict[layer_type][0][0]):
                # Get the layer tree and run the parser
                self.parse_folders(hub_diggest.get_layer_level_tree(sub_layer, layer_type)["folders"], G=G)
                if layer_type == LayerType.U:
                    root_dir =  LayerRootFolder.U + "/"
                elif layer_type == LayerType.O:
                    root_dir =  LayerRootFolder.O + "/"
                elif layer_type == LayerType.P:
                    root_dir =  LayerRootFolder.P + "/" + doc_url_dict[layer_type][0][1][k] + "/"
                else:
                    root_dir = ""
                
                data = doc_url_dict[layer_type][0][0][sub_layer]
                for j in range(len(data)):

                    final_doc_dict["remote_url"].append(data[j]['url'])
                    final_doc_dict["doc_id"].append(data[j]['id'])
                    final_doc_dict["doc_type"].append(data[j]['fileType'])
                    final_doc_dict["doc_source"].append("cosmos")
                    final_doc_dict["category"].append(layer_type.value)
                    # Default tag for data
                    final_doc_dict["tag"].append("drive")

                    # Deriving the file path using the fact that the root is of type LayerRootfolder
                    folder_path = hub_diggest.get_folder_path(G, data[j]["folderId"], return_names=True)
                    data[j].update({"filePath": 
                                    Path(root_dir + "/".join(folder_path if not isinstance(folder_path, LayerRootFolder) else ""))})
                    
                    # Create path in case it does not exist
                    if not os.path.exists(data[j]["filePath"]):
                        logger.info(f"Current path : {data[j]['filePath']}")
                        os.makedirs(data[j]["filePath"], exist_ok=True)
                    final_doc_dict["local_uri"].append(data[j]['filePath']/data[j]['name'])

            logger.info(f"Total : {len(doc_url_dict[layer_type][0][0])} --- {doc_url_dict[layer_type]}")
            existing_doc_ids = self.save_document(final_doc_dict)
            self.download_file_from_urls(existing_doc_ids, final_doc_dict)
            

if __name__=="__main__":

    start_time = time.time()
    hub_diggest = HubDigest()

    # Init org_id and user_id
    #org_id = "65aa39ca-2f82-4378-b8d2-9aa98eed3c4a"
    # Org with less data
    org_id = "8fb08acd-dd8b-4a6c-9b06-8d1030b380b1"
    user_id = "99999999-9999-9999-9999-999999999999"
    hub_diggest.mirror_cosmos_drive(org_id, user_id)

    
