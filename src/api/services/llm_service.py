from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Optional
from pydantic import ValidationError
from ..config import settings
from ...core.utils import detect_device, ReRanker
import logging
import torch

logger = logging.getLogger(__name__)



class LlmService:

    def __init__(self):

        self.llm = ChatOllama(
            model = settings.DEFAULT_CHAT_MODEL,
            base_url = settings.OLLAMA_HOST,
            temperature = 0.7
        )
        self.sentence_transformer = None
        self.cross_encoder_model = None
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful API assistant. Keep your answers concise."),
            ("user", "{user_input}")
        ])
        self.chain = self.prompt_template | self.llm | StrOutputParser()
       

    def set_ollama_model(
            self, 
            model_name : str, 
            temperature: float = 0.7, 
            top_p: float = 0.95, 
            top_k: int= 20
            ) -> Optional[ChatOllama]:

        try:
            logger.info(f"Setting up ollama model :{model_name}")
            if not model_name:
                return  self.llm 
            self.llm = ChatOllama(
            model = model_name, 
            base_url = settings.OLLAMA_HOST,
            temperature = temperature, 
            top_p = top_p, 
            top_k = top_k
            )
            return self.llm
        
        except ValidationError as e:
            # SCOPE: Catches Pydantic schema mismatches or invalid arguments
            logger.error(f"Configuration schema error! Check your parameter names and data types. {e}")
            return None

        except ValueError as e:
            # SCOPE: Catches logical value violations (e.g., out-of-bounds numbers)
            logger.error(f"Parameter value error: {e}")
            return None

        except Exception as e:
            # SCOPE: Catch-all fallback
            logger.error(f"An unexpected error occurred during object setup: {e}")
            return None
    
    
    def set_sentence_tranformer_model(
            self,
            model_name : str = settings.DEFAULT_SENTENCE_TRANSFORMER_MODEL
            ) -> Optional[SentenceTransformer]:

        try:
            logger.info(f"Setting up sentence transformer model : {model_name}")
            self.sentence_transformer = SentenceTransformer(model_name)
            self.sentence_transformer.to(detect_device()["device"])
            self.sentence_transformer.compile(dynamic=True)
            return self.sentence_transformer

        except OSError as e:
            # SCOPE: Highly Specific (Network/Disk level)
            logger.error(f"Model download failed. Check network or model name: {e}")
            return None

        except ValueError as e:
            # SCOPE: Specific (Data/Validation level)
            logger.error(f"Data validation error: {e}")
            return None

        except Exception as e:
            # SCOPE: Global/Generic (Catch-all)
            logger.error(f"An unexpected error occurred: {e}")
            return None
        
    
    def set_cross_encoder_model(
            self,
            model_name : str = settings.RE_RANKER_MODEL
            ) -> Optional[CrossEncoder]:
        try:
            logger.info(f"Setting up re-ranker model : {model_name}")
            self.cross_encoder_model = ReRanker(model_name=model_name)
            return self.cross_encoder_model
        
        except OSError as e:
            # SCOPE: Network/Download or local file-path issues
            logger.error(f"Failed to load or download the CrossEncoder model: {e}")
            return None

        except torch.cuda.OutOfMemoryError as e:
            # SCOPE: Precise hardware failure (VRAM Exhausted)
            logger.error("GPU ran out of memory. Try reducing the 'batch_size' parameter.")
            return None

        except TypeError as e:
            # SCOPE: Data type validation (e.g., accidentally passed an integer/None)
            logger.error(f"Data type error. Ensure all elements are clean strings: {e}")
            return None

        except ValueError as e:
            # SCOPE: Matrix shape validation (e.g., flat list instead of sequence pairs)
            logger.error(f"Formatting error. Did you forget to pair your queries and docs? {e}")
            return None

        except Exception as e:
            # SCOPE: Catch-all for any untracked anomaly
            logger.error(f"An unexpected error occurred during reranking: {e}")
            return None
    

    async def generate_langchain_response(self, user_prompt: str) -> str:
        """Executes the chain asynchronously using LangChain's ainvoke method."""
        try:
            # ainvoke runs smoothly in FastAPI's async loop without blocking other users
            response = await self.chain.ainvoke({"user_input": user_prompt})
            return response
        except Exception as e:
            return f"LangChain Error: {str(e)}"


    # Inside your LangChainOllamaService class
    async def stream_langchain_response(self, user_prompt: str):
        """Yields tokens as they are generated by LangChain."""
        async for chunk in self.chain.astream({"user_input": user_prompt}):
            yield chunk

    
    