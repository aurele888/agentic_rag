"""Run the FastAPI server."""
import uvicorn
import logging
import os, sys 


logger = logging.getLogger(__name__)
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    # {%(pathname)s:%(lineno)d}
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join("logs", "app.log"), mode='w'), 
        logging.StreamHandler(sys.stdout)         
    ], 
)


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )