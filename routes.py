from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from models import QuestionRequest
from llm import OpenRouterLLM
import milvus_client
import config
from prompts import create_rag_prompt
import time
import logging
import psutil
import os

router = APIRouter()

# Initialize clients
llm_client = OpenRouterLLM(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_resource_usage():
    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.1)
    memory_info = process.memory_info()
    return {
        "cpu_percent": cpu_percent,
        "memory_mb": memory_info.rss / (1024 * 1024)  # Convert to MB
    }

@router.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to my API"}

@router.get("/hello/{name}")
def hello_name(name: str) -> Dict[str, str]:
    return {"message": f"Hello, {name}!"}

@router.post("/ask")
def ask_question(request: QuestionRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        initial_resources = get_resource_usage()
        
        # Get relevant context from Milvus
        milvus_start = time.time()
        context = milvus_client.search_similar_texts(
            question=request.question,
            top_k=request.top_k,
        )
        milvus_time = time.time() - milvus_start
        milvus_resources = get_resource_usage()
        logger.info(f"Milvus search took: {milvus_time:.2f} seconds. CPU: {milvus_resources['cpu_percent']}%, Memory: {milvus_resources['memory_mb']:.1f}MB")

        # Create RAG prompt using template
        prompt_start = time.time()
        rag_prompt = create_rag_prompt(context=context, question=request.question)
        prompt_time = time.time() - prompt_start
        prompt_resources = get_resource_usage()
        logger.info(f"Prompt creation took: {prompt_time:.2f} seconds. CPU: {prompt_resources['cpu_percent']}%, Memory: {prompt_resources['memory_mb']:.1f}MB")

        # Generate response using RAG
        llm_start = time.time()
        response = llm_client.generate_response(
            system_prompt=request.system_prompt,
            user_prompt=rag_prompt
        )
        llm_time = time.time() - llm_start
        llm_resources = get_resource_usage()
        logger.info(f"LLM generation took: {llm_time:.2f} seconds. CPU: {llm_resources['cpu_percent']}%, Memory: {llm_resources['memory_mb']:.1f}MB")

        total_time = time.time() - start_time
        final_resources = get_resource_usage()
        logger.info(f"Total request took: {total_time:.2f} seconds. Final CPU: {final_resources['cpu_percent']}%, Memory: {final_resources['memory_mb']:.1f}MB")

        # Calculate memory change
        memory_change = final_resources['memory_mb'] - initial_resources['memory_mb']

        return {
            "response": response,
            "timing": {
                "milvus_search": round(milvus_time, 2),
                "prompt_creation": round(prompt_time, 2),
                "llm_generation": round(llm_time, 2),
                "total_time": round(total_time, 2)
            },
            "resources": {
                "initial": {
                    "cpu_percent": round(initial_resources['cpu_percent'], 1),
                    "memory_mb": round(initial_resources['memory_mb'], 1)
                },
                "final": {
                    "cpu_percent": round(final_resources['cpu_percent'], 1),
                    "memory_mb": round(final_resources['memory_mb'], 1)
                },
                "memory_change_mb": round(memory_change, 1)
            }
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 