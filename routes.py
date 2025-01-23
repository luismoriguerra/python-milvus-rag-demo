from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from models import QuestionRequest
from llm import OpenRouterLLM
import milvus_client
import config
from prompts import create_rag_prompt
import time
import logging

router = APIRouter()

# Initialize clients
llm_client = OpenRouterLLM(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        # Get relevant context from Milvus
        milvus_start = time.time()
        context = milvus_client.search_similar_texts(
            question=request.question,
            top_k=request.top_k,
        )
        milvus_time = time.time() - milvus_start
        logger.info(f"Milvus search took: {milvus_time:.2f} seconds")

        # Create RAG prompt using template
        prompt_start = time.time()
        rag_prompt = create_rag_prompt(context=context, question=request.question)
        prompt_time = time.time() - prompt_start
        logger.info(f"Prompt creation took: {prompt_time:.2f} seconds")

        # Generate response using RAG
        llm_start = time.time()
        response = llm_client.generate_response(
            system_prompt=request.system_prompt,
            user_prompt=rag_prompt
        )
        llm_time = time.time() - llm_start
        logger.info(f"LLM generation took: {llm_time:.2f} seconds")

        total_time = time.time() - start_time
        logger.info(f"Total request took: {total_time:.2f} seconds")

        return {
            "response": response,
            "timing": {
                "milvus_search": round(milvus_time, 2),
                "prompt_creation": round(prompt_time, 2),
                "llm_generation": round(llm_time, 2),
                "total_time": round(total_time, 2)
            }
        }
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 