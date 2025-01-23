from fastapi import APIRouter, HTTPException
from typing import Dict
from models import QuestionRequest
from llm import OpenRouterLLM
import milvus_client
import config
from prompts import create_rag_prompt

router = APIRouter()

# Initialize clients
llm_client = OpenRouterLLM(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY
)

@router.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to my API"}

@router.get("/hello/{name}")
def hello_name(name: str) -> Dict[str, str]:
    return {"message": f"Hello, {name}!"}

@router.post("/ask")
def ask_question(request: QuestionRequest) -> Dict[str, str]:
    try:
        # Get relevant context from Milvus
        context = milvus_client.search_similar_texts(
            question=request.question,
            top_k=request.top_k,
        )

        # Create RAG prompt using template
        rag_prompt = create_rag_prompt(context=context, question=request.question)

        # Generate response using RAG
        response = llm_client.generate_response(
            system_prompt=request.system_prompt,
            user_prompt=rag_prompt
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 