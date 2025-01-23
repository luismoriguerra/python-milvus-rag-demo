from fastapi import FastAPI, HTTPException
from typing import Dict
from openai import OpenAI
import config
from pydantic import BaseModel
from pymilvus import MilvusClient, model as milvus_model
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    system_prompt: str = "You are a helpful AI assistant."
    collection_name: str = "my_rag_collection"
    top_k: int = 3

class OpenRouterLLM:
    def __init__(self, base_url: str, api_key: str):
        if base_url is None:
            raise Exception("base_url is not set")
        if api_key is None:
            raise Exception("api_key is not set")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
    
    def check_connection(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as e:
            raise Exception(f"Failed to connect to OpenRouter: {str(e)}")
    
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

# Initialize the LLM client
llm_client = OpenRouterLLM(
    base_url="https://openrouter.ai/api/v1",
    api_key=config.OPENROUTER_API_KEY
)

# Initialize Milvus client
CLUSTER_ENDPOINT = os.getenv("API")
TOKEN = os.getenv("TOKEN")

if not CLUSTER_ENDPOINT or not TOKEN:
    raise Exception("Milvus API endpoint or token not set in environment variables")

milvus_client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN
)

# Initialize embedding model
embedding_model = milvus_model.DefaultEmbeddingFunction()

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to my API"}

@app.get("/hello/{name}")
def hello_name(name: str) -> Dict[str, str]:
    return {"message": f"Hello, {name}!"}

@app.post("/ask")
def ask_question(request: QuestionRequest) -> Dict[str, str]:
    try:
        # Get relevant context from Milvus
        search_results = milvus_client.search(
            collection_name=request.collection_name,
            data=embedding_model.encode_queries([request.question]),
            limit=request.top_k,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"],
        )

        # Extract context from search results
        context = "\n".join([
            res["entity"]["text"] for res in search_results[0]
        ])

        # Create RAG prompt
        rag_prompt = f"""Use the following context to answer the question. If the context doesn't help, you can answer based on your general knowledge.

Context:
{context}

Question: {request.question}"""

        # Generate response using RAG
        response = llm_client.generate_response(
            system_prompt=request.system_prompt,
            user_prompt=rag_prompt
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 