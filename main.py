from fastapi import FastAPI, HTTPException
from typing import Dict
from openai import OpenAI
import config
from pydantic import BaseModel

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    system_prompt: str = "You are a helpful AI assistant."

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

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to my API"}

@app.get("/hello/{name}")
def hello_name(name: str) -> Dict[str, str]:
    return {"message": f"Hello, {name}!"}

@app.post("/ask")
def ask_question(request: QuestionRequest) -> Dict[str, str]:
    try:
        response = llm_client.generate_response(
            system_prompt=request.system_prompt,
            user_prompt=request.question
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 