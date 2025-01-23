from pydantic import BaseModel
from prompts import DEFAULT_ASSISTANT_PROMPT

class QuestionRequest(BaseModel):
    question: str
    system_prompt: str = DEFAULT_ASSISTANT_PROMPT
    collection_name: str = "my_rag_collection"
    top_k: int = 3 