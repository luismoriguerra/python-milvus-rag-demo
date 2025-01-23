def create_rag_prompt(context: str, question: str) -> str:
    return f"""Use the following context to answer the question. If the context doesn't help, you can answer based on your general knowledge.

Context:
{context}

Question: {question}"""

# Default system prompts
DEFAULT_ASSISTANT_PROMPT = "You are a helpful AI assistant." 