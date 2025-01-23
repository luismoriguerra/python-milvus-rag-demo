from openai import OpenAI

model = "meta-llama/llama-3.2-1b-instruct"

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
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content 