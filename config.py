import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENROUTER_API_KEY") is None:
    raise Exception("OPENROUTER_API_KEY is not set")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
