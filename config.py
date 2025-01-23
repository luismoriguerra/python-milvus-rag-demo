import os
from dotenv import load_dotenv

load_dotenv()
# # Load .env file only in development
# if os.getenv("RENDER") is None:  # RENDER is an environment variable automatically set by Render.com

if os.getenv("OPENROUTER_API_KEY") is None:
    raise Exception("OPENROUTER_API_KEY is not set")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
