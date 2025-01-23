import os
from dotenv import load_dotenv

# Load .env file only in development
if os.getenv("RENDER") is None:  # RENDER is an environment variable automatically set by Render.com
    load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 

