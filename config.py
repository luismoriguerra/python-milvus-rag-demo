import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("OPENROUTER_API_KEY") is None:
    raise Exception("OPENROUTER_API_KEY is not set")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Milvus Configuration
CLUSTER_ENDPOINT = os.getenv("API")

if CLUSTER_ENDPOINT is None:
    raise Exception("CLUSTER_ENDPOINT is not set")

TOKEN = os.getenv("TOKEN")

if TOKEN is None:
    raise Exception("TOKEN is not set")