# Milvus Python Demo App

A FastAPI-based Retrieval-Augmented Generation (RAG) application that combines the power of Milvus vector database with Large Language Models for intelligent question answering.

## Features

- 🚀 FastAPI-powered REST API
- 🔍 Vector similarity search using Milvus
- 🤖 LLM integration through OpenRouter
- 📊 Detailed performance metrics and resource usage tracking
- 🎯 Configurable top-k similar document retrieval
- ⚡ Optimized search parameters for better performance

## Prerequisites

- Python 3.8+
- Milvus Cloud account
- OpenRouter API key

## Environment Variables

Create a `.env` file with the following variables:

```env
CLUSTER_ENDPOINT=your_milvus_endpoint
TOKEN=your_milvus_token
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd milvus-python-demo-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the FastAPI server:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
- Welcome endpoint
- Returns: `{"message": "Welcome to my API"}`

### GET /hello/{name}
- Simple greeting endpoint
- Returns: `{"message": "Hello, {name}!"}`

### POST /ask
- Main QA endpoint
- Request body:
```json
{
    "question": "Your question here",
    "system_prompt": "Optional system prompt",
    "top_k": 5
}
```
- Returns:
```json
{
    "response": "LLM-generated answer",
    "timing": {
        "milvus_search": 0.00,
        "prompt_creation": 0.00,
        "llm_generation": 0.00,
        "total_time": 0.00
    },
    "resources": {
        "initial": {
            "cpu_percent": 0.0,
            "memory_mb": 0.0
        },
        "final": {
            "cpu_percent": 0.0,
            "memory_mb": 0.0
        },
        "memory_change_mb": 0.0
    }
}
```

## Architecture

The application follows a modular architecture:

- `main.py`: Application entry point and FastAPI setup
- `routes.py`: API endpoint definitions and request handling
- `milvus_client.py`: Milvus vector database integration
- `llm.py`: LLM client implementation
- `models.py`: Pydantic models for request/response validation
- `config.py`: Configuration management
- `prompts.py`: RAG prompt templates

## Performance Monitoring

The application includes detailed performance monitoring:
- Execution time for each processing step
- CPU usage tracking
- Memory consumption metrics
- Resource usage changes during request processing

## Deployment

The application is configured for deployment on Render with the following files:
- `Procfile`: Process type definitions
- `render.yaml`: Render deployment configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here] 