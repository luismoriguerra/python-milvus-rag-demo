from fastapi import FastAPI
from typing import Dict

app = FastAPI()

@app.get("/")
def read_root() -> Dict[str, str]:
    return {"message": "Welcome to my API"}

@app.get("/hello/{name}")
def hello_name(name: str) -> Dict[str, str]:
    return {"message": f"Hello, {name}!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 