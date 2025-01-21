import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from chain import chain

app = FastAPI(
    title="Promtior chatbot",
    version="1.0",
    description="Promtior technical test - RAG"
)

add_routes(app, chain , path="/promtior")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8765)