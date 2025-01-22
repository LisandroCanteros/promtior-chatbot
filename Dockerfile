FROM python:3.13

WORKDIR /app

# Install system dependencies for curl and Ollama
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    apt-get clean
    
COPY . .

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8765

CMD ollama serve & sleep 5 && ollama run llama3.2 && python server.py



