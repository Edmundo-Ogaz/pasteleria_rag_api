from flask import Flask, request

from llm.chroma import chroma
from llm.groq import llm
from repository.session import session
from llm.jina import reranker

import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route("/health", methods=["GET"]) 
def health():
    return "OK"

@app.route("/ask", methods=["POST"]) 
def ask():
    json_content = request.json
    query = str(json_content.get("query")).lower()
    session_id = request.headers.get("sessionId")
    print(f"query: {query}")
    print(f"sessionId: {session_id}")

    retriever = chroma.get_similarity(query)
    rerank = reranker.rerank(retriever, query)

    return {"respuesta": rerank}

@app.route("/ask-model", methods=["POST"]) 
def ask_model():
    json_content = request.json
    query = str(json_content.get("query")).lower()
    session_id = request.headers.get("sessionId")
    print(f"query: {query}")
    print(f"session_id: {session_id}")

    response = llm.invoke(query)
    print(f"response: {response}")
    
    return {"respuesta": response}

@app.route("/ask-model-history", methods=["POST"]) 
def ask_model_history():
    json_content = request.json
    query = str(json_content.get("query")).lower()
    session_id = request.headers.get("sessionId")
    print(f"query: {query}")
    print(f"session_id: {session_id}")

    return {"respuesta": session.invoke_with_history(session_id, query)}
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port,debug=True)