from flask import Flask, request

from langchain.vectorstores import Chroma
from langchain_community.embeddings import JinaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

embeddings = JinaEmbeddings( 
    jina_api_key=os.environ.get("JINA_API_KEY"), model_name="jina-embeddings-v2-base-es" 
)
vectordb = Chroma(
    persist_directory='db', embedding_function=embeddings, collection_name="mi_coleccion",
)
raw_prompt = PromptTemplate.from_template("""
    <s>[INST] Eres un asistente de la Pastelería la Palmera. Debes:
    - Responder ÚNICAMENTE usando la información del contexto proporcionado
    - No inventar ni inferir información adicional
    - Si la información no está explícitamente en el contexto, responder explicitamente lo siguiente: "No puedo responder esta pregunta con la información proporcionada"
    - No hacer suposiciones sobre productos, precios o servicios que no estén mencionados
    - Responder siempre en el mismo idioma en que se realiza la pregunta [/INST] </s>
    
    [INST] Pregunta: {input}
    Contexto disponible: {context}
    Respuesta: [/INST]
"""
)
retriever = vectordb.as_retriever(
    search_type= "similarity_score_threshold",
    search_kwargs={
        "k":3,
        "score_threshold":0.1
    }
)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_retries=2
)
document_chain = create_stuff_documents_chain(
    llm, raw_prompt
)
chain = create_retrieval_chain(retriever, document_chain)

app = Flask(__name__)

@app.route("/helth", methods=["GET"]) 
def health():
    return "ok"

@app.route("/ask", methods=["POST"]) 
def ask():
    json_content = request.json
    query= json_content.get("query")
    print(f"query: {query}")

    result = vectordb.similarity_search_with_score(query, k=3)
    print("\nResultados de la búsqueda:")
    for doc, i in result:
        print(doc.page_content, i)
        print("---")
    
    response_answer= {"respuesta": result[0][0].page_content.strip()}
    return response_answer

@app.route("/ask-model", methods=["POST"]) 
def ask_model():
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    result = chain.invoke({"input":query})
    print(result)

    response_answer= {"respuesta": result['answer'].replace("<s>[INST]", "").replace("[/INST]</s>","").strip()}
    return response_answer
    
if __name__ == "__main__":
    app.run(debug=True)