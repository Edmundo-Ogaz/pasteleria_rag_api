from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def cargar_y_dividir_markdown(ruta_archivo):
    """Carga un archivo Markdown y lo divide usando encabezados."""
    try:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:  # Manejo de encoding
            markdown_text: str = archivo.read().lower()
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {ruta_archivo}")
        return []
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return []

    headers_to_split_on = [
        ("#", "Nivel 1"),
        ("##", "Nivel 2"),
        ("###", "Nivel 3"),
        ("####", "Nivel 4"),
        ("#####", "Nivel 5"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    md_header_splits = markdown_splitter.split_text(markdown_text)

    return md_header_splits

def guardar_en_chroma(documentos, nombre_coleccion="mi_coleccion"):
    """Genera embeddings y los guarda en Chroma."""

    embeddings = FastEmbedEmbeddings(model_name="jinaai/jina-embeddings-v2-base-es")
    vectordb = Chroma.from_documents(
        documents=documentos,
        embedding=embeddings,
        persist_directory="db",
        collection_name=nombre_coleccion,
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )
    vectordb.persist()
    return vectordb

ruta_markdown = "PasteleriaLaPalmera.md"
documentos = cargar_y_dividir_markdown(ruta_markdown)

if documentos:
    vectordb = guardar_en_chroma(documentos)
    print(f"Documentos guardados en Chroma en la colección '{vectordb._collection.name}'")

    # Ejemplo de consulta (opcional)
    query = "¿Cuál es el horario de atención?"

    result = vectordb.similarity_search_with_score(query, k=3)
    print("\nResultados de la búsqueda:")
    for doc, i in result:
        print(doc.page_content, i)
        print("---")
    print("---"*2)
    result = vectordb.similarity_search_with_relevance_scores(query, k=3)
    print("\nResultados de la búsqueda:")
    for doc, i in result:
        print(doc.page_content, i)
        print("---")
else:
    print("No se pudieron procesar los documentos Markdown.")