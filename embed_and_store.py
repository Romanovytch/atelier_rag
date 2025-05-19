import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY")
MODEL_NAME = "text-embedding-3-small"
CHROMA_PATH = "./chroma_db"

embedding_model = OpenAIEmbeddings(
    model=MODEL_NAME,
    openai_api_key=EMBEDDING_API_KEY
)


def embed_and_store_chunks(chunks):
    documents = [
        Document(
            page_content=chunk['contenu'],
            metadata={"titre": chunk['titre'], "sous_titre": chunk['sous_titre']}
        )
        for chunk in chunks
    ]

    print("🕒 Génération des embeddings et stockage dans ChromaDB...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    print(f"✅ {len(documents)} documents embedded et stockés avec succès dans {CHROMA_PATH}.")

    return db  # temporairement pour test


if __name__ == "__main__":
    from scrape_insee import scrape_urls, urls
    from chunking import create_chunks

    documents = scrape_urls(urls, use_cache=True)
    all_chunks = create_chunks(documents)

    # db = embed_and_store_chunks(all_chunks)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)

    # Affichage
    embedding_result = db._collection.get(limit=1, include=["embeddings", "documents"])
    exemple_embedding = embedding_result['embeddings'][0]

    print("\n🔍 Exemple de chunk stocké dans ChromaDB :\n", "-"*50)
    print(embedding_result['documents'][0][:500], "...")

    print("\n📐 Embedding correspondant (premiers 10 éléments) :\n", "-"*50)
    print(exemple_embedding[:10], "... (total longueur:", len(exemple_embedding), ")")
