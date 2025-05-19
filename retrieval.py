import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Chargement du modèle d'embedding
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("EMBEDDING_API_KEY")
)

# Connexion à ChromaDB
CHROMA_PATH = "./chroma_db"
vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

# Création du retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


def retrieve_documents(query):
    """
    Effectue une recherche dans la base vectorielle et retourne les documents trouvés.
    """
    docs = retriever.invoke(query)
    return docs


def get_retriever():
    return retriever


# Test interactif
if __name__ == "__main__":
    print("🔎 Test de retrieval depuis ChromaDB (tapez 'exit' pour quitter)")
    while True:
        query = input("\nVotre question : ")
        if query.lower() == "exit":
            break

        results = retrieve_documents(query)

        print(f"\n📚 {len(results)} documents pertinents trouvés :")
        for i, doc in enumerate(results, 1):
            titre = doc.metadata.get("titre", "Sans titre")
            sous_titre = doc.metadata.get("sous_titre", "")
            content = doc.page_content[:300].replace("\n", " ") + "..."
            print(f"\n--- Chunk {i} ---")
            print(f"Titre       : {titre}")
            print(f"Sous-titre  : {sous_titre}")
            print(f"Contenu     : {content}")
