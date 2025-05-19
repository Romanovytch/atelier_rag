import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from retrieval import get_retriever

# Chargement API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# Création du modèle LLM OpenAI
llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0)

# Création d'un prompt simple avec placeholder {question}
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant expert en statistiques économiques françaises provenant de l'Insee."
        ),
        (
            "human",
            """
            Réponds à la question en utilisant uniquement les informations fournies par ce contexte:

            <context>
            {context}
            </context>

            Question: {input}
            """
        ),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = get_retriever()
rag_chain = create_retrieval_chain(retriever, document_chain)


def chat_simple():
    print("\n💬 Chat simple avec RAG. Tapez 'exit' pour sortir.\n")
    while True:
        question = input("Votre question : ")
        if question.lower() == "exit":
            print("Fin de la discussion.")
            break

        response = rag_chain.invoke({
            "input": question
        })

        # Affichage de la réponse générée
        print("\n🤖 Réponse :")
        print("-" * 60)
        print(response["answer"])
        print("-" * 60)

        # Affichage des sources (contexte)
        sources = response.get("context", [])
        if sources:
            print("\n📚 Sources utilisées :")
            for i, doc in enumerate(sources, 1):
                titre = doc.metadata.get("titre", "Sans titre")
                sous_titre = doc.metadata.get("sous_titre", "")
                print(f"\nSource {i}:")
                print(f"📄 {titre}")
                if sous_titre:
                    print(f"   📝 {sous_titre}")
                print(f"   🔎 Extrait : {doc.page_content[:300].strip()}...")
        else:
            print("\n⚠️ Aucun document de contexte trouvé.")


if __name__ == "__main__":
    chat_simple()
