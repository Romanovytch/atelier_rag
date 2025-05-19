import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from retriever import get_retriever

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

retriever = get_retriever()

map_input = RunnableLambda(lambda x: {
    "input": x["input"],
    "context": retriever.invoke(x["input"])
})

rag_chain = map_input | prompt | llm


def chat_rag():
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
        print(response.content)
        print("-" * 60)


if __name__ == "__main__":
    chat_rag()
