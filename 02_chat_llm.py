import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Chargement des variables d'environnement
load_dotenv()
LLM_API_KEY = os.environ.get("LLM_API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL")

# Création du modèle LLM OpenAI
llm = ChatOpenAI(
    model_name=LLM_MODEL_NAME,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0)

# Création d'un prompt simple avec placeholder {question}
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Tu es un assistant expert en statistiques économiques françaises (Insee).",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm


def chat_simple():
    print("\n💬 Chat simple avec le modèle (sans retrieval). Tapez 'exit' pour sortir.\n")
    while True:
        question = input("Votre question : ")
        if question.lower() == "exit":
            print("Fin de la discussion simple.")
            break

        response = chain.invoke({
            "input": question
        })
        print(f"\n🤖 Réponse : {response.content}\n")


if __name__ == "__main__":
    chat_simple()
