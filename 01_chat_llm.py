import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

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
            "Tu es un assistant expert en statistiques économiques françaises (Insee).",
        ),
        ("human", "{question}"),
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
            "question": question
        })
        print(f"\n🤖 Réponse : {response.content}\n")


if __name__ == "__main__":
    chat_simple()
