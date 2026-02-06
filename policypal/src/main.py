import os
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load Env
load_dotenv()

CHROMA_PATH = "./chroma_db"

# You are a helpful HR assistant. Answer the user's question based ONLY on the following context.
# If the answer is not in the context, say "I don't know based on the handbook."
# Do not make up information.

PROMPT_TEMPLATE = """
You are and expert Data Scientists, take the role as teacher also. 
YOU MUST ANSWER THE USER'S QUESTIONS USING THE NEXT CONTEXT.

If the answer is not in the context, say "I don't know, ADD MORE INFORMATION TO TEH CONTEXT"
Do not make up information.

Context:
{context}

Question:
{question}
"""


def main():
    # 1. PREPARE DB
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    # "k=5" means retrieve the top 5 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 2. SETUP LLM
    llm = ChatOllama(model="mistral", temperature=0)

    # 3. DEFINE PROMPT
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 4. BUILD CHAIN (The "Magic" Link)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. INTERACTIVE CHAT
    print("=" * 50)
    print("PolicyPal - Interactive Chat")
    print("=" * 50)
    print("Type 'quit' or 'exit' to end the conversation.\n")

    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        print("\nAI: ", end="", flush=True)
        response = rag_chain.invoke(query)
        print(response)
        print()


if __name__ == "__main__":
    main()
