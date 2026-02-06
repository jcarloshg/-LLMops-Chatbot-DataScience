That is a great choice. Getting the foundation right is the most critical part of LLMOps. If the data isn't chunked or stored correctly, no amount of prompt engineering will fix the output later.

Here is the step-by-step execution guide for **Saturday: The Build (Phase 1)** of "PolicyPal".

---

### **Step 1: Project & Environment Setup (30 Mins)**

We need a clean workspace. Do not install everything globally.

1. **Create the Folder Structure:**
   Open your terminal (VS Code or Command Prompt) and run:

```bash
mkdir policypal
cd policypal
mkdir src data tests
touch src/ingest.py src/rag_chain.py .env

```

_Your folder should look like this:_

```text
policypal/
├── data/           <-- Put your PDF here
├── src/
│   ├── ingest.py   <-- Script to create the database
│   └── main.py     <-- Script to run the RAG chain
├── tests/
└── .env            <-- Store your API keys here

```

2. **Virtual Environment & Installation:**

```bash
# Create virtual env
python -m venv venv

# Activate it (Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate)

# Install the specific stack for this weekend
pip install langchain langchain-openai langchain-community chromadb pypdf python-dotenv

```

_(Note: If you are using Ollama instead of OpenAI, install `langchain-ollama` as well.)_ 3. **Get Data:**

- Download a generic "Employee Handbook PDF" (search for "GitLab Employee Handbook PDF" or "HubSpot Culture Code PDF" for good real-world examples).
- Rename it to `handbook.pdf` and place it in the `data/` folder.

---

### **Step 2: Ingestion & Vector Storage (09:30 AM - 01:00 PM)**

We will write a script to load the PDF, split it into chunks, and save it into a local Chroma database.

**Create/Edit `src/ingest.py`:**

```python
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

# CONFIGURATION
DATA_PATH = "./data/handbook.pdf"
CHROMA_PATH = "./chroma_db"

def ingest_docs():
    # 1. LOAD
    print("Loading PDF...")
    loader = PyPDFLoader(DATA_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages.")

    # 2. CHUNK (Critical Step)
    # We use a 1000 character window with 200 character overlap
    # to ensure context isn't lost between splits.
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,  # Helps us trace where text came from
    )
    chunks = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(chunks)} chunks.")

    # 3. EMBED & STORE
    # Note: If using Ollama, swap this for OllamaEmbeddings()
    print("Saving to ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )
    print(f"Saved to {CHROMA_PATH}. Ingestion complete!")

if __name__ == "__main__":
    ingest_docs()

```

**Action:** Run `python src/ingest.py`. You should see a new folder `chroma_db` appear in your project root.

---

### **Step 3: The RAG Chain Implementation (02:00 PM - 05:00 PM)**

Now we build the brain. We will connect the database we just built to the LLM.

**Create/Edit `src/main.py`:**

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# Load Env
load_dotenv()

CHROMA_PATH = "./chroma_db"

PROMPT_TEMPLATE = """
You are a helpful HR assistant. Answer the user's question based ONLY on the following context.
If the answer is not in the context, say "I don't know based on the handbook."
Do not make up information.

Context:
{context}

Question:
{question}
"""

def main():
    # 1. PREPARE DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # "k=3" means retrieve the top 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 2. SETUP LLM
    # Use gpt-3.5-turbo for speed/cost, or "llama3" if using ChatOllama
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 3. DEFINE PROMPT
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # 4. BUILD CHAIN (The "Magic" Link)
    # This checks: Question -> Find Context -> Fill Prompt -> Send to LLM -> Parse String
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. TEST IT
    query = "What is the policy on remote work?"
    print(f"User Question: {query}\n")

    response = rag_chain.invoke(query)
    print("AI Answer:")
    print(response)

if __name__ == "__main__":
    main()

```

---

### **Phase 1 Verification: Definition of Done**

Before you sleep on Saturday, verify these 3 things:

1. **Persistence Check:** If you comment out `ingest.py` and run `main.py`, does it still work? (It should, because the data is saved in `chroma_db`).
2. **Context Check:** Modify the `main.py` to print the retrieved documents. Are they actually relevant to the question?

- _Quick debug:_ Change `rag_chain` to just `retriever.invoke(query)` to see exactly what chunks come back.

3. **Hallucination Check:** Ask a question that **isn't** in the PDF (e.g., "How do I bake a cake?"). The bot should say "I don't know based on the handbook."

**You are now ready for Phase 2 (Sunday): The "Ops" Layer.**
_Would you like me to prepare the code for the Sunday phase (Setting up the Ragas Evaluation script)?_
