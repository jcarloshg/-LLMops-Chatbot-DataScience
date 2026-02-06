# **The LLMOps Lifecycle Visualized**

Before we dive into the details, here is the high-level flow of how an LLM project moves from idea to production.

```text
+------------------+      +------------------+      +------------------+
| 1. Model         |      | 2. Data Prep &   |      | 3. Customization |
|    Selection     | ---> |    Prompt Eng.   | ---> |    (RAG / FT)    |
+------------------+      +------------------+      +------------------+
                                                             |
+------------------+      +------------------+               v
| 6. Monitoring &  |      | 5. Deployment &  |      +------------------+
|    Observability | <--- |    Serving       | <--- | 4. Evaluation &  |
+------------------+      +------------------+      |    Testing       |
                                                    +------------------+

```

---

### **Phase 1: Model Selection & Foundation**

| Section            | Details                                                                                                                                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**    | This is the "shopping" phase. You decide whether to use a proprietary model (like GPT-4) or an open-source model (like Llama 3 or Mistral). This decision dictates your cost, privacy, and performance ceiling. |
| **Key Activities** | • Defining use-cases (e.g., chat, summary, coding).<br>                                                                                                                                                         |

<br>• Comparing benchmarks (MMLU, HumanEval).<br>

<br>• Checking licensing (Commercial vs. Research only).<br>

<br>• Estimating costs (Token pricing vs. GPU rental). |
| **Tools** | • **Hugging Face Hub:** The "GitHub of AI" where you find models.<br>

<br>• **Ollama:** A tool to run open-source models locally on your laptop.<br>

<br>• **Open LLM Leaderboard:** Rankings of top-performing models. |
| **Prerequisites** | Basic understanding of **Transformer architecture** (context window, parameters) and Python basics. |
| **Tips & Best Practices** | • **Start small:** Don't default to the largest model. A 7B or 8B parameter model is often enough for simple tasks.<br>

<br>• **Check the License:** "Open Weights" doesn't always mean open source for commercial use. |
| **Learning Resources** | **[Hugging Face Tasks Guide](https://www.google.com/search?q=https://huggingface.co/docs/transformers/tasks)** – Excellent docs on what different models do. |

---

### **Phase 2: Data Preparation & Prompt Engineering**

| Section            | Details                                                                                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**    | Models are only as good as the context you give them. In this phase, you prepare your data (documents, PDFs, SQL) and design the instructions (prompts) that guide the model's behavior. |
| **Key Activities** | • **Chunking:** Breaking long documents into smaller pieces.<br>                                                                                                                         |

<br>• **Embedding:** Converting text into numbers (vectors).<br>

<br>• **Prompting:** Writing and refining system prompts (Zero-shot, Few-shot). |
| **Tools** | • **LangChain / LlamaIndex:** Frameworks to connect data to LLMs.<br>

<br>• **ChromaDB / Qdrant:** Vector databases to store your data embeddings.<br>

<br>• **Unstructured.io:** Tool to clean messy data (PDFs, HTML). |
| **Prerequisites** | Knowledge of **APIs** (JSON, REST) and basic **NLP concepts** (tokenization). |
| **Tips & Best Practices** | • **Garbage In, Garbage Out:** If your text chunks are messy, the AI will be confused.<br>

<br>• **Version Control Prompts:** Treat prompts like code. Save versions so you can roll back if a change breaks the output. |
| **Learning Resources** | **[DeepLearning.AI: ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)** – A classic free course. |

---

### **Phase 3: Customization (RAG & Fine-Tuning)**

| Section            | Details                                                                                                                                                                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**    | Generic models don't know your private data. Here, you teach the model your specific domain. You usually start with RAG (Retrieval Augmented Generation) to give it "textbook" access, or Fine-Tuning to change its "style" or behavior. |
| **Key Activities** | • **RAG Pipeline:** Setting up the retriever to find relevant data.<br>                                                                                                                                                                  |

<br>• **Fine-Tuning:** Training the model on dataset pairs (Input -> Output) using techniques like LoRA (Low-Rank Adaptation). |
| **Tools** | • **SentenceTransformers:** For creating high-quality embeddings.<br>

<br>• **Axolotl / Unsloth:** Beginner-friendly tools to fine-tune models efficiently.<br>

<br>• **Haystack:** An alternative to LangChain for building pipelines. |
| **Prerequisites** | Understanding of **Vector Search** (similarity) and basic **Machine Learning concepts** (training vs. inference). |
| **Tips & Best Practices** | • **RAG First:** Always try RAG before fine-tuning. It is cheaper, faster, and reduces hallucinations.<br>

<br>• **Data Quality:** For fine-tuning, 100 high-quality examples are better than 10,000 poor ones. |
| **Learning Resources** | **[Unsloth Github & Wiki](https://github.com/unslothai/unsloth)** – Currently the easiest way to start fine-tuning. |

---

### **Phase 4: Evaluation & Testing**

| Section            | Details                                                                                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**    | How do you know the AI isn't lying? Unlike standard code, LLM output is probabilistic. This phase uses metrics and "LLM-as-a-Judge" to verify accuracy, faithfulness, and safety. |
| **Key Activities** | • **Unit Testing:** checking for specific keywords or formats.<br>                                                                                                                |

<br>• **Benchmarking:** Running the model against golden datasets.<br>

<br>• **Red Teaming:** Trying to trick the model into saying harmful things. |
| **Tools** | • **Ragas:** specifically calculates metrics for RAG (e.g., did the answer actually use the retrieved context?).<br>

<br>• **DeepEval:** An open-source testing framework for LLMs.<br>

<br>• **Arize Phoenix:** For tracing and visualizing where an LLM went wrong. |
| **Prerequisites** | Basic **Statistics** (understanding non-deterministic outputs) and critical thinking. |
| **Tips & Best Practices** | • **Don't trust "Vibes":** "It looks good to me" is not a metric. Use a framework.<br>

<br>• **Test for Regressions:** Ensure that fixing one prompt didn't break a different use case. |
| **Learning Resources** | **[Ragas Documentation](https://docs.ragas.io/en/stable/)** – Learn how to mathematically score your RAG pipeline. |

---

### **Phase 5: Deployment & Serving**

| Section            | Details                                                                                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Description**    | Moving the model from a Python notebook to a real server where users can access it. This involves optimizing the model to run fast (low latency) and handling multiple users at once (concurrency). |
| **Key Activities** | • **Quantization:** Reducing model precision (e.g., 4-bit) to save memory.<br>                                                                                                                      |

<br>• **Containerization:** Wrapping the app in Docker.<br>

<br>• **API Wrapper:** Exposing the model via REST API. |
| **Tools** | • **vLLM:** A high-speed library for serving open-source models.<br>

<br>• **Docker:** Standard tool for packaging software.<br>

<br>• **FastAPI:** Python framework to build the API endpoints. |
| **Prerequisites** | **Docker** fundamentals and **Cloud basics** (knowing what a GPU instance is). |
| **Tips & Best Practices** | • **Watch Memory:** LLMs are memory hungry. Ensure your GPU VRAM > Model Size.<br>

<br>• **Batching:** Use tools like vLLM that handle "continuous batching" to serve many users without crashing. |
| **Learning Resources** | **[vLLM Documentation](https://docs.vllm.ai/en/latest/)** – The gold standard for open-source serving currently. |

---

### **Phase 6: Monitoring & Observability**

| Section            | Details                                                                                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Description**    | Once live, you must watch the AI. Is it getting slower? Is it starting to be rude? Is it costing too much money? This phase ensures reliability in the real world. |
| **Key Activities** | • **Tracing:** Tracking a request through the whole chain.<br>                                                                                                     |

<br>• **Cost Tracking:** Monitoring token usage per user.<br>

<br>• **Drift Detection:** Checking if user inputs are changing significantly over time. |
| **Tools** | • **LangSmith:** Excellent for debugging and tracing (generous free tier).<br>

<br>• **Grafana / Prometheus:** Standard industry tools for visualizing metrics.<br>

<br>• **OpenTelemetry:** Standard for collecting traces. |
| **Prerequisites** | Understanding of **Logging** and basic system metrics (Latency, Throughput). |
| **Tips & Best Practices** | • **Log Inputs/Outputs:** You can't debug an LLM error if you don't know what the user asked.<br>

<br>• **Set Budget Alerts:** API costs can skyrocket overnight. Set hard limits. |
| **Learning Resources** | **[LangChain/LangSmith Walkthroughs](https://docs.smith.langchain.com/)** – Great for seeing how tracing works in practice. |
