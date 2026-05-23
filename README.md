# GenAI LangChain Tutorials 🦜🔗

A hands-on notebook series for building production-grade RAG pipelines, advanced retrieval strategies, and observability-backed evaluations with LangChain 1.0+ (LCEL). All tutorials run on **NVIDIA NIM** (free tier) — no paid API keys required.

---

## 🔒 Secure Secret Management & Environment Configuration

To keep applications production-ready and secure, **never hardcode API keys, credentials, or sensitive configurations** in your notebooks or source files. 

All notebooks in this repository are pre-configured to dynamically load secrets from a local `.env` environment file.

### 1. Configure Your Secrets File
Copy the provided `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Open `.env` and fill in your keys:
```ini
# NVIDIA NIM API Key (Get a free key at build.nvidia.com)
NVIDIA_API_KEY=nvapi-your-key-here

# comet.com Opik Observability Keys
OPIK_API_KEY=your-opik-key-here

# Optional: Tavily Search API Key (tavily.com)
TAVILY_API_KEY=your-tavily-key-here

# Optional: Pinecone Vector Database Key
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_INDEX_NAME=ragudsclassdemo
```

### 2. Standard Secure Loading Pattern in Python
To load these secrets securely inside any Python script or Jupyter Notebook cell, use the `python-dotenv` library to inject them directly into system environment variables:

```python
import os
from dotenv import load_dotenv

# Load all environment variables from the .env file
load_dotenv()

# Access keys securely without exposing them in your code
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
opik_api_key = os.getenv("OPIK_API_KEY")

if not nvidia_api_key:
    raise ValueError("❌ NVIDIA_API_KEY is missing! Please configure it in your .env file.")
```

---

## 🚀 Quick Start

### 1. Install Dependencies
Make sure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter notebook
```

---

## 📓 Notebooks Curriculum & Quick-Tutorial

All notebooks are organized into a logical 14-step sequential curriculum:

### 1. Core LangChain & RAG Basics
| # | Topic | Key Concept | File Link |
|---|---|---|---|
| **01** | Introduction & Fundamentals | Prompts, LLMs, Output Parsers, LCEL | [01_Introduction_and_Fundamentals.ipynb](01_Introduction_and_Fundamentals.ipynb) |
| **02** | Document Loaders | Extracting text from PDFs and Web pages | [02_Document_Loaders.ipynb](02_Document_Loaders.ipynb) |
| **03** | Text Splitting Strategies | Recursive character splitting, overlap | [03_Text_Splitting_Strategies.ipynb](03_Text_Splitting_Strategies.ipynb) |
| **04** | Embeddings & Representations | Text-to-vector space, Cosine similarity | [04_Embeddings_and_Vector_Representations.ipynb](04_Embeddings_and_Vector_Representations.ipynb) |
| **05** | Vector Stores | Storing and searching vectors locally | [05_Vector_Stores.ipynb](05_Vector_Stores.ipynb) |
| **06** | Simple RAG with LangChain | Connecting retriever and LLM in a chain | [06_Simple_RAG_LangChain.ipynb](06_Simple_RAG_LangChain.ipynb) |
| **07** | Simple RAG with NVIDIA NIM | Deploying NIM-optimized RAG chains | [07_Simple_RAG_NVIDIA_NIM.ipynb](07_Simple_RAG_NVIDIA_NIM.ipynb) |
| **08** | Local RAG with Ollama | Completely local, offline RAG pipeline | [08_Local_RAG_Ollama.ipynb](08_Local_RAG_Ollama.ipynb) |

### 2. Advanced RAG Techniques
| # | Topic | Key Concept | File Link |
|---|---|---|---|
| **09** | Query Transformations | Rewrite-Retrieve-Read, Step-back, Sub-queries | [09_Query_Transformations.ipynb](09_Query_Transformations.ipynb) |
| **10** | Fusion Retrieval | Multi-query generation, Reciprocal Rank Fusion (RRF) | [10_Fusion_Retrieval.ipynb](10_Fusion_Retrieval.ipynb) |
| **11** | Reranking | Cross-encoder relevance context ordering | [11_Reranking.ipynb](11_Reranking.ipynb) |
| **12** | External Index Retrievers | Live searchers (ArXiv, Wikipedia, Tavily) | [12_External_Index_Retrievers.ipynb](12_External_Index_Retrievers.ipynb) |

### 3. RAG Evaluation & Observability
| # | Topic | Key Concept | File Link |
|---|---|---|---|
| **13** | RAG Evaluation with Ragas | Ground-truth datasets & Ragas evaluation | [13_RAG_Evaluation_with_Ragas.ipynb](13_RAG_Evaluation_with_Ragas.ipynb) |
| **14** | RAGAS Metrics Deep Dive | Faithfulness, Answer Relevancy, Context Recall | [14_RAGAS_Metrics_Deep_Dive.ipynb](14_RAGAS_Metrics_Deep_Dive.ipynb) |

---

## 📖 Quick Tutorial: Core Concepts & Python Code Patterns

### Module 1: Core LangChain & RAG Basics

#### 01. Introduction & Fundamentals
* **Concept**: The foundational building blocks of LangChain 1.0+ centered around **LCEL** (LangChain Expression Language). It uses the pipe operator (`|`) to stream data between Prompt Templates, Large Language Models (LLMs), and Output Parsers.
* **Code Pattern**:
  ```python
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_nvidia_ai_endpoints import ChatNVIDIA
  from langchain_core.output_parsers import StrOutputParser

  prompt = ChatPromptTemplate.from_template("Summarize this: {topic}")
  llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0)
  chain = prompt | llm | StrOutputParser()
  response = chain.invoke({"topic": "LangChain LCEL syntax"})
  ```

#### 02. Document Loaders
* **Concept**: Extracting unstructured data from various sources (such as PDFs, HTML, text files) and converting them into LangChain `Document` objects containing the raw text content and metadata (like source file or page number).
* **Code Pattern**:
  ```python
  from langchain_community.document_loaders import PyPDFLoader

  loader = PyPDFLoader("sample_data/document.pdf")
  documents = loader.load()  # Returns List[Document]
  ```

#### 03. Text Splitting Strategies
* **Concept**: LLMs have limited context windows and retrieval is more precise on smaller segments. **RecursiveCharacterTextSplitter** chunking recursively breaks paragraphs, lines, and sentences into chunks of defined size (e.g. 1024 characters) while maintaining overlapping context (e.g. 128 characters) so information is not cut in half.
* **Code Pattern**:
  ```python
  from langchain_text_splitters import RecursiveCharacterTextSplitter

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
  chunks = text_splitter.split_documents(documents)
  ```

#### 04. Embeddings & Representations
* **Concept**: Embeddings map text chunks into high-dimensional vector spaces (e.g., 1536 dimensions) where semantic similarity represents geometric proximity. We measure similarity between user query and text vectors using **Cosine Similarity**.
* **Code Pattern**:
  ```python
  from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

  embeddings = NVIDIAEmbeddings(model="nvidia/llama-nemotron-embed-1b-v2")
  vector = embeddings.embed_query("What are embeddings?")
  ```

#### 05. Vector Stores (FAISS, Chroma)
* **Concept**: Dedicated databases engineered to index and store millions of high-dimensional vectors, enabling extremely fast similarity searches.
* **Code Pattern**:
  ```python
  from langchain_community.vectorstores import FAISS

  vectorstore = FAISS.from_documents(chunks, embeddings)
  vectorstore.save_local("faiss_index")
  ```

#### 06 & 07. Simple RAG with LangChain & NVIDIA NIM
* **Concept**: End-to-end RAG pipeline. The user query retrieves similar document chunks from the vector store, wraps them in a prompt as context, and feeds them into the LLM to generate a factual response.
* **Code Pattern**:
  ```python
  retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
  prompt = ChatPromptTemplate.from_template("Context:\n{context}\n\nQuestion: {question}")
  
  # Robust LLM with automatic retry logic for API stability
  llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0).with_retry(
      stop_after_attempt=5, wait_exponential_jitter=True
  )
  
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)
      
  rag_chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )
  response = rag_chain.invoke("What is the platform's SLA?")
  ```

#### 08. Local RAG with Ollama
* **Concept**: Orchestrating a private, completely offline, and local RAG system running LLMs (like Llama 3) and Embeddings models locally on your system using Ollama.
* **Code Pattern**:
  ```python
  from langchain_community.chat_models import ChatOllama
  from langchain_community.embeddings import OllamaEmbeddings

  local_llm = ChatOllama(model="llama3", temperature=0)
  local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
  ```

---

### Module 2: Advanced RAG Techniques

#### 09. Query Transformations
* **Concept**: Rewriting, expanding, or decomposing a query to improve retrieval precision. Includes **Rewrite-Retrieve-Read** (rewriting vague queries), **Step-Back Prompting** (generating high-level abstraction queries), and **Sub-Query Generation** (splitting complex questions into smaller parts).
* **Code Pattern**:
  ```python
  rewrite_prompt = ChatPromptTemplate.from_template("Rewrite this search query: {query}")
  rewriter_chain = rewrite_prompt | llm | StrOutputParser()
  better_query = rewriter_chain.invoke({"query": "sla details"})
  ```

#### 10. Fusion Retrieval
* **Concept**: Generating multiple search query variations, executing parallel retrievals for all queries, and combining results using **Reciprocal Rank Fusion (RRF)** to rank document chunks based on their consensus relevance across queries.
* **Code Pattern**:
  ```python
  def rrf(results: list[list[Document]], k=60):
      # Reciprocal Rank Fusion algorithm scoring and combining doc lists
      scores = {}
      for doc_list in results:
          for rank, doc in enumerate(doc_list):
              scores[doc.page_content] = scores.get(doc.page_content, 0) + 1.0 / (k + rank + 1)
      return scores
  ```

#### 11. Reranking
* **Concept**: Similarity search (Bi-Encoders) is fast but misses deep semantic context. Reranking uses a slower, highly accurate **Cross-Encoder** model to evaluate the relationship between the query and retrieved documents, re-ordering chunks to place the most relevant information at the top.
* **Code Pattern**:
  ```python
  from langchain_nvidia_ai_endpoints import NVIDIARerank

  reranker = NVIDIARerank(model="nvidia/llama-nemotron-rerank-1b-v2")
  reranked_docs = reranker.compress_documents(query="SLA details", documents=retrieved_docs)
  ```

#### 12. External Index Retrievers
* **Concept**: Searching live external databases (e.g. academic papers via **ArxivRetriever**, general knowledge via **WikipediaRetriever**, or live web indexing via **TavilySearchAPIRetriever**) instead of static local vector stores.
* **Code Pattern**:
  ```python
  from langchain_community.retrievers import WikipediaRetriever
  
  # Configure unique User-Agent to avoid Wikipedia API HTTP 429 rate limiting
  import wikipedia
  wikipedia.set_user_agent('GenAILangChainTutorialsBot/1.0 (contact: name@domain.com)')

  wiki_retriever = WikipediaRetriever(top_k_results=2)
  docs = wiki_retriever.invoke("large language models")
  ```

---

### Module 3: RAG Evaluation & Observability

#### 13. RAG Evaluation with Ragas
* **Concept**: Rigorous testing of RAG performance. This involves generating synthetic test datasets (Question, Context, Ground Truth answers) using an LLM, and executing evaluations with the **Ragas** framework.
* **Code Pattern**:
  ```python
  from ragas import evaluate
  from datasets import Dataset

  data = {
      "question": ["What is the SLA?", "How is data encrypted?"],
      "contexts": [["CloudFlow guarantees 99.99% uptime SLA."], ["AES-256 is used for rest encryption."]],
      "answer": ["99.99% SLA is guaranteed.", "AES-256 rest encryption."],
      "ground_truth": ["99.99% SLA.", "AES-256 Rest Encryption."]
  }
  dataset = Dataset.from_dict(data)
  result = evaluate(dataset=dataset, metrics=[Faithfulness(), AnswerRelevancy()])
  ```

#### 14. Ragas Metrics Deep Dive
* **Concept**: Evaluating quality across Ragas metrics:
  * **Faithfulness**: Measures if the generated answer is strictly supported by the retrieved context (detects hallucinations).
  * **Answer Relevancy**: Measures if the generated answer addresses the actual user question.
  * **Context Recall**: Measures if the retriever retrieved all necessary information to answer the question (compared against Ground Truth).
  * **Context Precision**: Measures if the highly relevant chunks are ranked at the top of the context window.
  * **Noise Sensitivity**: Measures the model's robustness against irrelevant, noisy text inside the context.

---

## 🔧 Migration to NVIDIA NIM

This repo includes a automated migration script to convert legacy notebooks (configured for OpenAI, Google Gemini, or Ollama) to use high-performance **NVIDIA NIM** endpoints:

```bash
python migrate_to_nvidia.py
```

The script automatically parses the notebook cells as JSON, re-writes legacy components, and saves them safely:
- Replaces `ChatOpenAI` / `ChatGoogleGenerativeAI` → `ChatNVIDIA`
- Replaces `OpenAIEmbeddings` → `NVIDIAEmbeddings`
- Dynamically maps OpenAI/Gemini models to the latest active successor NVIDIA models:

| Legacy Model | Active NVIDIA successor |
|---|---|
| `gpt-4` / `gpt-4o` | `meta/llama-3.3-70b-instruct` |
| `gpt-3.5-turbo` / `gpt-4o-mini` | `meta/llama-3.1-8b-instruct` |
| `text-embedding-3-small` / `large` | `nvidia/llama-nemotron-embed-1b-v2` |

---

## 🛠 Technology Stack

* **LangChain** — LCEL chains, loaders, splitters, retrievers
* **NVIDIA NIM** — Hosted inference & embedding endpoints (free tier)
* **FAISS / Chroma** — Local vector indexes
* **Ragas** — RAG evaluations and assertions
* **Opik** — Comet-backed Observability and experiment tracking
* **Sentence-Transformers** — Semantic local similarity models
