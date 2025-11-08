# 🤖🧠 BioRAG AI Agent: An Intelligent Clinical Trial Semantic-Retrieval System

## Overview  
This repository develops and implements an **AI Agent** for **clinical trials semantic-retrieval data extraction**.  
It **autonomously** retrieves, classifies, embeds, and performs semantic search extracting data (in stracture format) from PubMed throw **Model Context Protocol** (MCP), functioning as a biomedical **Retrieval-Augmented Generation (RAG)** system.

The **AI Agent**:  
`1.` Converts natural-language prompts into optimized PubMed queries.  
`2.` Retrieves PMIDs/titles/abstracts via the **PubMed API through an MCP server**.  
`3.` Embeds selected records into a normalized FAISS vector db.  
`4.` Classifies and filters retrieved docs based on a given **system prompt**.  
`5.` Performs semantic search retrieving features based on given **system pormpt** (drug names, indications, sponsors, NCT IDs).  

## 🧩 System Architecture  
**`User Prompt -> OpenAI (prompt → PubMed syntax) -> [MCP Server] for PubMed -> RAG Vector DB + Classification -> Semantic Search & Feature Extraction -> Structured Biomedical Metadata`**

## Table of Contents  
1. 🧩 **[MCP Server](#1-mcp-server)** - MCP server description  
2. 👨‍💻 **[Client](#2-client-query0generator-and-retriever)** - Query generator and retriever  
3. 🗂️ **[RAG System](#rag-system-and-classification)** - Retrieval, embeddings & classification
4. 🔍 **[Semantic Retrieval](#semantic-retrieval)** - Semantic search & data extraction  
5. ⚠️ **[Future Enhancements](#future-enhancements)** - Next steps and improvements
6. 🔁 **[Reproducibility](#reproducibility)** - Steps to reproduce the pipeline  
7.    **[Licence](#licence)** - GNU General Public LICENS V3  

### 1. MCP Server  
**Purpose:**  
A Model Context Protocol (MCP) service that queries PubMed metadata.  
Is designed to fetch and preprocess documents using NCBI’s E-utilities (`esearch` + `efetch`) returning PMID/title/abstract metadata.  
**Features**  
- `FastMCP-based` HTTP service
- `Batched (N=100)` PubMed XML parsed (`efetch`)
- Handles labeled abstracts and fallbacks
- Rate-limited (≈3 requests/sec)
- Returns: PMID, title, abstract
- MCP Server active at http://127.0.0.1:8000/mcp  
**Note:**: Abstracts returned as N/A should be additionaly considered in XML parse conditions

### 2. Client - Query generator and retriever
**Purpose:**
Takes user natural language request into a PubMed query, retrieves matching docs via the MCP server, and passes the results for classification and embedding storage.  
It uses `gpt-3.5-turbo` LLM from **OpenAI** to generate precise PubMed syntax.  
The client asynchronously calls the MCP server, receives structured metadata (PMID, title, abstract), and passes it for indexing, embedding, and classification.  
It links user input, PubMed data retrieval, and data storage, **forming the orchestration layer between vector db and AI-driven biomedical knowledge extraction**.  
**Features**   
- OpenAI GPT converts user text to valid PubMed query syntax.  
- **Asynchronously** extracts number of requested results from prompt using regex.  
- Calls the MCP tool via HTTP and gives PubMed syntax.  
- Calls the RAG system which stores results in json (doc metadata with `SHA-256 hash` IDs at title/abstract level for duplicate detection) and `FAISS` (vector db metadata).   

### 3. RAG System and Classification
**Purpose:**
The RAG system performs semantic retrieval and structured information extraction from previously stored PubMed metadata.  
It combines embeddings, vector search, and LLM reasoning to identify biomedical entities from doc abstracts.  
For a given document, it encodes the text into bio-embeddings using OpenAI transformer (`text-embedding-3-small`; 1,536 dims) or a local SentenceTransformer fallback (`all-roberta-large-v1`; 1,024 dims).  
FAISS uses cosine similarity to rank documents by meaning rather, ensuring that semantically related abstracts appear near each other in vector space.    
Then retrieve the douments into batches of 5 and concatenated with the system prompt to form an enriched input for classification.  

Classification combines three procedures (two LLMs and one rule-based using regex) to add variance in the output.  
Furthermore, adds two more characteristics on each doc (`in-scope`: binary, `reason`: text related to in-scope decision).  
- If all procedures match then output considered robust and reason take the word `Match: reason`.  
- If they missmacth (or NULL produced with at least one of the remote LLMs) then the strongest LLM (`gpt-4-mini`) only considered with reason `Unmatched: reason`.
- If rule-based procedure unable to decide then the strongest LLM cosidered (`gpt-4-mini`) and reason `Unrecognised`
**Note**: Performance of the Agent **`(Accuracy: 0.79, Precision: 0.8, Recall: 0.66)`**. Docs assessed: 14 (power is extremely low; more docs needed for evaluation).

#### Classification logic (system prompt) 
- **In-scope**: phase 1–3, randomized, double-blind, placebo-controlled, interventional  
- **Out-of-scope**: biosimilar, generic, post-approval, phase 4, real-world, observational, retrospective, meta-analysis, systematic review, preclinical, in vitro, mice,  
					financial, socioeconomic, sociodemographic, panel, follow-up, pilot study, case report, case series, review article, literature review, animal study, non-industry sponsored
					
**Features**
- Deduplicate by `SHA-256 hash` 
- Vector embedding `(Remote: OpenAI or Local fallback: Rule-based)` 
- FAISS cosine index update   
- Rule-based keyword classification & Dual LLM verification (`gpt-3.5-turbo` and `gpt-4o-mini`)
- Batches of 5 concatenated each time with system prompt for speed and precision
- Metadata JSONL append  
**Note**: Remote (OpenAI) and local (SentenceTransformer) models produce embeddings with different dimensions. Mixing them in one FAISS index causes dimensional mismatches.  
Use common dim embedding models or align dimensions (e.g. PCA) before indexing.

**File outputs**:  
`1.` Output/pubmed_index.faiss: **FAISS cosine index**  
`2.` Output/pubmed_meta.jsonl: **One record per PubMed article**  
**Note**: Each record contains:	**PMID, title, abstract, classification, timestamp**

### 4. Semantic Retrieval and Feature Extraction Agent
**Purpose:**  
Performs semantic similarity search and structured feature extraction from in-scope PubMed abstracts.  
User's prompt is converted into bio-embeddings, in the same way as previously on RAG.  
For each in_scope = 1 doc, it retrieves the top-6 most similar abstracts from the FAISS index.  
These retrieved texts are concatenated with the original prompt to build a context-enriched input.  
The LLM (`gpt-3.5-turbo`) then extracts the requested metadata fields asked from system prompt -> (drug_names, indication, sponsor, nct_id).

All extracted information is finally written into a .jsonl file for downstream use.
**Features**
- Embeds each new query.
- Retrieves semantically related abstracts using FAISS (K = 6).
- Augments the query text with this context.
- Runs two LLMs (Remote: `gpt-3.5-turbo`, or Local fallback: `Hermes-2-Pro-Llama-3-8B`) to extract structured biomedical entities.
- Stores all results.

**File outputs**:  
`1.` Output/features_pubmed_meta.jsonl: **One record per PubMed article**  

## 🧠 AI Agent Behavior
Agent Component Description  
- `Perception`: Reads user queries or abstracts
- `Reasoning`: Uses GPT models to plan searches and classify results  
- `Action`: Retrieves data, extracts knowledge, updates index  
- `Memory`: Maintains FAISS index + JSONL metadata  
- `Reflection`: Cross-verifies classification with rule and LLM logic  
- `Output`: Produces structured biomedical knowledge base  

### 5. ⚠️ Future Enhancements
`1.` Async/Parallel Batching for LLM + Embeddings  
Use asyncio to run LLM classification and embedding requests concurrently, with batched inputs and rate-limit aware throttling.  
`2.` Cloud-Hosted Autonomous Ingestion Agent  
Deploy the pipeline as a scheduled cloud job (e.g., weekly) that runs with a fixed system prompt and automatically fetches, filters, classifies, and stores only newly published PubMed documents since the last run.  
`3.` Higher-Dimensional Embedding Models for Improved Semantic Retrieval  
Use larger embedding models (e.g., text-embedding-3-large or domain-specific biomedical embeddings) to gain richer semantic representation.  
`4.` Higher-Precision LLM Models for Classification & Feature Extraction  

### 6. 🔁 Reproducibility  
Run the following sequencially:  
#### `1.` Create and activate a conda environment
Open Anaconda Prompt and run:  
`conda create -n venv python=3.10`  
`conda activate venv`  

#### `2.` Install required packages  
`pip install -r requirements.txt`  
**Note**: If pip fails, try:  
`conda install --file requirements.txt`  

#### `3.` Start the MCP Server  
`python ./Programs/1.fastmcp_pubmed_server.py`

#### `4.` Run the Client
Open a new terminal and run the Client  
`python ./Programs/2.fastmcp_pubmed_client.py`

#### `5.` Perform Semantic Search
`python ./Programs/3.feature_extraction.py`  

**General Note**: First time run will need time as to download all loac models.  

## Licence  
© 2025 Christos Papagiannopoulos — Licensed under GPL-3.0 — Attribution required.  