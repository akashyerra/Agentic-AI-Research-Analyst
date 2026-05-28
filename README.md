# 🤖 Agentic AI Research Analyst

An enterprise-grade, multi-agent reasoning system designed to autonomously conduct deep research, parse proprietary documents, synthesize data, and generate business-grade Markdown reports.

Built with **LangGraph**, **LangChain**, and **Streamlit**.

---

# 🛑 The Problem

Modern professionals such as analysts, consultants, and researchers spend countless hours manually researching market trends, reading large PDFs, comparing conflicting data points, and formatting reports.

While standard AI chatbots exist, they typically fail in enterprise environments because they:

- Hallucinate when missing data
- Lack orchestration to reason across complex, multi-step workflows
- Cannot self-correct or audit their own conclusions
- Struggle to combine live web data with proprietary internal documents (RAG)

---

# 💡 The Solution

The **Agentic AI Research Analyst** operates not as a single chatbot, but as a digital micro-company.

It utilizes a stateful, cyclical LangGraph architecture to coordinate a team of specialized AI agents:

---

## 🧠 Multi-Agent Architecture

### 1. Planner Agent
Deconstructs the user's complex query into a step-by-step actionable research plan.

### 2. Researcher Agent
Surfs the live web using Tavily to gather exhaustive, real-time market data and news.

### 3. RAG Agent
Queries a local FAISS vector database to extract semantic context from user-uploaded PDFs.

### 4. Analysis Agent
Synthesizes the raw data streams into a highly detailed analytical draft with strict inline citations.

### 5. Critic Agent (QA)
Acts as a zero-temperature gatekeeper that cross-references the draft against the raw data, rejecting hallucinations and forcing revisions until factual alignment is achieved.

### 6. Report Generator
Formats the final approved draft into a clean, structured Markdown report.

---

# ✨ Key Enterprise Features

## 🔒 Closed-Book Mode
A strict UI toggle that disables web access, forcing the system to rely exclusively on uploaded internal documents for compliance and privacy.

## 🛡️ Automated Self-Correction
The Critic Agent prevents hallucinations and unnecessary API billing loops by enforcing strict revision limits and returning targeted feedback to the reasoning engine.

## 📚 Local RAG Pipeline
Privacy-first document chunking and semantic retrieval using local HuggingFace embeddings.

## ⚡ Stateful LangGraph Orchestration
Cyclical workflows allow agents to revise, retry, and collaboratively reason through complex analytical tasks.

## 📄 Business-Grade Report Generation
Produces polished Markdown reports ready for enterprise documentation and presentations.

---

# 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Orchestration | LangGraph |
| Framework | LangChain |
| LLM Engine | Groq (`llama-3.3-70b-versatile`) |
| Search API | Tavily |
| Vector Database | FAISS |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Frontend | Streamlit |
| Language | Python |

---

# 🚀 Local Setup & Installation

## 1. Prerequisites

Ensure you have the following installed:

- Python 3.9+
- Groq API Key
- Tavily API Key

### API Key Links

- Groq: https://console.groq.com/
- Tavily: https://app.tavily.com/

---

## 2. Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-research-analyst.git

cd agentic-research-analyst
```

---

## 3. Create a Virtual Environment

### macOS/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
GROQ_API_KEY="your_groq_api_key_here"

TAVILY_API_KEY="your_tavily_api_key_here"
```

---

## 6. Run the Application

```bash
streamlit run app.py
```

---

# 💻 Usage Guide

## 1. Upload Context (Optional)

Upload PDF documents such as:
- Financial reports
- Research papers
- Internal business documents

The system automatically:
- Extracts text
- Chunks documents
- Generates embeddings
- Indexes them into FAISS

---

## 2. Configure Constraints

Use the UI toggle to:

- Enable live web search
- Disable web access for strict internal-only analysis

---

## 3. Run Analysis

Example Prompt:

```text
Analyze NVIDIA's growth potential using the latest AI market trends and the uploaded Q3 earnings report.
```

---

## 4. Observe Agent Workflow

Watch the LangGraph state machine orchestrate tasks between:
- Planner
- Researcher
- RAG Agent
- Analyst
- Critic

---

## 5. Export Results

Download the final QA-approved report directly as a Markdown file.

---

# 🧠 System Workflow

```text
User Query
    │
    ▼
Planner Agent
    │
    ├──► Researcher Agent (Web Search)
    │
    ├──► RAG Agent (PDF Retrieval)
    │
    ▼
Analysis Agent
    │
    ▼
Critic Agent (QA Validation)
    │
    ├──► Reject & Revise
    │
    ▼
Report Generator
    │
    ▼
Final Markdown Report
```

---

# 📁 Project Structure

```plaintext
agentic-research-analyst/
│
├── app.py
├── .env
├── requirements.txt
│
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   ├── rag_agent.py
│   ├── analyst.py
│   ├── critic.py
│   └── report_generator.py
│
├── graph/
│   ├── state.py
│   └── workflow.py
│
├── rag/
│   ├── embeddings.py
│   ├── retriever.py
│   └── vector_store.py
│
├── utils/
│   ├── pdf_loader.py
│   └── helpers.py
│
└── data/
```

---

# 🔄 LangGraph Workflow Overview

The system uses a cyclical LangGraph workflow where:

- Agents communicate through shared state
- The Critic Agent can trigger revisions
- Conditional routing determines the next action
- The graph maintains memory across execution cycles

This enables:
- Reflection
- Self-correction
- Multi-step reasoning
- Enterprise-grade reliability

---


# 🔒 Privacy & Security

- Local FAISS vector storage
- Local embedding generation
- Optional web-disabled mode
- No external document persistence
- Enterprise-friendly architecture

---

# 🧪 Future Improvements

- Multi-document conversational memory
- SQL database integration
- Persistent vector storage
- Docker deployment
- Multi-user authentication
- Agent performance analytics
- Streaming agent execution visualization
- Human-in-the-loop approval workflows

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a pull request

---
