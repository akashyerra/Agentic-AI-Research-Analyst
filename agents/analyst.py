import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState

load_dotenv()

def analyze_data(state: AgentState) -> dict:
    """
    LangGraph node function that takes raw web data and internal document context,
    synthesizing it into a comprehensive analytical draft.
    """
    print("--- ANALYSIS AGENT: Synthesizing Data ---")
    
    query = state["query"]
    
    research_data = state.get("research_data", "No external web research was conducted.")
    retrieved_docs = state.get("retrieved_docs", "No internal documents were provided.")

    research_data = research_data[:15000]
    retrieved_docs = retrieved_docs[:15000]

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = """You are a Senior Business/Research Analyst. 
    Your objective is to write a comprehensive, analytical draft addressing the user's query.
    
    RULES:
    1. You MUST rely EXCLUSIVELY on the provided Web Research Data and Internal Document Data.
    2. Do NOT hallucinate or invent statistics, companies, or events. If the data doesn't answer the query fully, acknowledge the gaps.
    3. Compare and contrast data points if both web and internal data are present.
    4. Structure your response logically with clear headings (e.g., Overview, Key Findings, Market Trends, Risks/Gaps).
    5. This is a draft for internal review, focus on depth, accuracy, and insight.
    IN-LINE CITATIONS: Every single claim, statistic, or fact MUST be followed by a bracketed citation indicating its source (e.g., [Web Data] or [Internal Doc]).
    
    If a data source explicitly states "No data", ignore that source and rely on the available one.
    """

    human_prompt = f"""
    USER QUERY: {query}
    
    =========================================
    SOURCE 1: INTERNAL DOCUMENT DATA (RAG)
    =========================================
    {retrieved_docs}
    
    =========================================
    SOURCE 2: LIVE WEB RESEARCH DATA
    =========================================
    {research_data}
    
    =========================================
    
    Synthesize this information into a detailed analytical draft.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)

    print("    -> Draft synthesis complete.")
    return {"analysis": response.content}