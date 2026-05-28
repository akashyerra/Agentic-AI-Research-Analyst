import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from graph.state import AgentState

load_dotenv()

class ResearchPlan(BaseModel):
    steps: list[str] = Field(description="A sequential list of specific research tasks.")

def plan_research(state: AgentState) -> dict:
    """
    LangGraph node function that takes the current state,
    analyzes the query, and generates a structured research plan.
    """
    print("--- PLANNER AGENT: Deconstructing Query ---")
    query = state["query"]
    use_web_search = state.get("use_web_search", True) 

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    structured_llm = llm.with_structured_output(ResearchPlan)

    available_tools = "1. Document Retrieval (for extracting context from uploaded PDFs)"
    if use_web_search:
        available_tools += "\n2. Live Web Search (for recent news, market data, and current events)"
    
    system_prompt = f"""You are a Senior AI Project Manager. 
    Your job is to break down the user's research query into a logical, step-by-step plan.
    The system has two tools available:
    {available_tools}
    
    Create a highly specific, actionable checklist of 3 to 5 steps.
    Do NOT answer the query yourself. Only provide the plan.
    If a tool is not listed above, DO NOT include steps that require it."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate a research plan for the following query: {query}")
    ])

    chain = prompt | structured_llm
    response = chain.invoke({"query": query})

    return {"plan": response.steps, "revision_count": 0}