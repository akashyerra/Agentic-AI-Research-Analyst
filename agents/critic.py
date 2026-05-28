import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from graph.state import AgentState

load_dotenv()

class CritiqueOutput(BaseModel):
    is_acceptable: bool = Field(description="True if the draft is factual and answers the query perfectly. False if it contains errors or hallucinations.")
    feedback: str = Field(description="Specific, actionable feedback on what needs to be fixed. Leave empty if acceptable.")

def review_draft(state: AgentState) -> dict:
    """
    LangGraph node function that reviews the analytical draft against the raw data
    to detect hallucinations and ensure the query was fully answered.
    """
    print("--- CRITIC AGENT: Auditing Draft ---")
    
    query = state["query"]
    draft = state.get("analysis", "")
    research_data = state.get("research_data", "")
    retrieved_docs = state.get("retrieved_docs", "")
    current_revisions = state.get("revision_count", 0)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    structured_llm = llm.with_structured_output(CritiqueOutput)

    # system_prompt = """You are a ruthless, highly meticulous QA Reviewer for a Senior Analyst team.
    # Your job is to read the Analyst's draft and compare it STRICTLY against the provided source data.
    
    # FAIL THE DRAFT IF:
    # 1. It contains facts, numbers, or claims not found in the source data (Hallucination).
    # 2. It fails to directly answer the user's original query.
    # 3. It is poorly structured or lacks professional tone.
    
    # If it fails, provide exact, harsh, actionable feedback on what to change.
    # If it passes all criteria, mark it as acceptable."""

    system_prompt = """You are a meticulous QA Reviewer auditing an Analyst's draft.
    Your job is to compare the draft STRICTLY against the provided source data.
    
    FAIL THE DRAFT IF (Return False):
    1. HALLUCINATION: The draft contains numbers, dates, or factual claims that DO NOT exist in the source data.
    2. CONTRADICTION: The draft says the opposite of what the data says.
    
    PASS THE DRAFT IF (Return True):
    1. The draft accurately reflects the source data, even if the source data is sparse.
    2. The draft correctly admits that certain information is unavailable.
    
    Do NOT fail the draft just because the original search data was weak. Only fail it if the Analyst lied or fabricated information.
    """

    human_prompt = f"""
    USER QUERY: {query}
    
    RAW SOURCE DATA (Web & Internal):
    {retrieved_docs}
    ---
    {research_data}
    
    ANALYST'S DRAFT:
    {draft}
    
    Evaluate the draft based on the source data.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = structured_llm.invoke(messages)

    if response.is_acceptable:
        print("    -> QA PASSED. Draft approved.")
        return {
            "critique": "Draft approved.",
            "draft_approved": True
        }
    else:
        print(f"    -> QA FAILED. Feedback: {response.feedback}")
        return {
            "critique": response.feedback,
            "draft_approved": False,
            "revision_count": current_revisions + 1 # Increment the loop counter
        }