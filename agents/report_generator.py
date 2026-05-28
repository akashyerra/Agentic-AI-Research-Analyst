import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from graph.state import AgentState

load_dotenv()

def generate_report(state: AgentState) -> dict:
    """
    LangGraph node function that takes the approved analytical draft
    and formats it into a highly structured, business-grade markdown report.
    """
    print("--- REPORT GENERATOR: Formatting Final Output ---")
    
    query = state["query"]
    approved_draft = state.get("analysis", "")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    system_prompt = """You are an Executive Corporate Communicator.
    Your job is to take an analytical draft and format it into a pristine, business-grade Markdown report.
    
    DO NOT add any new facts, numbers, or external information. 
    Rely ONLY on the provided draft.
    
    You MUST structure the report exactly with these Markdown headings:
    # Executive Summary
    ## Market Analysis
    ## Key Findings
    ## AI-Generated Recommendations
    ## Risk Assessment
    ## Source References
    ## Confidence Score (Assign a score out of 100% based on data quality)
    
    CRITICAL INSTRUCTION: Output ONLY the markdown report. Do not include introductory phrases like "Here is the report" or concluding phrases like "Let me know if you need anything else."
    """

    human_prompt = f"""
    ORIGINAL QUERY: {query}
    
    APPROVED ANALYTICAL DRAFT:
    {approved_draft}
    
    Format this into the final professional report.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm.invoke(messages)

    print("    -> Final report generated successfully.")
    
    return {"final_report": response.content}