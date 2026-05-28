from langgraph.graph import StateGraph, END
from graph.state import AgentState

from agents.planner import plan_research
from agents.researcher import research_info
from agents.rag_agent import retrieve_documents
from agents.analyst import analyze_data
from agents.critic import review_draft
from agents.report_generator import generate_report

workflow = StateGraph(AgentState)

workflow.add_node("planner", plan_research)
workflow.add_node("researcher", research_info)
workflow.add_node("rag_agent", retrieve_documents)
workflow.add_node("analyst", analyze_data)
workflow.add_node("critic", review_draft)
workflow.add_node("generator", generate_report)

def route_after_planning(state: AgentState) -> list[str]:
    """
    Decides which data gathering nodes to execute.
    If use_web_search is False, it completely bypasses the Tavily API.
    """
    destinations = ["rag_agent"]
    if state.get("use_web_search", True):
        destinations.append("researcher")
    return destinations

def route_after_critic(state: AgentState) -> str:
    """
    The Gatekeeper logic. Checks if the draft passed QA or hit the revision limit.
    """
    if state.get("draft_approved", False):
        return "generator"
    
    if state.get("revision_count", 0) >= 3:
        print("--- SYSTEM NOTICE: Max revisions reached. Forcing progression. ---")
        return "generator"
    
    return "analyst"

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    route_after_planning,
    {
        "rag_agent": "rag_agent",
        "researcher": "researcher"
    }
)

workflow.add_edge("rag_agent", "analyst")
workflow.add_edge("researcher", "analyst")

workflow.add_edge("analyst", "critic")

workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "generator": "generator",
        "analyst": "analyst"
    }
)

workflow.add_edge("generator", END)

app = workflow.compile()