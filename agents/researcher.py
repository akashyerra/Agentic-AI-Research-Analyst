import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from graph.state import AgentState

load_dotenv()

# tavily_tool = TavilySearch(
#     max_results=5, 
#     search_depth="advanced",
#     include_answer=True 
# )
tavily_tool = TavilySearch(max_results=3)

def research_info(state: AgentState) -> dict:
    """
    LangGraph node function that takes the current state, reads the plan,
    and executes live web searches to gather raw data.
    """
    print("--- RESEARCH AGENT: Gathering Live Data ---")
    
    query = state["query"]
    plan = state.get("plan", [])
    
    plan_text = "\n".join([f"- {step}" for step in plan])

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    llm_with_tools = llm.bind_tools([tavily_tool])

    # system_prompt = """You are a Senior Research Analyst executing a research plan.
    # You have access to a web search tool (Tavily). 
    # Your goal is to gather raw, factual data, statistics, and recent news regarding the user's query.
    
    # Do NOT write the final report. 
    # ONLY return the raw facts, data points, and context you found so the Analysis team can process it.
    # If you do not find relevant information, state that clearly."""

    system_prompt = """You are a Senior Web Researcher.
    Your goal is to gather exhaustive, factual data regarding the user's query.
    
    RULES:
    1. Break complex queries down and execute MULTIPLE specific searches if necessary.
    2. Focus on highly credible sources (financial reports, official documentation, verified news).
    3. Do not summarize too heavily. Do NOT write the final report. ONLY return the raw facts, data points, and context you found so the Analysis team can process it. If you do not find relevant information, state that clearly.
    """

    human_prompt = f"""
    Original Request: {query}
    
    Research Plan to Execute:
    {plan_text}
    
    Execute the necessary searches and provide a comprehensive summary of the raw data.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    response = llm_with_tools.invoke(messages)
    
    research_results = []
    
    if response.tool_calls:
        print(f"    -> Executing {len(response.tool_calls)} search queries...")
        for tool_call in response.tool_calls:
            search_query = tool_call['args'].get('query')
            print(f"       Searching: '{search_query}'")
            
            search_output = tavily_tool.invoke({"query": search_query})
            
            research_results.append(f"Query: {search_query}\nResults: {search_output}\n")
    else:
        research_results.append(response.content)

    compiled_research = "\n---\n".join(research_results)

    return {"research_data": compiled_research}