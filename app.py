import os
import streamlit as st
import time
import warnings
from graph.workflow import app as compiled_graph
from rag.vector_store import create_vector_store

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

st.set_page_config(
    page_title="Agentic Research Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    .report-container { background-color: #f8f9fa; padding: 2rem; border-radius: 10px; color: #1e1e1e;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown("Control the behavior of the multi-agent system.")
    
    st.divider()
    st.subheader("📄 Knowledge Base (RAG)")
    st.markdown("Upload internal PDFs to ground the AI's analysis.")
    
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing and indexing document..."):
            try:
                # 1. Ensure the data directory exists
                os.makedirs("data", exist_ok=True)
                
                # 2. Save the uploaded file temporarily to disk
                file_path = os.path.join("data", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                # 3. Trigger the RAG pipeline to chunk and embed
                create_vector_store(file_path)
                st.success("✅ Document indexed and ready for search!")
            except Exception as e:
                st.error("Failed to process document.")
                st.exception(e)
    
    st.divider()
    use_web_search = st.toggle(
        "Enable Live Web Search", 
        value=True,
        help="If disabled, the system will ONLY rely on uploaded internal documents (Closed-Book mode)."
    )
    
    st.divider()
    st.markdown("### Active Agents:")
    st.markdown("- 🧠 **Planner**\n- 🌐 **Researcher**\n- 📚 **RAG Retriever**\n- 📊 **Analyst**\n- 🛡️ **Critic (QA)**\n- 📝 **Report Generator**")

st.title("🤖 Agentic AI Research Analyst")
st.markdown("An enterprise-grade multi-agent reasoning system.")

if "final_report" not in st.session_state:
    st.session_state.final_report = None

query = st.text_input("Enter your research objective:", placeholder="e.g., Analyze NVIDIA's growth potential using latest AI market trends...")

if st.button("Run Analysis", type="primary"):
    if not query:
        st.warning("Please enter a research objective.")
    else:
        st.session_state.final_report = None 
        
        initial_state = {
            "query": query,
            "use_web_search": use_web_search,
            "plan": [],
            "research_data": "",
            "retrieved_docs": "",
            "analysis": "",
            "critique": "",
            "final_report": "",
            "revision_count": 0,
            "draft_approved": False
        }
        
        with st.status("Initializing Multi-Agent Workflow...", expanded=True) as status:
            try:
                st.write("🚀 **Planner Agent:** Deconstructing query...")
                
                final_state = compiled_graph.invoke(initial_state)
                
                if final_state.get("final_report"):
                    st.session_state.final_report = final_state["final_report"]
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                else:
                    status.update(label="Execution failed or yielded no report.", state="error")
                    st.error("The system failed to generate a final report.")
                    
            except Exception as e:
                status.update(label="System Error Encountered", state="error")
                st.exception(e) 

if st.session_state.final_report:
    st.subheader("📑 Final Research Report")
    
    with st.container(border=True):
        st.markdown(st.session_state.final_report)
        
    st.download_button(
        label="Download Report (Markdown)",
        data=st.session_state.final_report,
        file_name="research_report.md",
        mime="text/markdown"
    )