import streamlit as st
import asyncio
import time
from agent import CorrectiveRAGAgent

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="CRAG Agent Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern dark UI
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4338CA 0%, #6D28D9 100%);
        color: #E0E7FF;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    .agent-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Agent Initialization
@st.cache_resource
def load_agent():
    return CorrectiveRAGAgent(provider="fallback")

agent = load_agent()

# 3. Sidebar Control Panel
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("Configure your autonomous research agent's target scope and strategy.")
    
    target_url = st.text_input(
        "🌐 Target Website URL",
        value="https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
        help="Provide the root web link to scrape and index into the hybrid vector store."
    )
    
    query_type = st.radio(
        "🎯 Execution Strategy:",
        options=["shallow", "deep", "problem_solving"],
        format_func=lambda x: {
            "shallow": "⚡ Shallow (1-Page Fast Lookup)",
            "deep": "🔗 Deep Crawl (Multi-Hop Web Search)",
            "problem_solving": "🧩 Problem Solving (Deconstruct & Auto-Fix)"
        }[x],
        help="Shallow answers directly from 1 page. Deep crawls child links. Problem Solving searches developer hubs for fixes."
    )
    
    st.divider()
    st.markdown("**Engine Architecture:**")
    st.caption("• Framework: **LangGraph Agent**")
    st.caption("• LLM: **Ollama (Qwen2.5 Local)**")
    st.caption("• Vector DB: **ChromaDB + BM25**")
    st.caption("• Search Tool: **DuckDuckGo API**")

# 4. Main UI Dashboard
st.markdown('<p class="agent-header">⚡ Autonomous Corrective RAG Studio</p>', unsafe_allow_html=True)
st.markdown("Autonomous web scraper, dynamic evaluator, and multi-hop reasoning agent.")

# Top System Metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metric-card"><p style="color:#94A3B8;margin:0;">Target Engine</p><h3 style="color:#38BDF8;margin:0;">Ollama Local</h3></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><p style="color:#94A3B8;margin:0;">Selected Strategy</p><h3 style="color:#818CF8;margin:0;">{query_type.upper()}</h3></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><p style="color:#94A3B8;margin:0;">Token Cost</p><h3 style="color:#34D399;margin:0;">$0.00 (Free)</h3></div>', unsafe_allow_html=True)

st.write("")

# Query Input
user_question = st.text_area(
    "💬 Enter Prompt / Complex Error Stack:",
    height=110,
    placeholder="e.g., What primary advantage does Retrieval-Augmented Generation offer over standard static LLMs?"
)

# 5. Execution Workflow
if st.button("🚀 Run Agentic Pipeline", type="primary"):
    if not target_url or not user_question:
        st.error("Please enter both a valid URL and a query to proceed.")
    else:
        start_time = time.time()
        
        # Execution Live Monitor
        status_box = st.status("🧠 Agent Actively Executing LangGraph Pipeline...", expanded=True)
        
        async def execute_agent():
            status_box.write("🌐 **Node 1: Web Ingestor** -> Crawling site & creating hybrid vector index...")
            await asyncio.sleep(0.5)
            
            status_box.write("🧐 **Node 2: Evaluator & Analyzer** -> Checking context relevancy & decomposing query...")
            await asyncio.sleep(0.5)
            
            status_box.write("🌐 **Node 3: Retrieval Router** -> Querying vector memory & fallback search...")
            
            # Run agent core loop
            response = await agent.run(target_url, user_question, query_type=query_type)
            
            status_box.write("🧠 **Node 4: Synthesizer** -> Formulating structured output...")
            return response

        try:
            final_output = asyncio.run(execute_agent())
            elapsed_time = round(time.time() - start_time, 2)
            
            status_box.update(label=f"✅ Pipeline Execution Completed in {elapsed_time}s!", state="complete", expanded=False)
            
            # Display Results Container
            st.divider()
            st.markdown("### 🎯 Final Agent Output")
            
            # Card Display
            st.info(final_output)
            
            # System Logs Expander
            with st.expander("🔍 View Execution Diagnostic Logs"):
                st.code(
                    f"URL: {target_url}\n"
                    f"Query Type: {query_type}\n"
                    f"Execution Time: {elapsed_time} seconds\n"
                    f"Status: SUCCESS (200)\n"
                    f"Active Model: Ollama qwen2.5:3b (Local)",
                    language="yaml"
                )

        except Exception as e:
            status_box.update(label="❌ Agent Execution Error", state="error", expanded=True)
            st.error(f"Details: {str(e)}")