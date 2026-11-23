import asyncio
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from llm_factory import get_llm
from web_ingestor import DynamicWebIngestor

# 1. Define Expanded Graph State
class GraphState(TypedDict):
    question: str
    target_url: str
    query_type: str  # 'shallow', 'deep', or 'problem_solving'
    documents: List[Document]
    sub_queries: List[str]
    generation: str
    loop_count: int

class CorrectiveRAGAgent:
    def __init__(self, provider: str = "fallback"):
        print("⚡ Initializing Adaptive Query-Type LangGraph Agent...")
        self.llm = get_llm(provider=provider)
        self.ingestor = DynamicWebIngestor()
        self.search_tool = DuckDuckGoSearchRun()
        self.workflow = self._build_graph()

    # --- NODE 1: Ingest Base URL ---
    async def ingest_node(self, state: GraphState) -> GraphState:
        url = state["target_url"]
        query_type = state.get("query_type", "shallow")
        
        # Deep mode crawls sub-pages, shallow only grabs main page
        mode = "deep" if query_type in ["deep", "problem_solving"] else "shallow"
        
        print(f"\n🌐 [Node: Web Ingestor] Processing target URL in [{mode.upper()}] mode...")
        if not self.ingestor.docs:
            await self.ingestor.ingest(url, mode=mode)
            
        return state

    # --- NODE 2: Query Analyzer & Problem Decomposer ---
    def analyze_query_node(self, state: GraphState) -> GraphState:
        query_type = state["query_type"]
        question = state["question"]
        print(f"\n⚙️ [Node: Query Analyzer] Strategy configured for: [{query_type.upper()}]")

        if query_type == "problem_solving":
            prompt = (
                f"You are a technical problem solver. Break down the following complex issue into 2 target search queries "
                f"to find solutions on developer platforms (e.g., GitHub, StackOverflow, Docs).\n"
                f"Problem: {question}\n"
                f"Output ONLY the 2 search queries separated by a newline, no numbering."
            )
            res = self.llm.invoke(prompt).content.strip().split("\n")
            sub_queries = [q.strip() for q in res if q.strip()][:2]
            print(f"🧩 Problem decomposed into sub-queries: {sub_queries}")
            return {"sub_queries": sub_queries, "loop_count": 0}
        
        return {"sub_queries": [question], "loop_count": 0}

    # --- NODE 3: Hybrid Retrieval & Platform Search ---
    def retrieve_node(self, state: GraphState) -> GraphState:
        print("\n📥 [Node: Retrieve] Searching indexed content & external developer platforms...")
        question = state["question"]
        query_type = state["query_type"]
        sub_queries = state.get("sub_queries", [question])
        
        retrieved_docs = []
        
        # 1. Retrieve from target site index
        if self.ingestor.docs:
            local_docs = self.ingestor.retrieve_hybrid_rerank(question, final_top_n=3)
            retrieved_docs.extend(local_docs)

        # 2. If problem_solving or deep, search external platforms autonomously
        if query_type in ["deep", "problem_solving"]:
            for sq in sub_queries:
                print(f"🔍 Searching online platforms for: '{sq}'")
                try:
                    search_res = self.search_tool.run(f"{sq} solution OR fix")
                    retrieved_docs.append(Document(
                        page_content=f"Online Platform Search [{sq}]:\n{search_res}",
                        metadata={"source": "external_platforms"}
                    ))
                except Exception as e:
                    print(f"⚠️ Search failed for '{sq}': {e}")

        return {"documents": retrieved_docs}

    # --- NODE 4: Solution Generator ---
    def generate_node(self, state: GraphState) -> GraphState:
        print("\n🧠 [Node: Synthesize] Generating structured solution...")
        question = state["question"]
        query_type = state["query_type"]
        docs = state["documents"]
        
        context_text = "\n\n".join([f"Source [{d.metadata.get('source', 'web')}]: {d.page_content}" for d in docs])
        
        if query_type == "problem_solving":
            prompt = (
                f"You are an expert AI problem solver. Resolve the following issue step-by-step using the context provided.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Problem to Solve: {question}\n\n"
                f"Provide a clear response with:\n"
                f"1. Root Cause / Core Explanation\n"
                f"2. Step-by-Step Fix / Code Solution\n"
                f"3. Key Considerations"
            )
        else:
            prompt = (
                f"Answer the user query accurately using the context.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Query: {question}\n"
                f"Answer:"
            )
            
        response = self.llm.invoke(prompt).content
        return {"generation": response}

    # --- GRAPH ASSEMBLY ---
    def _build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node("ingest", self.ingest_node)
        builder.add_node("analyze", self.analyze_query_node)
        builder.add_node("retrieve", self.retrieve_node)
        builder.add_node("generate", self.generate_node)

        builder.set_entry_point("ingest")
        builder.add_edge("ingest", "analyze")
        builder.add_edge("analyze", "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    async def run(self, target_url: str, question: str, query_type: str = "shallow"):
        initial_state = {
            "question": question,
            "target_url": target_url,
            "query_type": query_type,
            "documents": [],
            "sub_queries": [],
            "generation": "",
            "loop_count": 0
        }
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["generation"]