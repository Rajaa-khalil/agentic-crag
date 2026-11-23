import asyncio
from typing import List
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from crawl4ai import AsyncWebCrawler
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class DynamicWebIngestor:
    def __init__(self):
        print("⚙️ Loading Embeddings and Re-ranker models...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
        
        self.vector_store = None
        self.bm25 = None
        self.docs: List[Document] = []

    def _extract_domain_links(self, html_content: str, base_url: str, max_links: int = 3) -> List[str]:
        soup = BeautifulSoup(html_content, "html.parser")
        domain = urlparse(base_url).netloc
        discovered_links = set()

        for a_tag in soup.find_all("a", href=True):
            full_url = urljoin(base_url, a_tag["href"])
            if urlparse(full_url).netloc == domain and "#" not in full_url and full_url != base_url:
                discovered_links.add(full_url)
                if len(discovered_links) >= max_links:
                    break
        return list(discovered_links)

    async def ingest(self, target_url: str, mode: str = "shallow"):
        print(f"\n🌐 [Ingestor Mode: {mode.upper()}] Fetching: {target_url}")
        raw_documents = []

        async with AsyncWebCrawler(verbose=False) as crawler:
            main_result = await crawler.arun(url=target_url)
            if main_result.success:
                raw_documents.append(Document(
                    page_content=main_result.markdown,
                    metadata={"source": target_url}
                ))

                if mode == "deep":
                    child_urls = self._extract_domain_links(main_result.html, target_url)
                    for child_url in child_urls:
                        child_res = await crawler.arun(url=child_url)
                        if child_res.success:
                            raw_documents.append(Document(
                                page_content=child_res.markdown,
                                metadata={"source": child_url}
                            ))

        self.docs = self.text_splitter.split_documents(raw_documents)
        print(f"✂️ Created {len(self.docs)} text chunks.")

        self.vector_store = Chroma.from_documents(documents=self.docs, embedding=self.embeddings)
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ Website successfully indexed!\n")

    def retrieve_hybrid_rerank(self, query: str, top_k: int = 10, final_top_n: int = 3) -> List[Document]:
        if not self.docs:
            raise ValueError("Vector store is empty. Call `ingest()` first!")

        dense_results = self.vector_store.similarity_search(query, k=top_k)
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        sparse_results = [self.docs[i] for i in top_indices]

        doc_scores = {}
        for rank, doc in enumerate(dense_results):
            content = doc.page_content
            doc_scores[content] = {"doc": doc, "score": 1.0 / (60 + rank + 1)}
        for rank, doc in enumerate(sparse_results):
            content = doc.page_content
            doc_scores[content] = {"doc": doc, "score": doc_scores.get(content, {}).get("score", 0.0) + 1.0 / (60 + rank + 1)}

        fused_docs = [item["doc"] for item in sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]]
        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = self.reranker.predict(pairs)
        scored_docs = sorted(zip(fused_docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, score in scored_docs[:final_top_n]]