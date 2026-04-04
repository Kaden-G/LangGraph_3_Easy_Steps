"""
Multi-Agent RAG Pipeline with LangGraph
=========================================
A production-pattern Retrieval-Augmented Generation pipeline built with LangGraph.

Architecture:
  query_analyzer → retriever → doc_grader → [relevant?]
       ↑                                        ↓ YES → generator → hallucination_check → [grounded?] → END
       └──── refine query ←── NO ───────────────┘                        ↓ NO → retry generator ──────┘

Concepts demonstrated:
  - State reducers (Annotated[list, add]) for accumulating documents and messages
  - Conditional edges with LLM-powered routing (doc grader, hallucination checker)
  - Feedback loops (re-retrieve if docs aren't relevant, re-generate if not grounded)
  - Checkpointing for full audit trail
  - Tool-ready architecture (swap simulated retriever for real vector store)

PREREQUISITE:
  pip install langchain-anthropic langgraph
  export ANTHROPIC_API_KEY="sk-ant-..."

Portfolio framing (AI Security / AI Governance):
  - Every node transition is checkpointed → full audit trail
  - Hallucination check = output validation gate (OWASP LLM Top 10: LLM09)
  - Doc grading = input validation (ensuring retrieved context is relevant)
  - Feedback loops = self-correction without human intervention
  - Architecture maps to NIST AI RMF "Measure" function (continuous evaluation)
"""

from typing import Annotated, TypedDict
from operator import add
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ─── LLM SETUP ──────────────────────────────────────────────────────────
# Single LLM instance shared by all nodes.
# Each node gives it a different SystemMessage = different specialist persona.

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


# ─── SIMULATED DOCUMENT STORE ───────────────────────────────────────────
# In production, replace this with:
#   - FAISS + langchain (your Quokka pattern for air-gapped environments)
#   - Pinecone / Weaviate / ChromaDB for cloud
#   - Any vector store with a .similarity_search() method
#
# The retriever node just needs a function that takes a query string
# and returns a list of document strings. Swap the guts, keep the node.

DOCUMENT_STORE = [
    {
        "id": "doc-001",
        "content": "LangGraph is a framework for building stateful, multi-agent applications with LLMs. "
        "It extends LangChain with directed graph orchestration, allowing developers to define "
        "complex workflows where multiple AI agents collaborate through shared state.",
        "source": "langgraph-docs",
    },
    {
        "id": "doc-002",
        "content": "State reducers in LangGraph allow multiple nodes to accumulate data in shared state "
        "fields. Using Python's Annotated type with operator.add, list fields append rather than "
        "overwrite, enabling multi-agent collaboration without data loss.",
        "source": "langgraph-advanced-guide",
    },
    {
        "id": "doc-003",
        "content": "LangGraph checkpointing saves state snapshots at every node transition. This enables "
        "pause/resume workflows, human-in-the-loop patterns, and full audit trails. Production "
        "deployments typically use PostgresSaver for persistent checkpointing.",
        "source": "langgraph-checkpointing-guide",
    },
    {
        "id": "doc-004",
        "content": "The OWASP Top 10 for LLM Applications identifies key risks including prompt injection, "
        "insecure output handling, and training data poisoning. Mitigation strategies include "
        "input validation, output filtering, and human oversight gates.",
        "source": "owasp-llm-top10",
    },
    {
        "id": "doc-005",
        "content": "The supervisor-worker pattern in LangGraph uses one LLM agent to route tasks to "
        "specialist agents. The supervisor reads the full conversation state and decides which "
        "worker should act next, enabling dynamic multi-step workflows.",
        "source": "langgraph-patterns",
    },
    {
        "id": "doc-006",
        "content": "Retrieval-Augmented Generation (RAG) grounds LLM responses in external knowledge by "
        "retrieving relevant documents before generating answers. This reduces hallucination and "
        "enables the LLM to reference authoritative sources.",
        "source": "rag-overview",
    },
    {
        "id": "doc-007",
        "content": "The best recipe for chocolate chip cookies involves creaming butter and sugar, "
        "then folding in flour and chocolate chips. Bake at 375F for 12 minutes.",
        "source": "cooking-blog",
    },
]


def simple_retriever(query: str, top_k: int = 3) -> list[dict]:
    """
    Dead-simple keyword retriever.
    In production, replace with: vector_store.similarity_search(query, k=top_k)
    The interface stays the same: query in, list of docs out.
    """
    query_words = set(query.lower().split())
    scored = []
    for doc in DOCUMENT_STORE:
        content_words = set(doc["content"].lower().split())
        overlap = len(query_words & content_words)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ─── STATE ───────────────────────────────────────────────────────────────
# This combines everything from Hours 1-3:
#   - Reducers for accumulating documents and audit messages
#   - Overwrite fields for current query and routing decisions
#   - A retry counter to prevent infinite loops

class RAGState(TypedDict):
    # Input
    original_query: str                                # User's original question (never changes)
    refined_query: str                                 # LLM-refined search query (may update each loop)

    # Retrieved context
    documents: Annotated[list[str], add]               # REDUCER: retrieved doc contents accumulate
    doc_sources: Annotated[list[str], add]             # REDUCER: source citations accumulate

    # Quality decisions
    docs_are_relevant: bool                            # Set by doc_grader (overwrite)
    answer_is_grounded: bool                           # Set by hallucination_checker (overwrite)

    # Output
    generation: str                                    # The generated answer (overwrite)

    # Control flow
    retry_count: int                                   # Loop counter to prevent infinite retries
    audit_log: Annotated[list[str], add]               # REDUCER: every node logs what it did


# ─── NODE: QUERY ANALYZER ───────────────────────────────────────────────

def query_analyzer(state: RAGState) -> dict:
    """
    NODE: query_analyzer
    Role: Takes the user's raw question and produces an optimized search query.
    Reads: original_query
    Writes: refined_query, audit_log
    
    Why this matters: users ask vague questions ("tell me about LangGraph").
    The LLM rewrites it into something a retriever can match against
    ("LangGraph stateful multi-agent framework LLM orchestration").
    
    AI Security angle: this is an input transformation gate. You could add
    prompt injection detection here before the query hits the retriever.
    """
    query = state["original_query"]

    response = llm.invoke([
        SystemMessage(content="""You are a search query optimizer. Given a user's question,
produce a concise, keyword-rich search query that will retrieve the most relevant documents.
Respond with ONLY the optimized query, nothing else."""),
        HumanMessage(content=f"User question: {query}"),
    ])

    refined = response.content.strip()
    print(f"  [query_analyzer] '{query}' → '{refined}'")

    return {
        "refined_query": refined,
        "audit_log": [f"[query_analyzer] Refined '{query}' → '{refined}'"],
    }


# ─── NODE: RETRIEVER ────────────────────────────────────────────────────

def retriever(state: RAGState) -> dict:
    """
    NODE: retriever
    Role: Searches the document store using the refined query.
    Reads: refined_query
    Writes: documents (APPENDS via reducer), doc_sources (APPENDS), audit_log (APPENDS)
    
    This is where you'd swap in your Quokka FAISS retriever:
        docs = faiss_store.similarity_search(state["refined_query"], k=3)
        return {"documents": [d.page_content for d in docs], ...}
    
    The rest of the pipeline doesn't change.
    """
    query = state["refined_query"]
    results = simple_retriever(query, top_k=3)

    doc_contents = [doc["content"] for doc in results]
    doc_sources = [doc["source"] for doc in results]

    print(f"  [retriever] Found {len(results)} docs for '{query}'")
    for src in doc_sources:
        print(f"    - {src}")

    return {
        "documents": doc_contents,
        "doc_sources": doc_sources,
        "audit_log": [f"[retriever] Retrieved {len(results)} docs: {', '.join(doc_sources)}"],
    }


# ─── NODE: DOC GRADER ───────────────────────────────────────────────────

def doc_grader(state: RAGState) -> dict:
    """
    NODE: doc_grader
    Role: LLM evaluates whether retrieved documents are relevant to the query.
    Reads: original_query, documents
    Writes: docs_are_relevant, audit_log
    
    This is an LLM-as-judge pattern. The grader reads each document and
    decides: is this actually useful for answering the question?
    
    AI Security angle: this is a retrieval validation gate.
    - Prevents irrelevant context from polluting the generation
    - Catches retriever failures before they become hallucinations
    - Maps to NIST AI RMF "Measure" function (continuous evaluation)
    """
    query = state["original_query"]
    documents = state["documents"]

    # Ask the LLM to grade relevance
    doc_text = "\n\n---\n\n".join(documents[-3:])  # Grade the most recent batch
    response = llm.invoke([
        SystemMessage(content="""You are a document relevance grader. Given a user's question
and a set of retrieved documents, determine if the documents contain information
relevant to answering the question.

Respond with ONLY "relevant" or "not_relevant". Nothing else."""),
        HumanMessage(content=f"Question: {query}\n\nDocuments:\n{doc_text}"),
    ])

    decision = response.content.strip().lower()
    is_relevant = "relevant" in decision and "not_relevant" not in decision
    print(f"  [doc_grader] Relevance: {decision} → {'PASS' if is_relevant else 'FAIL'}")

    return {
        "docs_are_relevant": is_relevant,
        "audit_log": [f"[doc_grader] Relevance check: {'PASS' if is_relevant else 'FAIL'}"],
    }


# ─── NODE: GENERATOR ────────────────────────────────────────────────────

def generator(state: RAGState) -> dict:
    """
    NODE: generator
    Role: Synthesizes an answer from the retrieved documents.
    Reads: original_query, documents
    Writes: generation, audit_log
    
    The generator is instructed to ONLY use information from the provided
    documents. This is the core RAG constraint: ground the answer in evidence.
    
    AI Security angle: the system prompt explicitly constrains the LLM
    to provided context — a mitigation against hallucination (OWASP LLM09).
    """
    query = state["original_query"]
    documents = state["documents"]

    doc_text = "\n\n---\n\n".join(documents)
    response = llm.invoke([
        SystemMessage(content="""You are a precise research assistant. Answer the user's
question using ONLY information from the provided documents. If the documents don't
contain enough information, say so explicitly. Do not add information beyond what
the documents provide.

Cite which document supports each claim by referencing the content directly.
Keep your answer under 5 sentences."""),
        HumanMessage(content=f"Question: {query}\n\nDocuments:\n{doc_text}"),
    ])

    answer = response.content.strip()
    print(f"  [generator] Generated answer ({len(answer)} chars)")

    return {
        "generation": answer,
        "audit_log": [f"[generator] Produced answer ({len(answer)} chars)"],
    }


# ─── NODE: HALLUCINATION CHECKER ────────────────────────────────────────

def hallucination_checker(state: RAGState) -> dict:
    """
    NODE: hallucination_checker
    Role: Verifies the generated answer is grounded in the retrieved documents.
    Reads: generation, documents
    Writes: answer_is_grounded, audit_log
    
    Another LLM-as-judge node. This one compares the answer against the source
    documents and flags any claims that aren't supported by the evidence.
    
    AI Security angle: this is OUTPUT VALIDATION — the last gate before the
    answer reaches the user. Maps to:
    - OWASP LLM09 (Overreliance / Hallucination)
    - NIST AI RMF MG-2.2 (Monitor for unacceptable outputs)
    - MITRE ATLAS: defense against model manipulation
    
    In a production pipeline, a failure here could trigger:
    - Human review (human-in-the-loop via checkpointing)
    - Automatic retry with stricter constraints
    - Logging to an incident tracking system
    """
    answer = state["generation"]
    documents = state["documents"]

    doc_text = "\n\n---\n\n".join(documents)
    response = llm.invoke([
        SystemMessage(content="""You are a fact-checking auditor. Compare the generated answer
against the source documents. Determine if every claim in the answer is supported
by information in the documents.

Respond with ONLY "grounded" or "not_grounded". Nothing else."""),
        HumanMessage(content=f"Answer to check:\n{answer}\n\nSource documents:\n{doc_text}"),
    ])

    decision = response.content.strip().lower()
    is_grounded = "grounded" in decision and "not_grounded" not in decision
    print(f"  [hallucination_checker] Grounding: {decision} → {'PASS' if is_grounded else 'FAIL'}")

    return {
        "answer_is_grounded": is_grounded,
        "audit_log": [f"[hallucination_checker] Grounding check: {'PASS' if is_grounded else 'FAIL'}"],
    }


# ─── NODE: QUERY REFINER (for retry loop) ───────────────────────────────

def query_refiner(state: RAGState) -> dict:
    """
    NODE: query_refiner
    Role: When docs aren't relevant, refines the search query and bumps retry count.
    Reads: original_query, refined_query, retry_count
    Writes: refined_query (overwrite with better query), retry_count, audit_log
    """
    response = llm.invoke([
        SystemMessage(content="""The previous search query didn't return relevant results.
Generate a different search query for the same question. Try different keywords,
synonyms, or rephrasings. Respond with ONLY the new query."""),
        HumanMessage(content=f"Original question: {state['original_query']}\nFailed query: {state['refined_query']}"),
    ])

    new_query = response.content.strip()
    new_count = state.get("retry_count", 0) + 1
    print(f"  [query_refiner] Retry {new_count}: '{state['refined_query']}' → '{new_query}'")

    return {
        "refined_query": new_query,
        "retry_count": new_count,
        "audit_log": [f"[query_refiner] Retry {new_count}: refined to '{new_query}'"],
    }


# ─── ROUTERS (conditional edge functions) ────────────────────────────────

def route_after_grading(state: RAGState) -> str:
    """
    ROUTER: After doc grading, decide whether to proceed or re-retrieve.
    If docs aren't relevant AND we haven't retried too many times → refine query
    If docs are relevant OR we've exhausted retries → generate answer
    """
    if state.get("docs_are_relevant", False):
        return "generate"
    if state.get("retry_count", 0) >= 2:
        print("  [router] Max retries reached, generating with what we have")
        return "generate"
    return "refine"


def route_after_hallucination_check(state: RAGState) -> str:
    """
    ROUTER: After hallucination check, decide whether answer is good or needs retry.
    If grounded → done
    If not grounded AND haven't retried too many times → regenerate
    If not grounded AND max retries → accept what we have (log the failure)
    """
    if state.get("answer_is_grounded", False):
        return "accept"
    if state.get("retry_count", 0) >= 3:
        print("  [router] Max retries reached, accepting ungrounded answer with warning")
        return "accept"
    return "regenerate"


# ─── WIRE THE GRAPH ─────────────────────────────────────────────────────
graph = StateGraph(RAGState)

# Register all nodes
graph.add_node("query_analyzer", query_analyzer)
graph.add_node("retriever", retriever)
graph.add_node("doc_grader", doc_grader)
graph.add_node("generator", generator)
graph.add_node("hallucination_checker", hallucination_checker)
graph.add_node("query_refiner", query_refiner)

# Wire edges
graph.add_edge(START, "query_analyzer")               # Entry
graph.add_edge("query_analyzer", "retriever")          # Always retrieve after analyzing
graph.add_edge("retriever", "doc_grader")              # Always grade after retrieving

# Conditional: after grading → generate or refine?
graph.add_conditional_edges(
    "doc_grader",
    route_after_grading,
    {
        "generate": "generator",
        "refine": "query_refiner",
    },
)

graph.add_edge("query_refiner", "retriever")           # Retry loop: refine → retrieve → grade
graph.add_edge("generator", "hallucination_checker")   # Always check after generating

# Conditional: after hallucination check → accept or regenerate?
graph.add_conditional_edges(
    "hallucination_checker",
    route_after_hallucination_check,
    {
        "accept": END,
        "regenerate": "generator",
    },
)


# ─── COMPILE WITH CHECKPOINTING ─────────────────────────────────────────
memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# ─── RUN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-AGENT RAG PIPELINE")
    print("=" * 70)
    print()

    config = {"configurable": {"thread_id": "rag-session-1"}}

    result = app.invoke(
        {
            "original_query": "How does LangGraph handle state in multi-agent systems?",
            "refined_query": "",
            "documents": [],
            "doc_sources": [],
            "docs_are_relevant": False,
            "answer_is_grounded": False,
            "generation": "",
            "retry_count": 0,
            "audit_log": [],
        },
        config=config,
    )

    # ─── RESULTS ─────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(result["generation"])

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)
    for src in set(result["doc_sources"]):
        print(f"  - {src}")

    print()
    print("=" * 70)
    print("AUDIT LOG (checkpointed at every node)")
    print("=" * 70)
    for entry in result["audit_log"]:
        print(f"  {entry}")

    print()
    print("=" * 70)
    print("QUALITY GATES")
    print("=" * 70)
    print(f"  Documents relevant: {result['docs_are_relevant']}")
    print(f"  Answer grounded:    {result['answer_is_grounded']}")
    print(f"  Retry count:        {result['retry_count']}")

    # ─── BONUS: Second query to test different path ──────────────────────
    print()
    print()
    print("=" * 70)
    print("SECOND QUERY (tests retrieval of less relevant docs)")
    print("=" * 70)
    print()

    config2 = {"configurable": {"thread_id": "rag-session-2"}}

    result2 = app.invoke(
        {
            "original_query": "What are the OWASP risks for LLM applications?",
            "refined_query": "",
            "documents": [],
            "doc_sources": [],
            "docs_are_relevant": False,
            "answer_is_grounded": False,
            "generation": "",
            "retry_count": 0,
            "audit_log": [],
        },
        config=config2,
    )

    print()
    print("ANSWER:", result2["generation"])
    print()
    print("AUDIT LOG:")
    for entry in result2["audit_log"]:
        print(f"  {entry}")
