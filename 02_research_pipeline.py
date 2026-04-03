"""
Hour 2: State Reducers, Conditional Loops, and Checkpointing
=============================================================
This builds on Hour 1 with three new concepts:

1. STATE REDUCERS — fields that ACCUMULATE instead of overwrite
   Without a reducer:  Node returns {"findings": ["C"]}  → state becomes ["C"]      (A and B are gone)
   With a reducer:     Node returns {"findings": ["C"]}  → state becomes ["A","B","C"] (appended!)
   
   The syntax: Annotated[list[str], add]
   The 'add' is operator.add, which for lists means concatenation.

2. CONDITIONAL LOOPS — a graph that can cycle back to an earlier node
   Unlike a Python while loop, LangGraph manages the execution:
   - Checkpoints state at every node boundary
   - Can be interrupted and resumed mid-loop
   - Can be inspected at any point in the loop

3. CHECKPOINTING — automatic state snapshots at every node transition
   MemorySaver = in-memory (dev/testing)
   SqliteSaver = local file (small apps)
   PostgresSaver = production (multi-user, persistent)

The flow:
  START → search_web → analyze_findings → [enough?] → YES → summarize → END
                 ↑                            |
                 └── increment_loop ←── NO ───┘
"""

from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ─── STEP 1: STATE WITH REDUCERS ────────────────────────────────────────
# Compare this to Hour 1's IntelState where every field just overwrites.
#
# The magic is: Annotated[list[str], add]
#   - list[str]    = the type (a list of strings)
#   - add          = the REDUCER function (operator.add, which concatenates lists)
#
# When a node returns {"findings": ["new item"]}, LangGraph does:
#   state["findings"] = state["findings"] + ["new item"]   ← concatenation, not replacement
#
# Fields WITHOUT a reducer (like loop_count, summary) still overwrite normally.
# You choose per-field whether to accumulate or overwrite.
#
# Other useful reducers:
#   - operator.add  → concatenate lists, add numbers
#   - Custom function → def my_reducer(existing, new): return ...
#     e.g., keep only unique items, cap at N items, merge dicts, etc.

class ResearchState(TypedDict):
    query: str                                    # Input query (no reducer → overwrite)
    findings: Annotated[list[str], add]           # REDUCER: each node's findings get APPENDED
    sources: Annotated[list[str], add]            # REDUCER: each node's sources get APPENDED
    loop_count: int                                # No reducer → overwrite (we set it explicitly)
    summary: str                                   # No reducer → overwrite (final output)


# ─── STEP 2: NODES ──────────────────────────────────────────────────────

def search_web(state: ResearchState) -> dict:
    """
    NODE: search_web
    Role: Simulates searching the web for information about the query.
    Reads: query, loop_count
    Writes: findings (APPENDS via reducer), sources (APPENDS via reducer)
    
    Key insight: this node returns {"findings": [...], "sources": [...]}.
    Because both fields have reducers, these get APPENDED to existing state.
    First loop: state goes from [] to ["fact1", "fact2"]
    Second loop: state goes from ["fact1", "fact2"] to ["fact1", "fact2", "fact3", "fact4"]
    Nothing is lost!
    
    In a real system, you'd call a search API here:
        results = tavily_client.search(query)
        return {"findings": [r.content for r in results], "sources": [r.url for r in results]}
    """
    query = state["query"]
    loop = state.get("loop_count", 0)
    print(f"  [search_web] Loop {loop}: Searching for '{query}'")

    # Simulated results — each loop "discovers" different information
    if loop == 0:
        return {
            "findings": [
                "LangGraph is a library for building stateful multi-agent apps",
                "It uses directed graphs where nodes are functions",
            ],
            "sources": ["langgraph-docs", "blog-post-1"],
        }
    else:
        return {
            "findings": [
                "LangGraph supports checkpointing for pause/resume workflows",
                "State reducers allow accumulation across nodes",
            ],
            "sources": ["langgraph-advanced-docs", "github-examples"],
        }


def analyze_findings(state: ResearchState) -> dict:
    """
    NODE: analyze_findings
    Role: Reviews accumulated findings and adds its own analysis.
    Reads: findings (the FULL accumulated list so far)
    Writes: findings (APPENDS an analysis note via reducer)
    
    Notice: we return a LIST with one item, not a bare string.
    The reducer uses operator.add (list concatenation), so it expects list + list.
    Returning a bare string would cause: ["existing"] + "string" → TypeError
    Always return the same type the reducer expects.
    """
    num_findings = len(state["findings"])
    print(f"  [analyze] Reviewing {num_findings} findings so far")

    return {
        "findings": [f"Analysis: {num_findings} facts gathered, assessing completeness"],
        # ↑ Must be a list, not a string — the reducer does list + list
    }


def quality_gate(state: ResearchState) -> str:
    """
    ROUTER: The conditional edge that enables LOOPING.
    
    This is conceptually the same as route_by_threat from Hour 1,
    but now one of the return values points BACKWARD in the graph
    (to increment_loop, which connects back to search_web).
    
    That backward edge creates a CYCLE — the graph can loop.
    LangGraph handles this natively. A raw Python while loop can't:
      - Checkpoint between iterations
      - Be paused/resumed by a human
      - Be inspected at any iteration boundary
    
    The guard conditions (>= 5 findings OR >= 2 loops) prevent infinite loops.
    In production, you'd also set a recursion_limit on the compiled graph.
    """
    num_findings = len(state["findings"])
    loop_count = state.get("loop_count", 0)
    print(f"  [quality_gate] Findings: {num_findings}, Loops: {loop_count}")

    # Enough info? → finish up
    if num_findings >= 5 or loop_count >= 2:
        return "summarize"

    # Not enough? → go around again
    return "loop_back"


def increment_loop(state: ResearchState) -> dict:
    """
    NODE: increment_loop
    Role: Bumps the loop counter before routing back to search_web.
    Reads: loop_count
    Writes: loop_count (OVERWRITES — no reducer on this field)
    
    This is a "utility node" — it exists purely to manage control flow state.
    You'll see this pattern often: small nodes that don't do real work,
    just update counters, flags, or routing state.
    
    Why not just increment in search_web? Separation of concerns.
    search_web's job is searching. Loop management is a separate concern.
    """
    current = state.get("loop_count", 0)
    print(f"  [increment_loop] Bumping loop count from {current} to {current + 1}")
    return {"loop_count": current + 1}


def summarize(state: ResearchState) -> dict:
    """
    NODE: summarize
    Role: Produce a final summary from ALL accumulated findings.
    Reads: findings (the FULL list from every loop), sources
    Writes: summary
    
    This is the "payoff node" — it reads everything the pipeline has gathered.
    Because findings used a reducer, we have the complete picture here.
    Without a reducer, we'd only have the LAST node's findings.
    """
    all_findings = state["findings"]
    all_sources = state["sources"]
    summary = (
        f"Research complete.\n"
        f"  Total findings: {len(all_findings)}\n"
        f"  Sources consulted: {', '.join(all_sources)}\n"
        f"  Key facts:\n"
    )
    for i, finding in enumerate(all_findings, 1):
        summary += f"    {i}. {finding}\n"
    return {"summary": summary}


# ─── STEP 3: WIRE THE GRAPH ─────────────────────────────────────────────
graph = StateGraph(ResearchState)

# Register all nodes
graph.add_node("search_web", search_web)
graph.add_node("analyze_findings", analyze_findings)
graph.add_node("increment_loop", increment_loop)
graph.add_node("summarize", summarize)

# Fixed edges: the "always go here next" connections
graph.add_edge(START, "search_web")                 # Entry point
graph.add_edge("search_web", "analyze_findings")    # Always analyze after searching
graph.add_edge("increment_loop", "search_web")      # ← THE LOOP: after incrementing, search again
graph.add_edge("summarize", END)                     # Done

# Conditional edge: the quality gate decides whether to loop or finish
# "loop_back" maps to increment_loop (which then goes back to search_web)
# "summarize" maps to summarize (which then goes to END)
graph.add_conditional_edges(
    "analyze_findings",       # After this node runs...
    quality_gate,             # ...run this router function...
    {
        "summarize": "summarize",       # If router returns "summarize" → go to summarize node
        "loop_back": "increment_loop",  # If router returns "loop_back" → go to increment_loop node
    },
)


# ─── STEP 4: COMPILE WITH CHECKPOINTING ─────────────────────────────────
# Checkpointing = automatic state snapshots at every node boundary.
#
# MemorySaver stores snapshots in a Python dict (lost when process exits).
# For production, use:
#   from langgraph.checkpoint.sqlite import SqliteSaver   → local file
#   from langgraph.checkpoint.postgres import PostgresSaver → database
#
# When you compile with a checkpointer, you MUST pass a thread_id in config.
# Think of thread_id as a session/conversation ID:
#   - Same thread_id = same state history (can resume)
#   - Different thread_id = fresh state (new session)
#
# Why this matters for AI Security:
#   - Audit trail: inspect what the LLM decided at each step
#   - Human-in-the-loop: pause before an action, wait for approval, resume
#   - Replay: re-run from any checkpoint with different inputs

memory = MemorySaver()
app = graph.compile(checkpointer=memory)


# ─── STEP 5: RUN IT ─────────────────────────────────────────────────────
print("=" * 60)
print("RUNNING RESEARCH PIPELINE")
print("=" * 60)

# Config with thread_id — required when using a checkpointer
config = {"configurable": {"thread_id": "research-session-1"}}

result = app.invoke(
    {
        "query": "What is LangGraph and why use it?",
        "findings": [],       # Start empty — nodes will accumulate via reducer
        "sources": [],        # Start empty — nodes will accumulate via reducer
        "loop_count": 0,      # Start at 0 — increment_loop will bump this
        "summary": "",        # Start empty — summarize will fill this
    },
    config=config,
)

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(result["summary"])


# ─── STEP 6: INSPECT CHECKPOINTED STATE ─────────────────────────────────
# After the graph finishes, you can peek at the saved state.
# This is the same data that would let you resume a paused graph.
# In an AI Security context, this is your audit log.

print("=" * 60)
print("CHECKPOINT INSPECTION (your audit trail)")
print("=" * 60)
state_snapshot = app.get_state(config)
print(f"  Total findings accumulated: {len(state_snapshot.values['findings'])}")
print(f"  Sources: {state_snapshot.values['sources']}")
print(f"  Loop count: {state_snapshot.values['loop_count']}")

# You can also get the full history of state transitions:
# for snapshot in app.get_state_history(config):
#     print(f"  Step: {snapshot.metadata.get('step', '?')} → {snapshot.values.get('loop_count', 0)} loops")
