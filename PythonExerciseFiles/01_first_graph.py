"""
Step 1: Your First LangGraph Graph
====================================
Mental model:
  - A LangGraph graph is like a military OPORD turned into code.
  - STATE  = the Common Operational Picture (shared data every node can read/write)
  - NODE   = a staff section (one function that does one job)
  - EDGE   = a fixed arrow ("after Intel, always go to COA Selection")
  - CONDITIONAL EDGE = a decision diamond ("if threat HIGH → engage, else → monitor")
  - START/END = built-in entry and exit points for the graph

The flow:
  START → analyze_intel → [decision: HIGH/MEDIUM/LOW?] → respond_* → END
"""

from typing import TypedDict
from datetime import datetime
from langgraph.graph import StateGraph, START, END


# ─── STEP 1: DEFINE THE STATE ───────────────────────────────────────────
# This is your "COP" — the shared data structure.
# Every node receives the full state as input.
# Every node returns a dict of ONLY the fields it wants to update.
# Fields it doesn't return stay unchanged.
#
# TypedDict gives you type hints so your IDE can autocomplete state["field"].
# In Step 2, we'll add "reducers" that let fields ACCUMULATE instead of overwrite.

class IntelState(TypedDict):
    report: str          # Input: the raw intelligence report text
    threat_level: str    # Set by analyze_intel: "HIGH", "MEDIUM", or "LOW"
    action: str          # Set by respond_* nodes: what we're doing about it
    timestamp: str       # Set by analyze_intel: when the analysis happened


# ─── STEP 2: DEFINE THE NODES ───────────────────────────────────────────
# Each node is just a plain Python function.
#
# Rules:
#   - Takes ONE argument: the full state (typed as your TypedDict)
#   - Returns a dict with ONLY the fields to update (not the full state)
#   - Should NOT mutate the input state — return new values instead
#   - Should NOT know about other nodes — the graph handles routing
#
# Why this matters:
#   - You can swap any node independently (e.g., replace keyword matching with an LLM)
#   - You can test each node in isolation (just pass it a dict)
#   - You can reuse nodes across different graphs

def analyze_intel(state: IntelState) -> dict:
    """
    NODE: analyze_intel
    Role: S-2 Intelligence — assess the threat level from a raw report.
    Reads: report
    Writes: threat_level, timestamp
    
    Right now this is simple keyword matching.
    In a real system, you'd replace the if/elif/else with an LLM call like:
        response = llm.invoke(f"Classify threat level for: {report}")
    The rest of the graph wouldn't change at all — that's the power of nodes.
    """
    report = state["report"].lower()
    now = datetime.now().isoformat()

    # Classify threat based on keywords
    if "enemy" in report or "hostile" in report:
        threat = "HIGH"
    elif "suspicious" in report or "unusual" in report:
        threat = "MEDIUM"
    else:
        threat = "LOW"

    # Return ONLY the fields we're updating.
    # "report" stays unchanged because we don't include it here.
    return {"threat_level": threat, "timestamp": now}


def respond_high(state: IntelState) -> dict:
    """
    NODE: respond_high
    Role: Execute the HIGH threat response.
    Reads: report (for context in the action message)
    Writes: action
    
    This node only runs if the router sends us here.
    It has no idea respond_medium or respond_low exist.
    """
    return {"action": f"ENGAGE: Mobilizing QRF for report: {state['report']}"}


def respond_medium(state: IntelState) -> dict:
    """
    NODE: respond_medium
    Role: Execute the MEDIUM threat response.
    Reads: report
    Writes: action
    """
    return {"action": f"INVESTIGATE: Dispatching recon element. Report: {state['report']}"}


def respond_low(state: IntelState) -> dict:
    """
    NODE: respond_low
    Role: Execute the LOW threat response.
    Reads: report
    Writes: action
    """
    return {"action": f"MONITOR: Continuing patrol. Report: {state['report']}"}


# ─── STEP 3: DEFINE THE ROUTER ──────────────────────────────────────────
# A router is a function used by a CONDITIONAL EDGE.
# It reads the current state and returns a STRING — the name of the next node.
#
# Critical: the returned string MUST exactly match a node name you registered
# with add_node(). If it returns "respond_high", there must be a node called
# "respond_high". LangGraph won't warn you about typos — it'll just error
# at runtime.
#
# Also critical: every possible return value must be reachable.
# If your if/elif/else can never reach a branch, that node is orphaned —
# wired into the graph but never executed. (This was your Step 1 bug.)

def route_by_threat(state: IntelState) -> str:
    """
    ROUTER: Decides which respond_* node to run next.
    This function IS the "decision diamond" in the flowchart.
    """
    if state["threat_level"] == "HIGH":
        return "respond_high"       # ← Must match add_node("respond_high", ...)
    elif state["threat_level"] == "MEDIUM":
        return "respond_medium"     # ← Must match add_node("respond_medium", ...)
    return "respond_low"            # ← Default / fallback path


# ─── STEP 4: WIRE THE GRAPH ─────────────────────────────────────────────
# This is where you assemble the OPORD.
# StateGraph takes your state class so it knows the schema.

graph = StateGraph(IntelState)

# Register nodes — order doesn't matter here, just like org chart slots.
# The string name is how edges reference this node.
# The function is what runs when the graph reaches this node.
graph.add_node("analyze_intel", analyze_intel)
graph.add_node("respond_high", respond_high)
graph.add_node("respond_medium", respond_medium)
graph.add_node("respond_low", respond_low)

# FIXED EDGE: START → analyze_intel
# "No matter what, the first thing we do is analyze the intel."
graph.add_edge(START, "analyze_intel")

# CONDITIONAL EDGE: analyze_intel → ??? (depends on router output)
# Arguments:
#   1. Source node name: "analyze_intel"
#   2. Router function: route_by_threat
#   3. Path map: {router_return_value: target_node_name}
#      This map is technically optional (LangGraph can infer it),
#      but being explicit is best practice — it documents your graph.
graph.add_conditional_edges(
    "analyze_intel",
    route_by_threat,
    {
        "respond_high": "respond_high",
        "respond_medium": "respond_medium",
        "respond_low": "respond_low",
    },
)

# FIXED EDGES: all response nodes → END
# "After any response, the mission is complete."
graph.add_edge("respond_high", END)
graph.add_edge("respond_medium", END)
graph.add_edge("respond_low", END)


# ─── STEP 5: COMPILE AND RUN ────────────────────────────────────────────
# .compile() validates the graph (checks for missing edges, unreachable nodes)
# and returns a runnable object.
# After compiling, you can't add more nodes or edges.
# The compiled app behaves like a function: input state in, output state out.

app = graph.compile()

# .invoke() runs the graph synchronously.
# You pass in the INITIAL state — every field should have a value.
# The graph returns the FINAL state after all nodes have executed.

print("=" * 60)
print("TEST 1: HIGH threat")
print("=" * 60)
result1 = app.invoke({
    "report": "Enemy convoy spotted on MSR Tampa",
    "threat_level": "",     # Empty — analyze_intel will fill this
    "action": "",           # Empty — respond_* will fill this
    "timestamp": "",        # Empty — analyze_intel will fill this
})
print(f"  Threat:    {result1['threat_level']}")
print(f"  Action:    {result1['action']}")
print(f"  Timestamp: {result1['timestamp']}")

print()
print("=" * 60)
print("TEST 2: MEDIUM threat")
print("=" * 60)
result2 = app.invoke({
    "report": "Suspicious activity near the bridge",
    "threat_level": "",
    "action": "",
    "timestamp": "",
})
print(f"  Threat:    {result2['threat_level']}")
print(f"  Action:    {result2['action']}")
print(f"  Timestamp: {result2['timestamp']}")

print()
print("=" * 60)
print("TEST 3: LOW threat")
print("=" * 60)
result3 = app.invoke({
    "report": "Routine traffic on Highway 1",
    "threat_level": "",
    "action": "",
    "timestamp": "",
})
print(f"  Threat:    {result3['threat_level']}")
print(f"  Action:    {result3['action']}")
print(f"  Timestamp: {result3['timestamp']}")
