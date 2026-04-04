"""
Hour 3: Multi-Agent Patterns — "The Crew Chief"
=================================================
Three new concepts that make LangGraph production-grade:

1. LLMs INSIDE NODES — nodes stop being if/else and start being AI calls
2. MESSAGES AS STATE — the standard way to pass conversation history between agents
3. THE SUPERVISOR PATTERN — one LLM routes tasks to specialist LLMs

This builds a 3-agent system:
  - Supervisor: reads the query, decides which specialist handles it
  - Researcher: retrieves/finds information (has a search tool)
  - Writer: synthesizes findings into a polished answer

This is the exact architecture for your Hour 4 RAG pipeline.

PREREQUISITE:
  pip install langchain-anthropic
  export ANTHROPIC_API_KEY="sk-ant-..."
"""

from typing import Annotated, TypedDict, Literal
from operator import add
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ─── CONCEPT 1: THE LLM OBJECT ──────────────────────────────────────────
# This is the AI "brain" that nodes will use.
# It's created ONCE, outside the nodes, and shared.
#
# llm.invoke("string")           → quick and simple (like Hour 1 example)
# llm.invoke([messages])         → the real pattern (list of message objects)
#
# The messages pattern matters because LLMs understand ROLES:
#   SystemMessage  = "You are a ..."  (sets the agent's personality/instructions)
#   HumanMessage   = "Please do X"    (the task/query)
#   AIMessage      = "Here's my response" (previous LLM output, for context)
#
# When you send [SystemMessage, HumanMessage], the LLM sees:
#   System: You are an expert researcher.
#   Human: Find information about LangGraph.
#   → AI responds in character as a researcher

llm = ChatAnthropic(model="claude-sonnet-4-20250514")


# ─── CONCEPT 2: MESSAGE-BASED STATE ─────────────────────────────────────
# In Hour 1-2, our state had simple fields like "query" and "findings".
# Multi-agent systems typically use MESSAGES as the primary state.
#
# Why? Because agents communicate by adding messages to a shared history.
# Agent A writes "I found these docs", Agent B reads that and writes
# "Here's my analysis." The message list IS the conversation between agents.
#
# The reducer (add) means each node APPENDS messages — nothing gets lost.
# This is the same pattern as Hour 2's findings accumulator, but with
# message objects instead of plain strings.

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add]   # Conversation history (APPENDS)
    next_agent: str                                # Who should go next (supervisor sets this)
    final_answer: str                              # The finished output


# ─── CONCEPT 3: NODES WITH LLMs INSIDE ──────────────────────────────────
# Each node:
#   1. Reads relevant state
#   2. Builds a prompt (system message + context from state)
#   3. Calls llm.invoke()
#   4. Returns state updates (including the LLM's response as a new message)

def supervisor(state: AgentState) -> dict:
    """
    NODE: supervisor
    Role: The "Director" — reads the conversation and decides who goes next.
    Reads: messages (full conversation history)
    Writes: next_agent (routing decision)
    
    This is the KEY difference from Hour 1's route_by_threat:
    Instead of if/else logic, an LLM makes the routing decision.
    The LLM can reason about nuance that rules can't capture.
    
    The supervisor's prompt constrains its output to one of three choices:
    "researcher", "writer", or "FINISH". This makes parsing reliable.
    In production, you'd use structured output / tool-calling for even
    more reliable parsing (we'll see that in the researcher node below).
    """
    messages = state["messages"]

    # Build the supervisor's prompt
    # The system message defines WHO this agent is and WHAT it can do
    supervisor_prompt = [
        SystemMessage(content="""You are a supervisor coordinating a research team.
Your team has two specialists:
  - "researcher": Finds and retrieves information. Use when you need facts or data.
  - "writer": Synthesizes information into a clear answer. Use when you have enough facts.

Based on the conversation so far, decide who should act next.
If the research is complete and the writer has produced a good answer, respond with "FINISH".

Respond with ONLY one word: "researcher", "writer", or "FINISH".
No explanation, no punctuation — just the single word."""),
    ]

    # Add the full conversation history so the supervisor has context
    # This is why messages use a reducer — the supervisor sees EVERYTHING
    # that every agent has said so far.
    supervisor_prompt.extend(messages)

    # Call the LLM
    # This is it — the actual AI call. Everything else is plumbing.
    response = llm.invoke(supervisor_prompt)

    # Parse the routing decision from the LLM's response
    # response.content is a string like "researcher" or "writer" or "FINISH"
    decision = response.content.strip().lower().strip('"').strip("'")
    print(f"  [supervisor] Decision: {decision}")

    return {"next_agent": decision}


def researcher(state: AgentState) -> dict:
    """
    NODE: researcher
    Role: The specialist who finds information.
    Reads: messages (to understand what's been asked and found so far)
    Writes: messages (APPENDS its findings as a new AIMessage)
    
    In this example, the researcher is a plain LLM call.
    In Hour 4 (your RAG pipeline), this node will:
      1. Generate a search query from the conversation
      2. Call a vector store / retriever tool
      3. Return the retrieved documents as a message
    
    TOOL-CALLING PREVIEW:
    You can give agents tools like this:
    
        from langchain_core.tools import tool
        
        @tool
        def search_docs(query: str) -> str:
            '''Search the document store.'''
            results = vector_store.similarity_search(query)
            return "\\n".join([doc.page_content for doc in results])
        
        llm_with_tools = llm.bind_tools([search_docs])
        response = llm_with_tools.invoke(messages)
    
    bind_tools() tells the LLM "you can call these functions."
    The LLM decides WHEN to call them based on the conversation.
    LangGraph's prebuilt ToolNode handles executing the tool calls.
    We'll use this in Hour 4.
    """
    messages = state["messages"]

    research_prompt = [
        SystemMessage(content="""You are an expert researcher. Your job is to find
relevant information based on the conversation. Be specific and cite your reasoning.
Keep your response focused and under 3 sentences — you're feeding a writer, not
writing the final answer yourself."""),
    ]
    research_prompt.extend(messages)

    response = llm.invoke(research_prompt)
    print(f"  [researcher] Found: {response.content[:80]}...")

    # IMPORTANT: We wrap the response in a HumanMessage tagged with the agent name.
    # Why HumanMessage and not AIMessage? Because from the NEXT agent's perspective,
    # this is input (like a colleague handing them notes), not their own prior output.
    # The prefix "[Researcher]:" helps other agents know who said what.
    return {
        "messages": [HumanMessage(content=f"[Researcher]: {response.content}")],
    }


def writer(state: AgentState) -> dict:
    """
    NODE: writer
    Role: The specialist who synthesizes findings into a polished answer.
    Reads: messages (including researcher's findings)
    Writes: messages (APPENDS its draft), final_answer (the finished output)
    
    The writer sees EVERYTHING in the message history:
    - The original user query
    - All researcher findings (tagged with [Researcher]:)
    - Any previous writer drafts (if the supervisor sent it back for revision)
    
    This is the power of the message-based state with a reducer:
    the full conversation is automatically available to every node.
    """
    messages = state["messages"]

    writer_prompt = [
        SystemMessage(content="""You are an expert writer. Synthesize the research
findings in the conversation into a clear, concise answer for the user.
If you don't have enough information, say what's missing.
Keep your answer under 4 sentences."""),
    ]
    writer_prompt.extend(messages)

    response = llm.invoke(writer_prompt)
    print(f"  [writer] Wrote: {response.content[:80]}...")

    return {
        "messages": [HumanMessage(content=f"[Writer]: {response.content}")],
        "final_answer": response.content,
    }


# ─── THE ROUTER (conditional edge function) ─────────────────────────────
# This reads the supervisor's routing decision and returns the next node name.
# It's a thin wrapper — the real intelligence is in the supervisor node's LLM call.

def route_to_agent(state: AgentState) -> str:
    """
    ROUTER: Translates the supervisor's decision into a graph edge.
    
    Remember from Hour 1: routers return a STRING that matches a node name.
    The supervisor node sets next_agent via LLM; this function just reads it.
    
    "FINISH" maps to END — that's how the graph knows to stop.
    """
    next_agent = state.get("next_agent", "").lower()

    if next_agent == "finish":
        return "end"
    elif next_agent == "writer":
        return "writer"
    else:
        return "researcher"   # Default to researching if unclear


# ─── WIRE THE GRAPH ─────────────────────────────────────────────────────
# The topology:
#   START → supervisor → [researcher|writer|END]
#                ↑              |          |
#                └──────────────┘          │
#                └─────────────────────────┘
# Workers always report back to the supervisor.
# Supervisor decides: another worker, or FINISH?

graph = StateGraph(AgentState)

# Register nodes
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

# START → supervisor (supervisor always goes first)
graph.add_edge(START, "supervisor")

# Supervisor → conditional routing based on LLM decision
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "end": END,
    },
)

# Workers → back to supervisor (supervisor re-evaluates after each worker)
# This creates the LOOP: supervisor → worker → supervisor → worker → ...
# The loop exits when supervisor decides "FINISH"
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")


# ─── COMPILE AND RUN ────────────────────────────────────────────────────
memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "multi-agent-session-1"}}

print("=" * 60)
print("MULTI-AGENT RESEARCH SYSTEM")
print("=" * 60)
print()

result = app.invoke(
    {
        "messages": [
            HumanMessage(content="What is LangGraph and how does it compare to plain LangChain?")
        ],
        "next_agent": "",
        "final_answer": "",
    },
    config=config,
)

print()
print("=" * 60)
print("FINAL ANSWER")
print("=" * 60)
print(result["final_answer"])

print()
print("=" * 60)
print("FULL MESSAGE HISTORY (the agent conversation)")
print("=" * 60)
for i, msg in enumerate(result["messages"]):
    # Show who said what — this is your audit trail
    role = "USER" if isinstance(msg, HumanMessage) else "AI"
    content_preview = msg.content[:100]
    print(f"  [{i}] {role}: {content_preview}...")
