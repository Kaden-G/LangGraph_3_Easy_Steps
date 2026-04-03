# LangGraph Bootcamp — Exercise Files

Companion code for [LangGraph in Four Hours](./langgraph-in-four-hours.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # Required for Hours 3-4
```

## Exercises

| File | Hour | Concepts | Needs API Key? |
|------|------|----------|----------------|
| `01_first_graph.py` | 1 — The Wiring Diagram | State, nodes, edges, conditional routing | No |
| `02_research_pipeline.py` | 2 — The Traffic Controller | State reducers, loops, checkpointing | No |
| `03_multi_agent.py` | 3 — The Crew Chief | LLMs in nodes, supervisor pattern, messages | Yes |
| `04_rag_pipeline.py` | 4 — The Portfolio Piece | Multi-agent RAG with quality gates | Yes |

## Running

```bash
python exercises/01_first_graph.py
python exercises/02_research_pipeline.py
python exercises/03_multi_agent.py
python exercises/04_rag_pipeline.py
```

Hours 1–2 run without any API key (simulated logic). Hours 3–4 require an Anthropic API key.
