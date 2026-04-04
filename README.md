# LangGraph in Four Hours

**A practitioner's guide to building stateful multi-agent systems.**

For engineers who know Python and have used LLMs but haven't orchestrated them. Four hours from zero to a production-pattern multi-agent RAG pipeline with quality gates, feedback loops, and a full audit trail.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."   # Required for Hours 3-4
```

## Interactive Notebooks (Recommended)

The Jupyter notebooks are the best way to work through this material — each one interleaves explanations with executable code cells.

| Notebook | Hour | Concepts | Needs API Key? |
|----------|------|----------|----------------|
| [`01_first_graph.ipynb`](./01_first_graph.ipynb) | 1 — The Wiring Diagram | State, nodes, edges, conditional routing | No |
| [`02_research_pipeline.ipynb`](./02_research_pipeline.ipynb) | 2 — The Traffic Controller | State reducers, loops, checkpointing | No |
| [`03_multi_agent.ipynb`](./03_multi_agent.ipynb) | 3 — The Crew Chief | LLMs in nodes, supervisor pattern, messages | Yes |
| [`04_rag_pipeline.ipynb`](./04_rag_pipeline.ipynb) | 4 — The Portfolio Piece | Multi-agent RAG with quality gates | Yes |

```bash
pip install jupyter
jupyter notebook
```

## Python Scripts

The same exercises as standalone scripts, runnable from the command line.

| Script | Hour | Concepts | Needs API Key? |
|--------|------|----------|----------------|
| [`01_first_graph.py`](./01_first_graph.py) | 1 — The Wiring Diagram | State, nodes, edges, conditional routing | No |
| [`02_research_pipeline.py`](./02_research_pipeline.py) | 2 — The Traffic Controller | State reducers, loops, checkpointing | No |
| [`03_multi_agent.py`](./03_multi_agent.py) | 3 — The Crew Chief | LLMs in nodes, supervisor pattern, messages | Yes |
| [`04_rag_pipeline.py`](./04_rag_pipeline.py) | 4 — The Portfolio Piece | Multi-agent RAG with quality gates | Yes |

```bash
python 01_first_graph.py
python 02_research_pipeline.py
python 03_multi_agent.py
python 04_rag_pipeline.py
```

## Interactive Sequence Diagram

[`sequence_diagram.html`](./sequence_diagram.html) — An interactive step-through visualization of the multi-agent supervisor execution flow from Hour 3. Open it in any browser to click through all 5 phases of how the graph runtime, supervisor, LLM, and workers interact at runtime.

## Guide

[`LangGraph_3_Easy_Steps.md`](./LangGraph_3_Easy_Steps.md) — The full written guide covering mental models, vocabulary, and architecture patterns across all four hours.

## Hours 1-2 vs 3-4

Hours 1-2 use simulated logic (keyword matching, threshold checks) and run without any API key. This is deliberate: learn the graph mechanics before adding LLM complexity.

Hours 3-4 require an Anthropic API key (`ANTHROPIC_API_KEY` environment variable). These hours put LLMs inside nodes and build production-pattern multi-agent systems.
