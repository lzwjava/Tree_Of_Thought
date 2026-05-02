# Tree_Of_Thought

Tree_Of_Thought is an external Tree-of-Thought reasoning system for physics-heavy problem solving.

Instead of relying on a model's hidden built-in chain-of-thought inside one opaque completion, this project turns reasoning into an explicit, inspectable, controllable tree with live state, scoring, pruning, and deterministic tool support.

It combines:

- a FastAPI service that manages long-lived reasoning sessions
- a browser UI for creating sessions, inspecting nodes, and pruning branches
- a node-level FSM and tree scheduler for controlled branch growth
- a SymPy-backed skill layer for exact symbolic and analytical computation
- multi-model routing for planning, modeling, review, and non-terminal evaluation


## Why A Tree Beats Built-In CoT

Built-in CoT is useful, but it has hard limits when you care about reliability, auditability, or branch diversity.

- Built-in CoT is hidden. You usually see only the final answer, not a persistent reasoning structure you can inspect, compare, rank, or edit.
- Built-in CoT is mostly linear. Once a single completion drifts, alternative paths are lost. A tree keeps multiple viable routes alive at the same time.
- Built-in CoT is difficult to control step-by-step. This system constrains each node to one local move instead of letting a model solve too much at once.
- Built-in CoT is hard to debug. Here every node has explicit status, score, route metadata, review output, and frontier state.
- Built-in CoT does not naturally support operator intervention. This system lets you inspect a node, delete a subtree, reconnect to a session, or continue expansion later.

The practical result is that Tree-of-Thought is not just “more reasoning.” It is reasoning with persistence, structure, and control.


## Why An External System Helps

The biggest gain comes from externalizing reasoning out of a single model pass.

- State becomes durable. Sessions, nodes, frontier entries, and run phases live outside any one completion.
- Reasoning becomes reproducible. The same scheduler settings and backend settings can be replayed and regression-tested.
- Models become swappable. Planning, modeling, review, and evaluation can each use different models with different cost and latency profiles.
- Deterministic checks become first-class. Hard rules, symbolic math, and structured post-processing do not depend on a model remembering to be careful.
- Human oversight becomes possible. The system can expose the tree over HTTP and in a browser UI instead of trapping everything inside a prompt.
- System-level optimization becomes possible. You can tune expansion budget, frontier size, reflection limits, deletion policy, and model routing independently from the prompt text.

In short, externalization turns reasoning from a hidden behavior into a real software system.


## Main Innovations In This Repository

This repository is not just a generic tree search wrapper. Its key ideas are:

- Role-split model routing. Planning, modeling, review, and non-terminal evaluation are separated rather than collapsed into one all-purpose model call.
- Route-local incremental refinement. Non-terminal nodes are expected to add exactly one new local delta instead of paraphrasing the parent or jumping ahead.
- Parent-child semantic-delta enforcement. The system explicitly checks whether a child is meaningfully different from its parent and soft-prunes unresolved duplicates.
- FSM-governed node lifecycle. Proposal, calculation, evaluation, reflection, and finalization are modeled as explicit states rather than ad hoc prompt retries.
- Lightweight intermediate evaluation plus stronger review. Non-terminal nodes can be scored cheaply while terminal or deletion-sensitive decisions still go through stronger review paths.
- Review-gated subtree deletion. Branch removal is not a blind local UI action; it flows through backend review before deletion.
- Deterministic skill integration. Symbolic computation lives in `skills.py`, so exact mechanics and mathematical checks can be attached to LLM reasoning.
- Live inspectable frontier. The scheduler exposes not only the current tree but also frontier selection and expansion state for operator debugging.


## System Overview

The current architecture uses four reasoning roles:

- planning model: route selection and orchestration
- modeling model: propose or revise one local next step
- review model: review, deletion review, and terminal evaluation
- non-terminal evaluation model: lightweight scoring for intermediate nodes

The scheduler keeps multiple branches alive, enforces route-local refinement, and exposes the evolving tree through the web UI and HTTP API.


## Recommended Default Model Stack

The current recommended preset is:

- planning: `qwen3.5-9b-mlx`
- modeling: `openai/gpt-oss-120b`
- review: `qwen/qwen3-4b-2507`
- non-terminal evaluation: `qwen2.5-0.5b-instruct-mlx`

These defaults are aligned across backend and frontend.


## Requirements

- Conda or another Python environment manager
- A local OpenAI-compatible chat backend reachable at `http://localhost:1234/api/v1/chat`
- Access to the configured planning, modeling, review, and evaluation models

The provided environment file installs the Python dependencies used by the API, scheduler, tests, and symbolic skill layer.


## Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate tot
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate tot
```


## Run The App

Start the FastAPI server:

```bash
python tot_api.py
```

Then open:

```text
http://127.0.0.1:8000/
```

The UI is served from the same process and loads the terminal-style tree explorer from `frontend/`.


## Basic Workflow

1. Start the local chat backend.
2. Start `tot_api.py`.
3. Open `http://127.0.0.1:8000/`.
4. Enter a problem statement.
5. Create a session.
6. Inspect the tree, frontier, and node details.
7. Expand the session budget or prune low-value branches as needed.

The UI also supports:

- reconnecting to an existing session id
- manual run-budget expansion
- polling and refresh controls
- node deletion through backend review
- a recommended model preset button


## API Surface

Main endpoints:

- `GET /` - serve the frontend
- `GET /health` - lightweight health check
- `POST /api/tot/sessions` - create a session
- `GET /api/tot/sessions/{session_id}` - fetch current state
- `POST /api/tot/sessions/{session_id}/run` - spend more expansion budget
- `DELETE /api/tot/sessions/{session_id}/nodes/{node_id}` - delete a subtree after review
- `DELETE /api/tot/sessions/{session_id}` - delete a session

A minimal session-creation example:

```bash
curl -X POST http://127.0.0.1:8000/api/tot/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "run_on_create": true,
    "problem_context": {
      "problem_statement": "Compare two physically plausible solution routes before refining the winning branch."
    }
  }'
```


## Repository Layout

- `tot_api.py` - FastAPI app, session store, route handlers, and frontend serving
- `fsm/` - backend routing, node FSM, models, and tree scheduler
- `frontend/` - browser UI for tree inspection and session control
- `skills.py` - SymPy-backed computation toolkit used by the reasoning system
- `skill_registry.md` - human-readable map from problem classes to skill names
- `skills.md` - skill calling conventions and usage guidance
- `tests/` - API, scheduler, FSM, and backend regression tests
- `environment.yml` - conda environment definition


## Testing

Run the API tests:

```bash
python -m unittest tests.test_api
```

Run the main harness regression suite:

```bash
python -m unittest tests.test_harness -v
```


## Operational Notes

- Session state is stored in memory, not in a database.
- Session creation returns a session id immediately; deeper expansion can continue in the background when `run_on_create` is enabled.
- If you change backend code, restart `tot_api.py` so the running server picks up the new behavior.
- Non-terminal evaluation is intentionally lighter-weight than terminal review.
- Node deletion is review-gated on the backend before a subtree is removed.


## Status

The repository is suitable for a controlled beta workflow: the API and FSM regression suites are in place, the frontend and backend defaults are aligned, and the system is intended to be run locally against a compatible model backend.