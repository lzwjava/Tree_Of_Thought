# Tree_Of_Thought

Tree_Of_Thought is a local Tree-of-Thought harness for structured reasoning over physics-heavy problems.

It combines:

- a FastAPI service that manages ToT sessions in memory
- a browser UI for creating sessions, inspecting the tree, and pruning nodes
- a node-level FSM plus tree scheduler for controlled expansion
- a SymPy-backed skill library for exact symbolic and analytical backend work

The current system is designed for local use with an OpenAI-compatible chat backend.


## What The System Does

The project splits reasoning into separate roles:

- planning model: route selection and orchestration
- modeling model: propose or revise one local next step
- review model: review, deletion review, and terminal evaluation
- non-terminal evaluation model: lightweight scoring for intermediate nodes

The scheduler keeps multiple branches alive, enforces route-local incremental refinement, and exposes the evolving tree through the web UI.


## Recommended Default Model Stack

The current recommended preset is:

- planning: `qwen3.5-9b-mlx`
- modeling: `openai/gpt-oss-120b`
- review: `qwen/qwen3-4b-2507`
- non-terminal evaluation: `qwen2.5-0.5b-instruct-mlx`

These defaults are wired into both the backend and frontend.


## Requirements

- Conda or another Python environment manager
- A local OpenAI-compatible chat backend reachable at `http://localhost:1234/api/v1/chat`
- Access to the configured planning, modeling, review, and evaluation models

The provided environment file installs the Python dependencies used by the API, scheduler, tests, and symbolic skill layer.


## Setup

From this directory, create and activate the environment:

```bash
conda env create -f environment.yml
conda activate tot
```

If the environment already exists, update it instead:

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


## Basic UI Workflow

1. Start the local chat backend at the configured base URL.
2. Start `tot_api.py`.
3. Open `http://127.0.0.1:8000/`.
4. Enter a problem statement.
5. Create a session.
6. Inspect the tree, frontier, and node details in the browser.
7. Use advanced settings if you want to override models, timeout, or payload JSON.

The UI also supports:

- reconnecting to an existing session id
- manual run-budget expansion
- polling and refresh controls
- node deletion through backend review
- a recommended model preset button


## API Surface

The main HTTP endpoints are:

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
- `skill_registry.md` - human-readable map from problem types to skill names
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

This repository is suitable for a controlled beta workflow: the core API and FSM regression suites are in place, the frontend and backend defaults are aligned, and the system is intended to be run locally against a compatible model backend.