"""Backend request schemas and adapter contracts for the ToT harness."""

from __future__ import annotations

from copy import deepcopy
from difflib import get_close_matches
from importlib import import_module
import json
import socket
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from .models import (
    EvaluationPayload,
    MetaAnalysisPayload,
    NodeSnapshot,
    OrchestratorTaskPayload,
    ProposalPayload,
    ReflectionPayload,
)
from .utils import _model_field_names

DEFAULT_CHAT_API_URL = "http://localhost:1234/api/v1/chat"
DEFAULT_PLANNING_MODEL = "qwen3.5-9b-mlx"
DEFAULT_MODELING_MODEL = "openai/gpt-oss-120b"
DEFAULT_REVIEW_MODEL = "qwen/qwen3-4b-2507"
DEFAULT_NON_TERMINAL_EVALUATION_MODEL = "qwen2.5-0.5b-instruct-mlx"
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
STAGE_MODEL_FALLBACKS = {
    "meta-analysis": ["qwen/qwen3.5-9b", DEFAULT_REVIEW_MODEL],
    "orchestrator": ["qwen/qwen3.5-9b", DEFAULT_REVIEW_MODEL],
    "proposal": [DEFAULT_REVIEW_MODEL, "qwen/qwen3-1.7b", "qwen/qwen3-1.7b@4bit"],
    "reflection": [DEFAULT_REVIEW_MODEL, "qwen/qwen3-1.7b", "qwen/qwen3-1.7b@4bit"],
    "evaluation": [DEFAULT_NON_TERMINAL_EVALUATION_MODEL, "qwen/qwen3-1.7b", "qwen/qwen3-1.7b@4bit"],
    "delete-review": ["qwen/qwen3-1.7b", "qwen/qwen3-1.7b@4bit", "qwen2.5-0.5b-instruct-mlx"],
}

ChatRequester = Callable[[str, dict[str, Any], float], Any]


def _fallback_stage_prompt_contract(stage: str) -> dict[str, Any]:
    normalized_stage = str(stage).strip().lower()
    fallback_contracts = {
        "meta-analysis": {
            "optional_keys": ["route_options", "step_blueprints"],
            "prompt_fragment": (
                "You are the ToT planning model. Analyze the problem once at session creation time and return only a JSON object "
                "with keys objective, givens, unknowns, minimal_subproblems, step_ordering, first_step, completion_signals. "
                "Keep the plan coarse: make the first checkpoint a route-splitting step that preserves many modeling routes across dimensions such as force balance, energy, momentum, kinematics, geometry, symmetry, limiting cases, dimensional analysis, boundary conditions, approximations, and equivalent formulations. Within each plausible route, also contrast different correction quantities or closure choices when they change the modeling path. Keep each route option and step blueprint short and atomic: each one should represent the simplest route-local first move, such as naming one governing law/model, one decisive assumption, or one active correction quantity or closure. Later per-step orchestration will strictly decompose each checkpoint into one executable micro task. "
                "Do not solve the full problem. Do not use markdown."
            )
        },
        "orchestrator": {
            "optional_keys": ["selected_route_family", "candidate_tasks"],
            "prompt_fragment": (
                "You are the ToT orchestrator. Return only a JSON object with keys step_focus, current_step_guidance, task_breakdown, selected_task, deferred_tasks, completion_signals. "
                "You do not receive the full problem statement; operate only on the local checkpoint metadata already provided in the request. "
                "Read the current node state, parent state, review feedback, and meta-task progress, then strictly decompose the active step into the smallest executable micro tasks, choose exactly one selected_task for the modeling model to execute now, and defer everything else. During strategy_scan, the selected_task must isolate one route family only and do exactly one thing: name one governing law/model, state one decisive assumption, or choose one active correction quantity or closure. "
                "When several route families or correction modes remain plausible, also include optional selected_route_family and candidate_tasks objects so downstream reasoning can stay distributed without losing structure; candidate_tasks should preserve route_family, correction_mode, and correction_target whenever they matter. Do not derive equations or solve the task yourself. Do not use markdown."
            )
        },
        "proposal": {
            "prompt_fragment": (
                "You are the ToT modeling model. Return only a JSON object with keys thought_step, equations, known_vars, used_models, quantities, boundary_conditions. "
                "Produce exactly one minimal next-step candidate. Do not solve the whole problem or emit multiple alternatives. If the current phase is strategy_scan, stay route-local and atomic: do not compare many routes inside one node, and state only one short planning claim for the selected route. If the current phase is a non-terminal incremental_refinement and a parent node is present, the new step must add exactly one explicit local delta beyond the parent: one correction, one boundary condition, or one control parameter. The thought_step itself must name that new local delta and must not paraphrase the parent claim. Surface that same delta in equations, quantities, boundary_conditions, or known_vars with a short explicit marker such as active_correction, active_boundary_condition, or active_control_parameter. Do not use markdown."
            )
        },
        "reflection": {
            "prompt_fragment": (
                "You are the ToT modeling model refining an existing branch. Return only a JSON object with keys thought_step, equations, known_vars, used_models, quantities, boundary_conditions. "
                "Make exactly one local revision step. Do not restart the full solution. If the current phase is strategy_scan, keep the revision route-local and atomic. If the latest critique says the child repeated its parent, fix that by adding exactly one explicit local delta: one correction, one boundary condition, or one control parameter. The revised thought_step itself must name that delta instead of paraphrasing the parent claim. Surface that same delta in equations, quantities, boundary_conditions, or known_vars with a short explicit marker such as active_correction, active_boundary_condition, or active_control_parameter. Do not use markdown."
            )
        },
        "evaluation": {
            "prompt_fragment": (
                "You are the ToT review model. Return only a JSON object with keys physical_consistency, variable_grounding, contextual_relevance, simplicity_hint, reason, hard_rule_violations. "
                "You do not receive the full problem statement; score only against the local node state and the currently selected subtask. "
                "Use numeric values in [0,1] and an array for hard_rule_violations. Do not use markdown."
            )
        },
        "delete-review": {
            "prompt_fragment": (
                "You are the ToT audit model reviewing a node deletion request. Return only a JSON object with keys approved, reason, risk_level. Do not use markdown."
            )
        },
    }
    return dict(
        fallback_contracts.get(
            normalized_stage,
            {"prompt_fragment": "Return only a JSON object. Do not use markdown."},
        )
    )


def _load_stage_prompt_contract(stage: str) -> dict[str, Any]:
    try:
        skills_module = import_module("skills")
        invoke_skill = getattr(skills_module, "invoke_skill")
        result = invoke_skill("tot_stage_prompt_contract", {"stage": stage})
        if isinstance(result, dict) and result.get("prompt_fragment"):
            return dict(result)
    except Exception:
        pass
    return _fallback_stage_prompt_contract(stage)


def _coerce_meta_task_step_index(raw_value: Any, *, total_steps: int, fallback: int) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        parsed = fallback
    if total_steps <= 0:
        return max(0, parsed)
    return max(0, min(parsed, total_steps - 1))


def _build_meta_task_progress(
    meta_task: dict[str, Any],
    *,
    step_index: int,
    existing_progress: Any = None,
) -> dict[str, Any]:
    step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
    first_step = str(meta_task.get("first_step", "")).strip()
    total_steps = len(step_ordering)
    progress = dict(existing_progress) if isinstance(existing_progress, dict) else {}
    current_step_index = _coerce_meta_task_step_index(
        progress.get("current_step_index", step_index),
        total_steps=total_steps,
        fallback=step_index,
    )

    if step_ordering:
        current_step = step_ordering[current_step_index]
        previous_steps = step_ordering[:current_step_index]
        remaining_steps = step_ordering[current_step_index + 1 :]
    else:
        current_step = str(progress.get("current_step", first_step)).strip() or first_step
        previous_steps = [str(item) for item in progress.get("previous_steps", []) if str(item).strip()]
        remaining_steps = [str(item) for item in progress.get("remaining_steps", []) if str(item).strip()]

    phase = "strategy_scan" if current_step_index == 0 else "incremental_refinement"
    if phase == "strategy_scan":
        current_step_guidance = (
            "Analyze the next-step strategy space at planning level. "
            "Make one route-local planning claim only, keep all other routes deferred, and do not solve the final answer yet."
        )
    else:
        current_step_guidance = (
            f"Refine only the current subproblem: {current_step}. "
            "Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pending fixes deferred."
        )

    return {
        "current_step_index": current_step_index,
        "current_step": current_step,
        "current_step_guidance": current_step_guidance,
        "previous_steps": previous_steps,
        "remaining_steps": remaining_steps,
        "total_steps": total_steps,
        "phase": phase,
        "is_terminal_step": bool(step_ordering) and current_step_index >= total_steps - 1,
        "route_options": _coerce_structured_reasoning_list(meta_task.get("route_options", [])),
        "step_blueprints": _coerce_structured_reasoning_list(meta_task.get("step_blueprints", [])),
    }


def _propagate_meta_task(
    problem_context: dict[str, Any],
    meta_task: dict[str, Any],
    *,
    step_index: int = 0,
) -> dict[str, Any]:
    normalized_context = deepcopy(problem_context)
    normalized_meta_task = _coerce_model_payload(MetaAnalysisPayload, dict(meta_task))
    normalized_context["meta_task"] = normalized_meta_task
    normalized_context["meta_task_progress"] = _build_meta_task_progress(
        normalized_meta_task,
        step_index=step_index,
        existing_progress=normalized_context.get("meta_task_progress"),
    )

    children = normalized_context.get("children")
    if isinstance(children, list):
        propagated_children: list[Any] = []
        child_step_index = normalized_context["meta_task_progress"]["current_step_index"] + 1
        for item in children:
            if isinstance(item, dict):
                child_context = deepcopy(item)
                child_meta_task = child_context.get("meta_task")
                if not isinstance(child_meta_task, dict) or not child_meta_task:
                    child_meta_task = meta_task
                child_context = _propagate_meta_task(
                    child_context,
                    child_meta_task,
                    step_index=child_step_index,
                )
                propagated_children.append(child_context)
                continue
            propagated_children.append(item)
        normalized_context["children"] = propagated_children

    return normalized_context


def _serialize_raw_response_for_repair(raw_response: Any) -> str:
    if isinstance(raw_response, str):
        return raw_response
    try:
        return json.dumps(raw_response, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(raw_response)


class ChatBackendError(RuntimeError):
    """Base class for local chat backend failures."""


class ChatBackendTransportError(ChatBackendError):
    """Transport-level or HTTP failures when calling the local chat backend."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ChatBackendResponseError(ChatBackendError):
    """Raised when the local chat backend returns an unusable payload."""


def _model_dump(model: BaseModel) -> dict[str, Any]:
    try:
        return model.model_dump()
    except AttributeError:
        return model.dict()


def _build_model(model_type: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    unexpected = set(payload) - _model_field_names(model_type)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"Unexpected fields for {model_type.__name__}: {names}")
    try:
        return model_type.model_validate(payload)
    except AttributeError:
        return model_type.parse_obj(payload)


def _extract_json_payload(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            candidates.append(stripped[first_newline + 1:last_fence].strip())
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start:end + 1])
    candidates.extend(_extract_balanced_json_object_candidates(stripped))

    valid_payloads: list[tuple[int, int, dict[str, Any]]] = []
    for candidate in candidates:
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            valid_payloads.append((len(loaded), len(candidate), dict(loaded)))
    if valid_payloads:
        return max(valid_payloads, key=lambda item: (item[0], item[1]))[2]
    raise ValueError("Chat backend response did not contain a valid JSON object payload.")


def _extract_balanced_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: Optional[int] = None
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if in_string:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidate = text[start_index:index + 1].strip()
                if candidate:
                    candidates.append(candidate)
                start_index = None

    return candidates


def _content_to_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        preferred_parts: list[str] = []
        fallback_parts: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                fallback_parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_parts: list[str] = []
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                item_parts.append(text.strip())
            nested_content = item.get("content")
            if isinstance(nested_content, str) and nested_content.strip():
                item_parts.append(nested_content.strip())
            elif isinstance(nested_content, list):
                nested_text = _content_to_text(nested_content)
                if isinstance(nested_text, str) and nested_text.strip():
                    item_parts.append(nested_text.strip())
            if not item_parts:
                continue

            item_type = str(item.get("type", "")).strip().lower()
            target_parts = preferred_parts if item_type and item_type not in {"reasoning", "analysis"} else fallback_parts
            target_parts.extend(item_parts)
        if preferred_parts:
            return "\n".join(preferred_parts)
        if fallback_parts:
            return "\n".join(fallback_parts)
    return None


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            text = _coerce_string_scalar(item)
            if text:
                items.append(text)
        return items
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    text = _coerce_string_scalar(value)
    return [text] if text else []


def _coerce_string_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, dict):
        for key in (
            "action",
            "step",
            "first_step",
            "current_step",
            "selected_task",
            "step_focus",
            "thought_step",
            "objective",
            "title",
            "name",
            "summary",
            "description",
            "reason",
            "guidance",
            "text",
            "content",
        ):
            if key not in value:
                continue
            text = _coerce_string_scalar(value.get(key))
            if text:
                return text

        parts: list[str] = []
        for nested_value in value.values():
            text = _coerce_string_scalar(nested_value)
            if text and text not in parts:
                parts.append(text)
        return "; ".join(parts)
    if isinstance(value, (list, tuple, set)):
        parts: list[str] = []
        for item in value:
            text = _coerce_string_scalar(item)
            if text and text not in parts:
                parts.append(text)
        return "; ".join(parts)
    return str(value).strip()


def _dedupe_structured_reasoning_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not item:
            continue
        signature = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)
    return deduped


def _coerce_structured_reasoning_item(value: Any, *, default_status: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        label = ""
        for key in (
            "label",
            "action",
            "selected_task",
            "step_focus",
            "step",
            "title",
            "name",
            "summary",
            "description",
            "text",
            "content",
            "current_step",
            "first_step",
        ):
            label = _coerce_string_scalar(value.get(key))
            if label:
                break
        if label:
            normalized["label"] = label

        for target_key, source_keys in (
            ("route_family", ("route_family", "route", "family", "perspective")),
            ("step_type", ("step_type", "task_type", "phase", "mode")),
            ("target_quantity", ("target_quantity", "target", "quantity", "focus")),
            (
                "correction_mode",
                (
                    "correction_mode",
                    "correction_style",
                    "correction_family",
                    "closure_strategy",
                    "error_model",
                    "parameterization",
                ),
            ),
            (
                "correction_target",
                (
                    "correction_target",
                    "correction_quantity",
                    "correction_term",
                    "correction_focus",
                    "target_correction",
                    "deferred_correction",
                ),
            ),
            ("guidance", ("guidance", "current_step_guidance", "description", "rationale")),
            ("status", ("status", "selection", "role")),
            ("slot", ("slot", "reasoning_slot", "distribution_slot")),
        ):
            for source_key in source_keys:
                text = _coerce_string_scalar(value.get(source_key))
                if text:
                    normalized[target_key] = text
                    break

        for target_key, source_keys in (
            ("governing_models", ("governing_models", "used_models", "models")),
            ("assumptions", ("assumptions",)),
            ("deferred_terms", ("deferred_terms", "deferred_tasks", "pending_terms", "corrections")),
            ("completion_signals", ("completion_signals",)),
        ):
            aggregated: list[str] = []
            for source_key in source_keys:
                aggregated.extend(_coerce_string_list(value.get(source_key)))
            if aggregated:
                normalized[target_key] = list(dict.fromkeys(aggregated))

        priority = value.get("priority")
        if isinstance(priority, (int, float)):
            normalized["priority"] = priority

        if default_status and not normalized.get("status"):
            normalized["status"] = default_status

        if not normalized.get("label"):
            fallback_label = _coerce_string_scalar(value)
            if fallback_label:
                normalized["label"] = fallback_label
        return normalized

    text = _coerce_string_scalar(value)
    if not text:
        return {}
    normalized = {"label": text}
    if default_status:
        normalized["status"] = default_status
    return normalized


def _coerce_structured_reasoning_list(value: Any, *, default_status: str = "") -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [
            _coerce_structured_reasoning_item(item, default_status=default_status)
            for item in value
        ]
    else:
        items = [_coerce_structured_reasoning_item(value, default_status=default_status)]
    return _dedupe_structured_reasoning_items([item for item in items if item])


def _derive_meta_route_options(payload: dict[str, Any]) -> list[dict[str, Any]]:
    route_options = _coerce_structured_reasoning_list(payload.get("route_options", []))
    if route_options:
        return route_options

    derived: list[dict[str, Any]] = []
    for source_key in ("minimal_subproblems", "step_ordering"):
        derived.extend(_coerce_structured_reasoning_list(payload.get(source_key, [])))
        if derived:
            break
    if not derived:
        derived.extend(_coerce_structured_reasoning_list(payload.get("first_step", "")))
    return _dedupe_structured_reasoning_items(derived)


def _derive_meta_step_blueprints(payload: dict[str, Any]) -> list[dict[str, Any]]:
    step_blueprints = _coerce_structured_reasoning_list(payload.get("step_blueprints", []))
    if step_blueprints:
        return step_blueprints

    derived: list[dict[str, Any]] = []
    for index, item in enumerate(payload.get("step_ordering", [])):
        normalized = _coerce_structured_reasoning_item(item)
        if not normalized:
            continue
        normalized.setdefault("step_type", "strategy_scan" if index == 0 else "incremental_refinement")
        normalized.setdefault("slot", str(index))
        derived.append(normalized)
    if derived:
        return _dedupe_structured_reasoning_items(derived)

    for item in payload.get("minimal_subproblems", []):
        normalized = _coerce_structured_reasoning_item(item)
        if normalized:
            derived.append(normalized)
    return _dedupe_structured_reasoning_items(derived)


def _derive_orchestrator_candidate_tasks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_tasks = _coerce_structured_reasoning_list(payload.get("candidate_tasks", []))
    if candidate_tasks:
        return candidate_tasks

    derived: list[dict[str, Any]] = []
    selected_route_family = _coerce_string_scalar(payload.get("selected_route_family", ""))
    selected = _coerce_structured_reasoning_item(payload.get("selected_task", ""), default_status="selected")
    if selected_route_family and not selected.get("route_family"):
        selected["route_family"] = selected_route_family
    if selected:
        derived.append(selected)

    selected_label = _coerce_string_scalar(selected.get("label", ""))

    def _duplicates_selected(item: dict[str, Any]) -> bool:
        item_label = _coerce_string_scalar(item.get("label", ""))
        return bool(selected_label and item_label and item_label == selected_label)

    for item in _coerce_structured_reasoning_list(payload.get("task_breakdown", []), default_status="candidate"):
        if _duplicates_selected(item):
            continue
        if selected_route_family and not item.get("route_family"):
            item["route_family"] = selected_route_family
        derived.append(item)
    for item in _coerce_structured_reasoning_list(payload.get("deferred_tasks", []), default_status="deferred"):
        if _duplicates_selected(item):
            continue
        if selected_route_family and not item.get("route_family"):
            item["route_family"] = selected_route_family
        derived.append(item)
    return _dedupe_structured_reasoning_items(derived)


def _derive_selected_route_family(payload: dict[str, Any], candidate_tasks: list[dict[str, Any]]) -> str:
    selected_route_family = _coerce_string_scalar(payload.get("selected_route_family", ""))
    if selected_route_family:
        return selected_route_family
    for item in candidate_tasks:
        if str(item.get("status", "")).strip().lower() == "selected":
            route_family = _coerce_string_scalar(item.get("route_family", ""))
            if route_family:
                return route_family
    return ""


def _selected_candidate_task(candidate_tasks: Any) -> dict[str, Any]:
    for item in _coerce_structured_reasoning_list(candidate_tasks):
        if str(item.get("status", "")).strip().lower() == "selected":
            return item
    return {}


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        return {stripped: None} if stripped else {}
    if not isinstance(value, list):
        return {}

    normalized: dict[str, Any] = {}
    for item in value:
        if isinstance(item, dict):
            normalized.update({str(key): nested_value for key, nested_value in item.items()})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            normalized[str(item[0])] = item[1]
            continue
        text = str(item).strip()
        if text:
            normalized[text] = None
    return normalized


def _coerce_optional_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _coerce_optional_number_or_none(value: Any) -> Any:
    coerced = _coerce_optional_number(value)
    if isinstance(coerced, str):
        return None
    return coerced


def _coerce_evaluation_field_aliases(
    normalized: dict[str, Any],
    field_names: set[str],
) -> dict[str, Any]:
    expected_fields = {
        "physical_consistency",
        "variable_grounding",
        "contextual_relevance",
        "simplicity_hint",
        "score",
        "reason",
        "hard_rule_violations",
    }
    if not {"physical_consistency", "contextual_relevance"}.issubset(field_names):
        return normalized

    for alias in list(normalized):
        normalized_alias = str(alias).strip().lower()
        if normalized_alias in expected_fields:
            continue
        match = get_close_matches(normalized_alias, sorted(expected_fields), n=1, cutoff=0.72)
        if not match:
            continue
        normalized.setdefault(match[0], normalized.get(alias))
        normalized.pop(alias, None)
    return normalized


def _coerce_model_payload(model_type: type[BaseModel], payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    field_names = _model_field_names(model_type)
    normalized = _coerce_evaluation_field_aliases(normalized, field_names)

    for field_name in (
        "equations",
        "used_models",
        "hard_rule_violations",
        "givens",
        "unknowns",
        "minimal_subproblems",
        "step_ordering",
        "completion_signals",
        "task_breakdown",
        "deferred_tasks",
    ):
        if field_name in field_names and field_name in normalized:
            normalized[field_name] = _coerce_string_list(normalized[field_name])

    for field_name in ("known_vars", "quantities", "boundary_conditions"):
        if field_name in field_names and field_name in normalized:
            normalized[field_name] = _coerce_mapping(normalized[field_name])

    for field_name in (
        "thought_step",
        "reason",
        "objective",
        "first_step",
        "step_focus",
        "current_step_guidance",
        "selected_task",
        "risk_level",
    ):
        if field_name in field_names and field_name in normalized:
            normalized[field_name] = _coerce_string_scalar(normalized[field_name])

    for field_name in (
        "physical_consistency",
        "variable_grounding",
        "contextual_relevance",
        "score",
    ):
        if field_name in field_names and field_name in normalized:
            normalized[field_name] = _coerce_optional_number(normalized[field_name])

    if "simplicity_hint" in field_names and "simplicity_hint" in normalized:
        normalized["simplicity_hint"] = _coerce_optional_number_or_none(normalized["simplicity_hint"])

    if "route_options" in field_names:
        normalized["route_options"] = _derive_meta_route_options(payload)
    if "step_blueprints" in field_names:
        normalized["step_blueprints"] = _derive_meta_step_blueprints(payload)
    if "candidate_tasks" in field_names:
        normalized["candidate_tasks"] = _derive_orchestrator_candidate_tasks(payload)
    if "selected_route_family" in field_names:
        normalized["selected_route_family"] = _derive_selected_route_family(
            payload,
            normalized.get("candidate_tasks", []),
        )

    return normalized


def _normalize_chat_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        wrapper_keys = {"choices", "content", "data", "message", "output", "response", "text"}
        if not (set(response) & wrapper_keys):
            return dict(response)
        for key in ("output", "response", "content", "text", "data"):
            value = response.get(key)
            if isinstance(value, dict):
                return dict(value)
            content_text = _content_to_text(value)
            if content_text is not None:
                return _extract_json_payload(content_text)
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content", message.get("text"))
            if isinstance(content, dict):
                return dict(content)
            content_text = _content_to_text(content)
            if content_text is not None:
                return _extract_json_payload(content_text)
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content_text = _content_to_text(message.get("content", message.get("text")))
                    if content_text is not None:
                        return _extract_json_payload(content_text)
                for key in ("output", "content", "text"):
                    value = first.get(key)
                    if isinstance(value, dict):
                        return dict(value)
                    content_text = _content_to_text(value)
                    if content_text is not None:
                        return _extract_json_payload(content_text)
    if isinstance(response, str):
        return _extract_json_payload(response)
    raise TypeError("Chat backend response must be a dictionary or string payload.")


class LocalChatAPIClient:
    """Thin client for the local chat API exposed at ``/api/v1/chat``."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_CHAT_API_URL,
        timeout: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        requester: Optional[ChatRequester] = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative.")

        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._requester = requester or self._default_requester

    def chat(self, *, model: str, system_prompt: str, input_text: str) -> Any:
        normalized_model = str(model).strip()
        if not normalized_model:
            raise ValueError("model must be a non-empty string.")

        normalized_input_text = str(input_text).strip()
        if not normalized_input_text:
            raise ValueError("input_text must be a non-empty string.")

        payload = {
            "model": normalized_model,
            "system_prompt": "" if system_prompt is None else str(system_prompt),
            "input": normalized_input_text,
        }
        for attempt_index in range(self.max_retries + 1):
            try:
                response = self._requester(self.base_url, payload, self.timeout)
            except Exception as exc:
                normalized_exc = self._normalize_request_exception(exc)
                if attempt_index >= self.max_retries or not self._should_retry(normalized_exc):
                    if normalized_exc is exc:
                        raise
                    raise normalized_exc from exc
                self._sleep_before_retry(attempt_index)
                continue

            if response is None:
                raise ChatBackendResponseError("Chat backend returned no response body.")
            if isinstance(response, (bytes, bytearray)):
                response = response.decode("utf-8", errors="replace")
            if isinstance(response, str) and not response.strip():
                raise ChatBackendResponseError("Chat backend returned an empty response body.")
            return response

        raise ChatBackendTransportError("Chat backend request exhausted all retry attempts.")

    def _normalize_request_exception(self, exc: Exception) -> Exception:
        if isinstance(exc, ChatBackendError):
            return exc
        if isinstance(exc, (TimeoutError, socket.timeout)):
            return ChatBackendTransportError(
                f"Timed out after {self.timeout:.1f}s waiting for chat backend at {self.base_url}"
            )
        if isinstance(exc, HTTPError):
            details = exc.read().decode("utf-8", errors="replace")
            return ChatBackendTransportError(
                f"Chat backend returned HTTP {exc.code}: {details}",
                status_code=exc.code,
                response_body=details,
            )
        if isinstance(exc, URLError):
            return ChatBackendTransportError(
                f"Failed to reach chat backend at {self.base_url}: {exc.reason}"
            )
        return exc

    def _should_retry(self, exc: Exception) -> bool:
        if isinstance(exc, ChatBackendTransportError):
            return exc.status_code is None or exc.status_code in TRANSIENT_HTTP_STATUS_CODES
        return False

    def _sleep_before_retry(self, attempt_index: int) -> None:
        if self.retry_backoff_seconds <= 0:
            return
        time.sleep(self.retry_backoff_seconds * (2**attempt_index))

    def _default_requester(self, url: str, payload: dict[str, Any], timeout: float) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw


class ProposalRequest(BaseModel):
    """Strict request schema for the proposal backend."""

    attempt_index: int
    problem_context: dict[str, Any] = Field(default_factory=dict)
    current_node: NodeSnapshot
    parent_node: Optional[NodeSnapshot] = None


class EvaluationRequest(BaseModel):
    """Strict request schema for the evaluation backend."""

    attempt_index: int
    problem_context: dict[str, Any] = Field(default_factory=dict)
    current_node: NodeSnapshot


class ReflectionRequest(BaseModel):
    """Strict request schema for the reflection backend."""

    attempt_index: int
    problem_context: dict[str, Any] = Field(default_factory=dict)
    current_node: NodeSnapshot
    latest_critique: str = ""


class OrchestratorRequest(BaseModel):
    """Strict request schema for the per-step orchestration backend."""

    attempt_index: int
    target_stage: str = ""
    problem_context: dict[str, Any] = Field(default_factory=dict)
    current_node: NodeSnapshot
    parent_node: Optional[NodeSnapshot] = None
    latest_critique: str = ""


class DeleteNodeReviewRequest(BaseModel):
    """Strict request schema for AI-reviewed node deletion."""

    requested_by: str = ""
    reason: str = ""
    current_root_id: Optional[str] = None
    current_frontier_size: int = 0
    target_node: NodeSnapshot
    parent_node: Optional[NodeSnapshot] = None
    descendant_count: int = 0
    is_frontier_node: bool = False
    is_expanded_node: bool = False


class DeleteNodeReviewDecision(BaseModel):
    """Structured AI review decision for node deletion."""

    approved: bool
    reason: str = ""
    risk_level: str = "medium"


class ReasoningBackendAdapter(ABC):
    """Backend adapter contract for propose/evaluate/reflect stages."""

    name: str = "abstract-backend"

    def prepare_problem_context(self, problem_context: dict[str, Any]) -> dict[str, Any]:
        return dict(problem_context)

    @abstractmethod
    def propose(self, request: ProposalRequest) -> ProposalPayload | dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, request: EvaluationRequest) -> EvaluationPayload | dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def reflect(self, request: ReflectionRequest) -> ReflectionPayload | dict[str, Any]:
        raise NotImplementedError()


class NodeDeletionReviewAdapter(ABC):
    """Backend adapter contract for AI-reviewed node deletion."""

    name: str = "abstract-node-deletion-review"

    @abstractmethod
    def review_delete_node(
        self,
        request: DeleteNodeReviewRequest,
    ) -> DeleteNodeReviewDecision | dict[str, Any]:
        raise NotImplementedError()


class DeterministicContextBackendAdapter(ReasoningBackendAdapter):
    """Deterministic backend adapter that reads staged payloads from problem context."""

    name = "deterministic-context"

    def __init__(self, problem_context: dict[str, Any]) -> None:
        self.problem_context = problem_context

    def propose(self, request: ProposalRequest) -> dict[str, Any]:
        return self._select_payload("proposal", request.attempt_index)

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        return self._select_payload("evaluation", request.attempt_index)

    def reflect(self, request: ReflectionRequest) -> dict[str, Any]:
        return self._select_payload("reflection", request.attempt_index)

    def _select_payload(self, key: str, attempt_index: int) -> dict[str, Any]:
        payload = self.problem_context.get(key, {})
        if isinstance(payload, list):
            if not payload:
                return {}
            index = min(attempt_index, len(payload) - 1)
            selected = payload[index]
            return dict(selected) if isinstance(selected, dict) else {}
        return dict(payload) if isinstance(payload, dict) else {}


class LocalChatDualModelBackendAdapter(ReasoningBackendAdapter):
    """Reasoning backend that calls the local chat API with planning, modeling, and review models."""

    name = "local-chat-dual-model"
    META_ANALYSIS_TEXT_LIMIT = 1600
    META_ANALYSIS_SHORT_TEXT_LIMIT = 640
    META_ANALYSIS_LIST_LIMIT = 4
    META_ANALYSIS_SHORT_LIST_LIMIT = 2
    META_ANALYSIS_MAP_LIMIT = 4
    ORCHESTRATOR_TEXT_LIMIT = 240
    ORCHESTRATOR_SHORT_TEXT_LIMIT = 120
    ORCHESTRATOR_LIST_LIMIT = 4
    ORCHESTRATOR_SHORT_LIST_LIMIT = 2
    ORCHESTRATOR_MAP_LIMIT = 4
    REASONING_TEXT_LIMIT = 360
    REASONING_SHORT_TEXT_LIMIT = 180
    REASONING_LIST_LIMIT = 6
    REASONING_SHORT_LIST_LIMIT = 3
    REASONING_MAP_LIMIT = 6
    LOCAL_ROUTE_SCAN_TEXT_LIMIT = 2400

    def __init__(
        self,
        *,
        client: Optional[LocalChatAPIClient] = None,
        base_url: str = DEFAULT_CHAT_API_URL,
        timeout: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        requester: Optional[ChatRequester] = None,
        planning_model: str = DEFAULT_PLANNING_MODEL,
        modeling_model: str = DEFAULT_MODELING_MODEL,
        review_model: str = DEFAULT_REVIEW_MODEL,
        non_terminal_evaluation_model: str = DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    ) -> None:
        self.client = client or LocalChatAPIClient(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            requester=requester,
        )
        self.planning_model = planning_model
        self.modeling_model = modeling_model
        self.review_model = review_model
        self.non_terminal_evaluation_model = non_terminal_evaluation_model

    def prepare_problem_context(self, problem_context: dict[str, Any]) -> dict[str, Any]:
        normalized_context = deepcopy(problem_context)
        existing_meta_task = normalized_context.get("meta_task")
        if isinstance(existing_meta_task, dict) and existing_meta_task:
            return _propagate_meta_task(normalized_context, existing_meta_task)

        # Keep the full problem statement off the planning/orchestration path at
        # session creation time by generating the root meta-task locally.
        return _propagate_meta_task(
            normalized_context,
            self._build_local_meta_analysis(normalized_context),
        )

    def _should_use_local_fast_meta_analysis(self, problem_context: dict[str, Any]) -> bool:
        allowed_keys = {"problem_statement", "givens", "unknowns", "task", "notes", "known_context"}
        for key, value in problem_context.items():
            if key not in allowed_keys and value not in (None, "", [], {}):
                return False

        task = str(problem_context.get("task", "")).strip()
        if task and len(task) > self.META_ANALYSIS_SHORT_TEXT_LIMIT:
            return False

        notes = problem_context.get("notes", [])
        if notes not in (None, []):
            if not isinstance(notes, list):
                return False
            normalized_notes = [str(item).strip() for item in notes if str(item).strip()]
            if len(normalized_notes) > 4:
                return False
            if any(len(item) > self.META_ANALYSIS_SHORT_TEXT_LIMIT for item in normalized_notes):
                return False

        known_context = problem_context.get("known_context", {})
        if known_context not in (None, {}):
            if not isinstance(known_context, dict):
                return False
            normalized_known_context = {
                str(key).strip(): str(value).strip()
                for key, value in known_context.items()
                if str(key).strip() and str(value).strip()
            }
            if set(normalized_known_context) - {"objective", "expected_output"}:
                return False
            if any(len(value) > self.META_ANALYSIS_SHORT_TEXT_LIMIT for value in normalized_known_context.values()):
                return False

        problem_statement = str(problem_context.get("problem_statement", "")).strip()
        return bool(problem_statement) and len(problem_statement) <= self.META_ANALYSIS_SHORT_TEXT_LIMIT

    def _build_fast_local_meta_analysis(self, problem_context: dict[str, Any]) -> dict[str, Any]:
        summarized_context = self._summarize_problem_context_for_meta_analysis(problem_context, minimal=True)
        problem_statement = self._truncate_meta_analysis_text(
            summarized_context.get("problem_statement", ""),
            self.META_ANALYSIS_SHORT_TEXT_LIMIT,
        )
        objective = problem_statement or "Solve the problem through a small sequence of local steps."
        first_step = "identify governing relation"
        step_ordering = [
            first_step,
            "choose one active correction or closure",
            "express the target quantity in known variables",
        ]
        route_options = self._derive_fast_local_route_options(
            self._build_local_meta_analysis_route_seed_text(
                problem_context,
                summarized_context=summarized_context,
            )
        )

        fast_payload = {
            "objective": objective,
            "givens": self._compact_string_list_for_meta_analysis(problem_context.get("givens", []), short=True),
            "unknowns": self._compact_string_list_for_meta_analysis(problem_context.get("unknowns", []), short=True),
            "minimal_subproblems": step_ordering,
            "step_ordering": step_ordering,
            "first_step": first_step,
            "completion_signals": [
                "one route-local seed selected",
                "one active correction or closure fixed",
                "target quantity expressed in known variables",
            ],
            "route_options": route_options,
            "step_blueprints": self._derive_fast_local_step_blueprints(
                first_step=first_step,
                route_options=route_options,
            ),
        }
        return _coerce_model_payload(MetaAnalysisPayload, fast_payload)

    def _derive_fast_local_route_options(self, problem_statement: str) -> list[dict[str, Any]]:
        normalized = problem_statement.lower()
        asks_answer_change = self._contains_any_keyword(
            normalized,
            {
                "answer changes",
                "when the answer changes",
                "parameter regime",
                "regime",
                "flip",
                "threshold",
            },
        )

        if self._contains_any_keyword(normalized, {"terminal", "drag", "steady", "equilibrium"}):
            return [
                {
                    "label": "force-balance route",
                    "route_family": "force-balance",
                    "governing_models": ["Newton's Second Law"],
                    "guidance": "Name the steady-state balance or one dominant force term only.",
                    "correction_mode": "full-force inventory",
                    "correction_target": "active force term",
                },
                {
                    "label": "force-balance route with minimal closure",
                    "route_family": "force-balance",
                    "governing_models": ["Newton's Second Law"],
                    "guidance": "Fix one reduced balance first by choosing a minimal closure or neglected term only.",
                    "correction_mode": "minimal closure first",
                    "correction_target": "neglected force term",
                },
                {
                    "label": "drag-closure route",
                    "route_family": "drag-closure",
                    "governing_models": ["Constitutive drag law"],
                    "guidance": "Pick one drag-law family or closure without solving for the final value.",
                    "correction_mode": "closure-family scan",
                    "correction_target": "drag law",
                },
                {
                    "label": "scaling route",
                    "route_family": "scaling",
                    "governing_models": ["Dimensional Analysis"],
                    "guidance": "State one dominant-balance or limiting-regime estimate only.",
                    "correction_mode": "dominant-balance closure",
                    "correction_target": "dominant regime",
                },
                {
                    "label": "energy-dissipation route",
                    "route_family": "energy-dissipation",
                    "governing_models": ["Power balance"],
                    "guidance": "State one energy or power balance that could isolate the steady regime only.",
                    "correction_mode": "dissipation-balance scan",
                    "correction_target": "dissipation mechanism",
                },
                {
                    "label": "regime-map route",
                    "route_family": "regime-map",
                    "governing_models": ["Dimensionless regime comparison"],
                    "guidance": "Identify one regime split or drag-law applicability test before choosing a closure.",
                    "correction_mode": "regime-selection scan",
                    "correction_target": "applicable drag regime",
                },
            ]

        if self._contains_any_keyword(
            normalized,
            {"speed", "velocity", "energy", "force", "mass", "friction", "incline", "acceleration", "motion"},
        ):
            second_route = (
                {
                    "label": "regime-map route",
                    "route_family": "regime-map",
                    "governing_models": ["Dimensionless regime comparison"],
                    "guidance": "Identify one stage boundary or parameter threshold where the answer could flip before committing to a derivation.",
                    "correction_mode": "regime-selection scan",
                    "correction_target": "answer-flip boundary",
                }
                if asks_answer_change
                else {
                    "label": "energy route with one loss term",
                    "route_family": "energy",
                    "governing_models": ["Work-Energy Theorem"],
                    "guidance": "Keep the energy route but pick exactly one active loss term or closure only.",
                    "correction_mode": "single-loss activation",
                    "correction_target": "active loss term",
                }
            )
            return [
                {
                    "label": "energy route",
                    "route_family": "energy",
                    "governing_models": ["Work-Energy Theorem"],
                    "guidance": "Name one governing energy relation or one deferred loss term only.",
                    "correction_mode": "lossless baseline first",
                    "correction_target": "dissipation term",
                },
                second_route,
                {
                    "label": "force-balance route",
                    "route_family": "force-balance",
                    "governing_models": ["Newton's Second Law"],
                    "guidance": "Name one dominant force balance or one decisive force component only.",
                    "correction_mode": "direct force inventory",
                    "correction_target": "active force term",
                },
                {
                    "label": "kinematic route",
                    "route_family": "kinematics",
                    "governing_models": ["Kinematic relation"],
                    "guidance": "Name one state-transition relation or one piecewise propagation step only.",
                    "correction_mode": "piecewise-state propagation",
                    "correction_target": "state transition",
                },
                {
                    "label": "momentum route",
                    "route_family": "momentum",
                    "governing_models": ["Momentum balance"],
                    "guidance": "Name one momentum-transfer relation or one impulse-style approximation only.",
                    "correction_mode": "impulse-balance scan",
                    "correction_target": "momentum exchange",
                },
                {
                    "label": "scaling route",
                    "route_family": "scaling",
                    "governing_models": ["Dimensional Analysis"],
                    "guidance": "State one dominant regime, small parameter, or leading-order estimate only.",
                    "correction_mode": "dominant-balance closure",
                    "correction_target": "dominant regime",
                },
            ]

        if self._contains_any_keyword(
            normalized,
            {"probability", "random", "stochastic", "distribution", "expectation", "variance"},
        ):
            return [
                {
                    "label": "distribution route",
                    "route_family": "distribution",
                    "governing_models": ["Probability distribution"],
                    "guidance": "Name one governing distribution or one random variable family only.",
                    "correction_mode": "distribution-family scan",
                    "correction_target": "distribution choice",
                },
                {
                    "label": "conditioning route",
                    "route_family": "conditioning",
                    "governing_models": ["Law of Total Probability"],
                    "guidance": "Split on exactly one conditioning event or latent case only.",
                    "correction_mode": "single conditioning split",
                    "correction_target": "conditioning event",
                },
                {
                    "label": "expectation route",
                    "route_family": "expectation",
                    "governing_models": ["Expectation identity"],
                    "guidance": "State one expectation, moment, or averaging relation only.",
                    "correction_mode": "moment-closure scan",
                    "correction_target": "moment closure",
                },
                {
                    "label": "scaling route",
                    "route_family": "scaling",
                    "governing_models": ["Asymptotic scaling"],
                    "guidance": "State one limiting regime, variance scale, or dominant asymptotic term only.",
                    "correction_mode": "dominant-balance closure",
                    "correction_target": "dominant regime",
                },
                {
                    "label": "recursion route",
                    "route_family": "recursion",
                    "governing_models": ["Recursive decomposition"],
                    "guidance": "Write one recursive split, state transition, or conditional recurrence only.",
                    "correction_mode": "state-recursion scan",
                    "correction_target": "recursive state update",
                },
                {
                    "label": "generating-function route",
                    "route_family": "generating-function",
                    "governing_models": ["Generating function"],
                    "guidance": "Name one transform or generating-function representation only.",
                    "correction_mode": "transform-choice scan",
                    "correction_target": "transform representation",
                },
            ]

        return [
            {
                "label": "dependency route",
                "route_family": "dependency",
                "governing_models": ["Dependency graph"],
                "guidance": "Name one governing dependency or one causal relation only.",
                "correction_mode": "dependency-first scan",
                "correction_target": "active dependency",
            },
            {
                "label": "constraint route",
                "route_family": "constraint",
                "governing_models": ["Constraint relation"],
                "guidance": "Name one limiting constraint, boundary condition, or admissibility rule only.",
                "correction_mode": "constraint-first scan",
                "correction_target": "active constraint",
            },
            {
                "label": "invariant route",
                "route_family": "invariant",
                "governing_models": ["Invariant or conserved structure"],
                "guidance": "Name one invariant, symmetry, or conservation-style simplification only.",
                "correction_mode": "invariant-first scan",
                "correction_target": "invariant choice",
            },
            {
                "label": "scaling route",
                "route_family": "scaling",
                "governing_models": ["Dimensional Analysis"],
                "guidance": "State one dominant regime, asymptotic simplification, or size estimate only.",
                "correction_mode": "dominant-balance closure",
                "correction_target": "dominant regime",
            },
            {
                "label": "decomposition route",
                "route_family": "decomposition",
                "governing_models": ["Problem decomposition"],
                "guidance": "Split the task into one decisive substructure or one partial subsystem only.",
                "correction_mode": "subproblem-first scan",
                "correction_target": "active subproblem",
            },
            {
                "label": "extremal route",
                "route_family": "extremal",
                "governing_models": ["Extremal or limiting argument"],
                "guidance": "State one extremal case, bounding argument, or limiting configuration only.",
                "correction_mode": "limit-case scan",
                "correction_target": "extremal case",
            },
        ]

    def _derive_fast_local_step_blueprints(
        self,
        *,
        first_step: str,
        route_options: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        route_families = [
            str(option.get("route_family", "")).strip()
            for option in route_options
            if str(option.get("route_family", "")).strip()
        ]
        route_summary = ", ".join(route_families[:3])
        first_guidance = "Pick exactly one route option and make one tiny route-local claim."
        if route_summary:
            first_guidance = f"{first_guidance} Available route families: {route_summary}."

        return [
            {
                "label": first_step,
                "step_type": "strategy_scan",
                "guidance": f"{first_guidance} Name one governing law/model, one decisive assumption, or one active correction quantity only.",
            },
            {
                "label": "choose one active correction or closure",
                "step_type": "incremental_refinement",
                "guidance": "Choose exactly one deferred correction quantity or closure for the selected route, and defer everything else.",
                "correction_mode": "one correction at a time",
            },
            {
                "label": "express the target quantity in known variables",
                "step_type": "incremental_refinement",
                "guidance": "Write one short target relation in known variables and stop.",
            },
        ]

    def _contains_any_keyword(self, text: str, keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _build_local_meta_analysis_route_seed_text(
        self,
        problem_context: dict[str, Any],
        *,
        summarized_context: Optional[dict[str, Any]] = None,
    ) -> str:
        parts: list[str] = []
        problem_statement = str(problem_context.get("problem_statement", "")).strip()
        if problem_statement:
            parts.append(problem_statement)

        task = str(problem_context.get("task", "")).strip()
        if task:
            parts.append(task)

        notes = problem_context.get("notes", [])
        if isinstance(notes, list):
            parts.extend(str(item).strip() for item in notes[:4] if str(item).strip())

        for key in ("givens", "unknowns"):
            value = problem_context.get(key, [])
            if isinstance(value, list):
                parts.extend(str(item).strip() for item in value[:4] if str(item).strip())

        known_context = problem_context.get("known_context", {})
        if isinstance(known_context, dict):
            parts.extend(
                str(known_context.get(key, "")).strip()
                for key in ("objective", "expected_output")
                if str(known_context.get(key, "")).strip()
            )

        if summarized_context:
            summarized_problem_statement = str(summarized_context.get("problem_statement", "")).strip()
            if summarized_problem_statement and summarized_problem_statement not in parts:
                parts.append(summarized_problem_statement)

        seed_text = " ".join(part for part in parts if part)
        if len(seed_text) <= self.LOCAL_ROUTE_SCAN_TEXT_LIMIT:
            return seed_text
        return seed_text[: self.LOCAL_ROUTE_SCAN_TEXT_LIMIT]

    def _request_meta_analysis(self, problem_context: dict[str, Any]) -> dict[str, Any]:
        primary_request = self._build_meta_analysis_request(problem_context, minimal=False)
        try:
            return self._call_meta_analysis_model(primary_request)
        except ChatBackendResponseError:
            return self._build_local_meta_analysis(problem_context)
        except ChatBackendTransportError as exc:
            if not self._is_context_overflow_error(exc):
                return self._build_local_meta_analysis(problem_context)

        minimal_request = self._build_meta_analysis_request(problem_context, minimal=True)
        try:
            return self._call_meta_analysis_model(minimal_request)
        except ChatBackendResponseError:
            return self._build_local_meta_analysis(problem_context)
        except ChatBackendTransportError as exc:
            if not self._is_context_overflow_error(exc):
                return self._build_local_meta_analysis(problem_context)

        return self._build_local_meta_analysis(problem_context)

    def _build_local_meta_analysis(self, problem_context: dict[str, Any]) -> dict[str, Any]:
        if self._should_use_local_fast_meta_analysis(problem_context):
            return self._build_fast_local_meta_analysis(problem_context)
        return self._build_fallback_meta_analysis(problem_context)

    def _call_meta_analysis_model(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._call_chat_model(
            stage="meta-analysis",
            model=self.planning_model,
            system_prompt=self._stage_system_prompt("meta-analysis"),
            input_payload={"stage": "meta-analysis", "request": request},
            response_model=MetaAnalysisPayload,
        )

    def _build_meta_analysis_request(
        self,
        problem_context: dict[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        return {
            "problem_context": self._summarize_problem_context_for_meta_analysis(
                problem_context,
                minimal=minimal,
            )
        }

    def _summarize_problem_context_for_meta_analysis(
        self,
        problem_context: dict[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        problem_statement = self._truncate_meta_analysis_text(
            problem_context.get("problem_statement", ""),
            self.META_ANALYSIS_SHORT_TEXT_LIMIT if minimal else self.META_ANALYSIS_TEXT_LIMIT,
        )
        if problem_statement:
            summary["problem_statement"] = problem_statement

        child_payload = problem_context.get("children")
        if isinstance(child_payload, list):
            summary["child_context_count"] = len(child_payload)
            if not minimal:
                child_problem_statements = self._compact_string_list_for_meta_analysis(
                    [
                        child.get("problem_statement", "")
                        for child in child_payload
                        if isinstance(child, dict)
                    ],
                    short=True,
                )
                if child_problem_statements:
                    summary["child_problem_statements"] = child_problem_statements

        known_context = problem_context.get("known_context")
        if isinstance(known_context, dict) and known_context and not minimal:
            summary["known_context"] = self._compact_mapping_for_meta_analysis(
                known_context,
                limit=self.META_ANALYSIS_MAP_LIMIT,
                short=False,
            )

        for key, value in problem_context.items():
            if key in {
                "problem_statement",
                "children",
                "known_context",
                "meta_task",
                "meta_task_progress",
                "orchestrator_task",
                "proposal",
                "calculation",
                "evaluation",
                "reflection",
            }:
                continue
            compact_value = self._compact_scalar_for_meta_analysis(value, short=minimal)
            if compact_value in ("", [], {}, None):
                continue
            summary[str(key)] = compact_value

        return summary

    def _build_fallback_meta_analysis(self, problem_context: dict[str, Any]) -> dict[str, Any]:
        summarized_context = self._summarize_problem_context_for_meta_analysis(problem_context, minimal=True)
        problem_statement = self._truncate_meta_analysis_text(
            summarized_context.get("problem_statement", ""),
            self.META_ANALYSIS_SHORT_TEXT_LIMIT,
        )
        objective = problem_statement or "Solve the problem through a small sequence of local steps."
        first_step = "identify governing relation"
        step_ordering = [
            first_step,
            "choose one active correction or closure",
            "express the target quantity in known variables",
        ]
        route_options = self._derive_fast_local_route_options(
            self._build_local_meta_analysis_route_seed_text(
                problem_context,
                summarized_context=summarized_context,
            )
        )

        fallback_payload = {
            "objective": objective,
            "givens": self._compact_string_list_for_meta_analysis(problem_context.get("givens", []), short=True),
            "unknowns": self._compact_string_list_for_meta_analysis(problem_context.get("unknowns", []), short=True),
            "minimal_subproblems": step_ordering,
            "step_ordering": step_ordering,
            "first_step": first_step,
            "completion_signals": [
                "one route-local seed selected",
                "one active correction or closure fixed",
                "target quantity expressed in known variables",
            ],
            "route_options": route_options,
            "step_blueprints": self._derive_fast_local_step_blueprints(
                first_step=first_step,
                route_options=route_options,
            ),
        }
        return _coerce_model_payload(MetaAnalysisPayload, fallback_payload)

    def propose(self, request: ProposalRequest) -> dict[str, Any]:
        modeling_request, orchestrator_task = self._prepare_modeling_request(
            target_stage="proposal",
            request=request,
            parent_node=request.parent_node,
            latest_critique="",
        )
        compact_request = self._build_compact_reasoning_request(stage="proposal", request=modeling_request)
        proposal = self._call_chat_model(
            stage="proposal",
            model=self._select_modeling_model_for_request(modeling_request.problem_context),
            system_prompt=self._stage_system_prompt("proposal"),
            input_payload={"stage": "proposal", "request": compact_request},
            response_model=ProposalPayload,
        )
        if orchestrator_task:
            known_vars = dict(proposal.get("known_vars", {}))
            known_vars["orchestrator_task"] = dict(orchestrator_task)
            proposal["known_vars"] = known_vars
        return proposal

    def evaluate(self, request: EvaluationRequest) -> dict[str, Any]:
        return self._call_chat_model(
            stage="evaluation",
            model=self._select_evaluation_model_for_request(request.problem_context),
            system_prompt=self._stage_system_prompt("evaluation"),
            input_payload={
                "stage": "evaluate",
                "request": self._build_compact_reasoning_request(stage="evaluation", request=request),
            },
            response_model=EvaluationPayload,
        )

    def reflect(self, request: ReflectionRequest) -> dict[str, Any]:
        modeling_request, orchestrator_task = self._prepare_modeling_request(
            target_stage="reflection",
            request=request,
            parent_node=None,
            latest_critique=request.latest_critique,
        )
        compact_request = self._build_compact_reasoning_request(stage="reflection", request=modeling_request)
        reflection = self._call_chat_model(
            stage="reflection",
            model=self._select_modeling_model_for_request(modeling_request.problem_context),
            system_prompt=self._stage_system_prompt("reflection"),
            input_payload={"stage": "reflect", "request": compact_request},
            response_model=ReflectionPayload,
        )
        if orchestrator_task:
            known_vars = dict(reflection.get("known_vars", {}))
            known_vars["orchestrator_task"] = dict(orchestrator_task)
            reflection["known_vars"] = known_vars
        return reflection

    def _select_modeling_model_for_request(self, problem_context: dict[str, Any]) -> str:
        if self._is_planning_only_modeling_context(problem_context):
            return self.review_model
        return self.modeling_model

    def _select_evaluation_model_for_request(self, problem_context: dict[str, Any]) -> str:
        if self._is_terminal_meta_task_context(problem_context):
            return self.review_model
        return self.non_terminal_evaluation_model

    @staticmethod
    def _is_terminal_meta_task_context(problem_context: dict[str, Any]) -> bool:
        meta_task_progress = problem_context.get("meta_task_progress")
        return isinstance(meta_task_progress, dict) and bool(meta_task_progress.get("is_terminal_step", False))

    @staticmethod
    def _is_planning_only_modeling_context(problem_context: dict[str, Any]) -> bool:
        meta_task_progress = problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False
        phase = str(meta_task_progress.get("phase", "")).strip().lower()
        if phase != "incremental_refinement":
            return False
        if bool(meta_task_progress.get("is_terminal_step", False)):
            return False
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        selected_correction_mode = str(meta_task_progress.get("selected_correction_mode", "")).strip()
        selected_correction_target = str(meta_task_progress.get("selected_correction_target", "")).strip()
        if selected_route_family and (selected_correction_mode or selected_correction_target):
            return True
        route_focus = problem_context.get("route_focus")
        return isinstance(route_focus, dict) and bool(route_focus)

    def _prepare_modeling_request(
        self,
        *,
        target_stage: str,
        request: ProposalRequest | ReflectionRequest,
        parent_node: Optional[NodeSnapshot],
        latest_critique: str,
    ) -> tuple[ProposalRequest | ReflectionRequest, dict[str, Any]]:
        if not self._should_orchestrate(request.problem_context):
            return request, {}

        orchestrator_task = self._request_orchestrator_task(
            target_stage=target_stage,
            request=request,
            parent_node=parent_node,
            latest_critique=latest_critique,
        )
        orchestrated_context = self._with_orchestrator_task(request.problem_context, orchestrator_task)
        if isinstance(request, ProposalRequest):
            return (
                ProposalRequest(
                    attempt_index=request.attempt_index,
                    problem_context=orchestrated_context,
                    current_node=request.current_node,
                    parent_node=request.parent_node,
                ),
                orchestrator_task,
            )
        return (
            ReflectionRequest(
                attempt_index=request.attempt_index,
                problem_context=orchestrated_context,
                current_node=request.current_node,
                latest_critique=request.latest_critique,
            ),
            orchestrator_task,
        )

    @staticmethod
    def _should_orchestrate(problem_context: dict[str, Any]) -> bool:
        return bool(problem_context.get("meta_task") or problem_context.get("meta_task_progress"))

    @staticmethod
    def _with_orchestrator_task(
        problem_context: dict[str, Any],
        orchestrator_task: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_context = deepcopy(problem_context)
        normalized_context["orchestrator_task"] = dict(orchestrator_task)

        updated_progress = (
            dict(normalized_context.get("meta_task_progress"))
            if isinstance(normalized_context.get("meta_task_progress"), dict)
            else {}
        )
        step_focus = str(orchestrator_task.get("step_focus", "")).strip()
        if step_focus:
            updated_progress["current_step"] = step_focus
        guidance = str(orchestrator_task.get("current_step_guidance", "")).strip() or str(
            orchestrator_task.get("selected_task", "")
        ).strip()
        if guidance:
            updated_progress["current_step_guidance"] = guidance
        selected_route_family = str(orchestrator_task.get("selected_route_family", "")).strip()
        if selected_route_family:
            updated_progress["selected_route_family"] = selected_route_family
        selected_candidate_task = _selected_candidate_task(orchestrator_task.get("candidate_tasks", []))
        selected_correction_mode = _coerce_string_scalar(
            selected_candidate_task.get("correction_mode", "")
        )
        if selected_correction_mode:
            updated_progress["selected_correction_mode"] = selected_correction_mode
        selected_correction_target = _coerce_string_scalar(
            selected_candidate_task.get("correction_target", "")
        )
        if selected_correction_target:
            updated_progress["selected_correction_target"] = selected_correction_target
        if updated_progress:
            normalized_context["meta_task_progress"] = updated_progress
        return normalized_context

    def _build_compact_reasoning_request(
        self,
        *,
        stage: str,
        request: ProposalRequest | ReflectionRequest | EvaluationRequest,
    ) -> dict[str, Any]:
        include_full_problem = stage in {"proposal", "reflection"}
        compact_request: dict[str, Any] = {
            "attempt_index": request.attempt_index,
            "schema_id": f"{stage}.v1",
            "problem_context": self._summarize_problem_context_for_reasoning(
                request.problem_context,
                include_full_problem=include_full_problem,
            ),
            "current_node": _model_dump(
                self._summarize_node_snapshot_for_reasoning(request.current_node, short=False)
            ),
        }
        if isinstance(request, ProposalRequest) and request.parent_node is not None:
            compact_request["parent_node"] = _model_dump(
                self._summarize_node_snapshot_for_reasoning(request.parent_node, short=True)
            )
        if isinstance(request, ReflectionRequest):
            compact_request["latest_critique"] = self._truncate_reasoning_text(
                request.latest_critique,
                self.REASONING_SHORT_TEXT_LIMIT,
            )
        return compact_request

    def _summarize_problem_context_for_reasoning(
        self,
        problem_context: dict[str, Any],
        *,
        include_full_problem: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        if include_full_problem:
            problem_statement = self._truncate_reasoning_text(
                problem_context.get("problem_statement", ""),
                self.REASONING_TEXT_LIMIT,
            )
            if problem_statement:
                summary["problem_statement"] = problem_statement

        meta_task = problem_context.get("meta_task")
        if isinstance(meta_task, dict) and meta_task:
            summary["meta_task"] = self._summarize_meta_task_for_reasoning(
                meta_task,
                include_full_problem=include_full_problem,
            )

        meta_task_progress = problem_context.get("meta_task_progress")
        if isinstance(meta_task_progress, dict) and meta_task_progress:
            summary["meta_task_progress"] = self._summarize_meta_task_progress_for_reasoning(meta_task_progress)

        orchestrator_task = problem_context.get("orchestrator_task")
        if isinstance(orchestrator_task, dict) and orchestrator_task:
            summary["orchestrator_task"] = self._summarize_orchestrator_task_for_reasoning(orchestrator_task)

        child_payload = problem_context.get("children")
        if isinstance(child_payload, list):
            summary["child_context_count"] = len(child_payload)

        present_stage_payloads = [
            key for key in ("proposal", "calculation", "evaluation", "reflection") if key in problem_context
        ]
        if present_stage_payloads:
            summary["stage_payloads_present"] = present_stage_payloads

        return summary

    def _summarize_meta_task_for_reasoning(
        self,
        meta_task: dict[str, Any],
        *,
        include_full_problem: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        meta_task_keys = ("objective", "first_step") if include_full_problem else ("first_step",)
        for key in meta_task_keys:
            value = self._truncate_reasoning_text(meta_task.get(key, ""), self.REASONING_TEXT_LIMIT)
            if value:
                summary[key] = value
        summary["step_ordering"] = self._compact_string_list_for_reasoning(
            meta_task.get("step_ordering", []),
            short=False,
        )
        summary["completion_signals"] = self._compact_string_list_for_reasoning(
            meta_task.get("completion_signals", []),
            short=False,
        )
        route_options = self._compact_structured_reasoning_list_for_reasoning(
            meta_task.get("route_options", []),
            short=False,
        )
        if route_options:
            summary["route_options"] = route_options
        step_blueprints = self._compact_structured_reasoning_list_for_reasoning(
            meta_task.get("step_blueprints", []),
            short=False,
        )
        if step_blueprints:
            summary["step_blueprints"] = step_blueprints
        return summary

    def _summarize_meta_task_progress_for_reasoning(self, meta_task_progress: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "current_step_index": meta_task_progress.get("current_step_index", 0),
            "phase": meta_task_progress.get("phase", ""),
            "is_terminal_step": bool(meta_task_progress.get("is_terminal_step", False)),
            "total_steps": meta_task_progress.get("total_steps", 0),
        }
        for key in ("current_step", "current_step_guidance"):
            value = self._truncate_reasoning_text(meta_task_progress.get(key, ""), self.REASONING_TEXT_LIMIT)
            if value:
                summary[key] = value
        summary["previous_steps"] = self._compact_string_list_for_reasoning(
            meta_task_progress.get("previous_steps", []),
            short=True,
        )
        summary["remaining_steps"] = self._compact_string_list_for_reasoning(
            meta_task_progress.get("remaining_steps", []),
            short=False,
        )
        route_options = self._compact_structured_reasoning_list_for_reasoning(
            meta_task_progress.get("route_options", []),
            short=True,
        )
        if route_options:
            summary["route_options"] = route_options
        selected_route_family = self._truncate_reasoning_text(
            meta_task_progress.get("selected_route_family", ""),
            self.REASONING_TEXT_LIMIT,
        )
        if selected_route_family:
            summary["selected_route_family"] = selected_route_family
        selected_correction_mode = self._truncate_reasoning_text(
            meta_task_progress.get("selected_correction_mode", ""),
            self.REASONING_TEXT_LIMIT,
        )
        if selected_correction_mode:
            summary["selected_correction_mode"] = selected_correction_mode
        selected_correction_target = self._truncate_reasoning_text(
            meta_task_progress.get("selected_correction_target", ""),
            self.REASONING_TEXT_LIMIT,
        )
        if selected_correction_target:
            summary["selected_correction_target"] = selected_correction_target
        return summary

    def _summarize_orchestrator_task_for_reasoning(self, orchestrator_task: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in ("step_focus", "current_step_guidance", "selected_task"):
            value = self._truncate_reasoning_text(orchestrator_task.get(key, ""), self.REASONING_TEXT_LIMIT)
            if value:
                summary[key] = value
        summary["deferred_tasks"] = self._compact_string_list_for_reasoning(
            orchestrator_task.get("deferred_tasks", []),
            short=False,
        )
        selected_route_family = self._truncate_reasoning_text(
            orchestrator_task.get("selected_route_family", ""),
            self.REASONING_TEXT_LIMIT,
        )
        if selected_route_family:
            summary["selected_route_family"] = selected_route_family
        selected_candidate_task = _selected_candidate_task(orchestrator_task.get("candidate_tasks", []))
        selected_correction_mode = self._truncate_reasoning_text(
            selected_candidate_task.get("correction_mode", ""),
            self.REASONING_TEXT_LIMIT,
        )
        if selected_correction_mode:
            summary["selected_correction_mode"] = selected_correction_mode
        selected_correction_target = self._truncate_reasoning_text(
            selected_candidate_task.get("correction_target", ""),
            self.REASONING_TEXT_LIMIT,
        )
        if selected_correction_target:
            summary["selected_correction_target"] = selected_correction_target
        candidate_tasks = self._compact_structured_reasoning_list_for_reasoning(
            orchestrator_task.get("candidate_tasks", []),
            short=False,
        )
        if candidate_tasks:
            summary["candidate_tasks"] = candidate_tasks
        return summary

    def _summarize_node_snapshot_for_reasoning(self, node: NodeSnapshot, *, short: bool) -> NodeSnapshot:
        reflection_tail = node.reflection_history[-1:] if short else node.reflection_history[-3:]
        return NodeSnapshot(
            id=node.id,
            parent_id=node.parent_id,
            thought_step=self._truncate_reasoning_text(
                node.thought_step,
                self.REASONING_SHORT_TEXT_LIMIT if short else self.REASONING_TEXT_LIMIT,
            ),
            equations=self._compact_string_list_for_reasoning(node.equations, short=short),
            known_vars=self._summarize_known_vars_for_reasoning(node.known_vars, short=short),
            used_models=self._compact_string_list_for_reasoning(node.used_models, short=short),
            quantities=self._compact_mapping_for_reasoning(node.quantities, short=short),
            boundary_conditions=self._compact_mapping_for_reasoning(node.boundary_conditions, short=short),
            status=node.status,
            fsm_state=node.fsm_state,
            score=node.score,
            reflection_history=self._compact_string_list_for_reasoning(reflection_tail, short=True),
        )

    def _summarize_known_vars_for_reasoning(self, known_vars: dict[str, Any], *, short: bool) -> dict[str, Any]:
        if not isinstance(known_vars, dict) or not known_vars:
            return {}

        summary: dict[str, Any] = {}
        for key in (
            "evaluation_passed",
            "needs_deeper_reasoning",
            "low_score_reason",
            "hard_rule_violations",
            "recoverable_rule_violations",
            "expansion_priority",
            "selected_for_frontier",
            "scheduler_action",
            "route_family",
            "correction_mode",
            "correction_target",
            "distributed_reasoning_slot",
        ):
            if key not in known_vars:
                continue
            value = known_vars[key]
            if key in {"hard_rule_violations", "recoverable_rule_violations"}:
                summary[key] = self._compact_string_list_for_reasoning(value, short=True)
            else:
                summary[key] = self._compact_scalar_for_reasoning(value, short=short)

        hard_rule_check = known_vars.get("hard_rule_check")
        if isinstance(hard_rule_check, dict) and hard_rule_check:
            summary["hard_rule_check"] = {
                "passed": hard_rule_check.get("passed"),
                "violations": self._compact_string_list_for_reasoning(
                    hard_rule_check.get("violations", []),
                    short=True,
                ),
            }

        evaluation_breakdown = known_vars.get("evaluation_breakdown")
        if isinstance(evaluation_breakdown, dict) and evaluation_breakdown:
            summary["evaluation_breakdown"] = self._compact_mapping_for_reasoning(
                evaluation_breakdown,
                short=True,
            )

        orchestrator_task = known_vars.get("orchestrator_task")
        if isinstance(orchestrator_task, dict) and orchestrator_task:
            summary["orchestrator_task"] = self._summarize_orchestrator_task_for_reasoning(orchestrator_task)

        summary["known_var_keys"] = self._compact_string_list_for_reasoning(list(known_vars.keys()), short=short)
        return summary

    def _request_orchestrator_task(
        self,
        *,
        target_stage: str,
        request: ProposalRequest | ReflectionRequest,
        parent_node: Optional[NodeSnapshot],
        latest_critique: str,
    ) -> dict[str, Any]:
        if self._should_use_local_strategy_scan_orchestrator(
            request.problem_context,
            latest_critique=latest_critique,
        ):
            return self._build_fallback_orchestrator_task(
                problem_context=request.problem_context,
                latest_critique=latest_critique,
                target_stage=target_stage,
            )

        primary_request = self._build_orchestrator_request(
            target_stage=target_stage,
            request=request,
            parent_node=parent_node,
            latest_critique=latest_critique,
            minimal=False,
        )
        try:
            return self._call_orchestrator_model(primary_request)
        except ChatBackendTransportError as exc:
            if not self._should_fallback_to_local_orchestrator(exc):
                raise

        minimal_request = self._build_orchestrator_request(
            target_stage=target_stage,
            request=request,
            parent_node=parent_node,
            latest_critique=latest_critique,
            minimal=True,
        )
        try:
            return self._call_orchestrator_model(minimal_request)
        except ChatBackendTransportError as exc:
            if not self._should_fallback_to_local_orchestrator(exc):
                raise

        return self._build_fallback_orchestrator_task(
            problem_context=request.problem_context,
            latest_critique=latest_critique,
            target_stage=target_stage,
        )

    @staticmethod
    def _should_fallback_to_local_orchestrator(exc: ChatBackendTransportError) -> bool:
        return LocalChatDualModelBackendAdapter._is_context_overflow_error(exc) or (
            exc.status_code is None or exc.status_code in TRANSIENT_HTTP_STATUS_CODES
        )

    @staticmethod
    def _should_use_local_strategy_scan_orchestrator(
        problem_context: dict[str, Any],
        *,
        latest_critique: str,
    ) -> bool:
        meta_task_progress = problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False
        phase = str(meta_task_progress.get("phase", "")).strip().lower()
        if phase == "strategy_scan":
            return not str(latest_critique).strip()
        if phase != "incremental_refinement":
            return False
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        selected_correction_mode = str(meta_task_progress.get("selected_correction_mode", "")).strip()
        selected_correction_target = str(meta_task_progress.get("selected_correction_target", "")).strip()
        if selected_route_family and not bool(meta_task_progress.get("is_terminal_step", False)):
            return True
        if selected_route_family and (selected_correction_mode or selected_correction_target):
            return True
        route_focus = problem_context.get("route_focus")
        return isinstance(route_focus, dict) and bool(route_focus)

    def _call_orchestrator_model(self, request: OrchestratorRequest) -> dict[str, Any]:
        return self._call_chat_model(
            stage="orchestrator",
            model=self.planning_model,
            system_prompt=self._stage_system_prompt("orchestrator"),
            input_payload={"stage": "orchestrator", "request": _model_dump(request)},
            response_model=OrchestratorTaskPayload,
        )

    def _build_orchestrator_request(
        self,
        *,
        target_stage: str,
        request: ProposalRequest | ReflectionRequest,
        parent_node: Optional[NodeSnapshot],
        latest_critique: str,
        minimal: bool,
    ) -> OrchestratorRequest:
        return OrchestratorRequest(
            attempt_index=request.attempt_index,
            target_stage=target_stage,
            problem_context=self._summarize_problem_context_for_orchestrator(
                request.problem_context,
                minimal=minimal,
            ),
            current_node=self._summarize_node_snapshot_for_orchestrator(
                request.current_node,
                minimal=minimal,
            ),
            parent_node=(
                None
                if minimal or parent_node is None
                else self._summarize_node_snapshot_for_orchestrator(parent_node, minimal=True)
            ),
            latest_critique=self._truncate_orchestrator_text(
                latest_critique,
                self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
            ),
        )

    def _summarize_problem_context_for_orchestrator(
        self,
        problem_context: dict[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        meta_task = problem_context.get("meta_task")
        if isinstance(meta_task, dict) and meta_task:
            summary["meta_task"] = self._summarize_meta_task(
                meta_task,
                minimal=minimal,
                include_full_problem=False,
            )

        meta_task_progress = problem_context.get("meta_task_progress")
        if isinstance(meta_task_progress, dict) and meta_task_progress:
            summary["meta_task_progress"] = self._summarize_meta_task_progress(
                meta_task_progress,
                minimal=minimal,
            )

        existing_orchestrator_task = problem_context.get("orchestrator_task")
        if isinstance(existing_orchestrator_task, dict) and existing_orchestrator_task and not minimal:
            summary["previous_orchestrator_task"] = self._summarize_orchestrator_task(
                existing_orchestrator_task,
                minimal=True,
            )

        child_payload = problem_context.get("children")
        if isinstance(child_payload, list):
            summary["child_context_count"] = len(child_payload)

        present_stage_payloads = [
            key for key in ("proposal", "calculation", "evaluation", "reflection") if key in problem_context
        ]
        if present_stage_payloads:
            summary["stage_payloads_present"] = present_stage_payloads

        return summary

    def _summarize_meta_task(
        self,
        meta_task: dict[str, Any],
        *,
        minimal: bool,
        include_full_problem: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        meta_task_keys = ("objective", "first_step") if include_full_problem else ("first_step",)
        for key in meta_task_keys:
            value = self._truncate_orchestrator_text(
                meta_task.get(key, ""),
                self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
            )
            if value:
                summary[key] = value
        summary["step_ordering"] = self._compact_string_list_for_orchestrator(
            meta_task.get("step_ordering", []),
            short=minimal,
        )
        summary["completion_signals"] = self._compact_string_list_for_orchestrator(
            meta_task.get("completion_signals", []),
            short=minimal,
        )
        route_options = self._compact_structured_reasoning_list_for_orchestrator(
            meta_task.get("route_options", []),
            short=minimal,
        )
        if route_options:
            summary["route_options"] = route_options
        step_blueprints = self._compact_structured_reasoning_list_for_orchestrator(
            meta_task.get("step_blueprints", []),
            short=minimal,
        )
        if step_blueprints:
            summary["step_blueprints"] = step_blueprints
        return summary

    def _summarize_meta_task_progress(
        self,
        meta_task_progress: dict[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        summary = {
            "current_step_index": meta_task_progress.get("current_step_index", 0),
            "phase": meta_task_progress.get("phase", ""),
            "is_terminal_step": bool(meta_task_progress.get("is_terminal_step", False)),
            "total_steps": meta_task_progress.get("total_steps", 0),
        }
        for key in ("current_step", "current_step_guidance"):
            value = self._truncate_orchestrator_text(
                meta_task_progress.get(key, ""),
                self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
            )
            if value:
                summary[key] = value
        summary["previous_steps"] = self._compact_string_list_for_orchestrator(
            meta_task_progress.get("previous_steps", []),
            short=True,
        )
        summary["remaining_steps"] = self._compact_string_list_for_orchestrator(
            meta_task_progress.get("remaining_steps", []),
            short=minimal,
        )
        route_options = self._compact_structured_reasoning_list_for_orchestrator(
            meta_task_progress.get("route_options", []),
            short=True,
        )
        if route_options:
            summary["route_options"] = route_options
        selected_route_family = self._truncate_orchestrator_text(
            meta_task_progress.get("selected_route_family", ""),
            self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
        )
        if selected_route_family:
            summary["selected_route_family"] = selected_route_family
        selected_correction_mode = self._truncate_orchestrator_text(
            meta_task_progress.get("selected_correction_mode", ""),
            self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
        )
        if selected_correction_mode:
            summary["selected_correction_mode"] = selected_correction_mode
        selected_correction_target = self._truncate_orchestrator_text(
            meta_task_progress.get("selected_correction_target", ""),
            self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
        )
        if selected_correction_target:
            summary["selected_correction_target"] = selected_correction_target
        return summary

    def _summarize_orchestrator_task(
        self,
        orchestrator_task: dict[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for key in ("step_focus", "current_step_guidance", "selected_task"):
            value = self._truncate_orchestrator_text(
                orchestrator_task.get(key, ""),
                self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
            )
            if value:
                summary[key] = value
        summary["deferred_tasks"] = self._compact_string_list_for_orchestrator(
            orchestrator_task.get("deferred_tasks", []),
            short=minimal,
        )
        selected_route_family = self._truncate_orchestrator_text(
            orchestrator_task.get("selected_route_family", ""),
            self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
        )
        if selected_route_family:
            summary["selected_route_family"] = selected_route_family
        selected_candidate_task = _selected_candidate_task(orchestrator_task.get("candidate_tasks", []))
        selected_correction_mode = self._truncate_orchestrator_text(
            selected_candidate_task.get("correction_mode", ""),
            self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
        )
        if selected_correction_mode:
            summary["selected_correction_mode"] = selected_correction_mode
        selected_correction_target = self._truncate_orchestrator_text(
            selected_candidate_task.get("correction_target", ""),
            self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
        )
        if selected_correction_target:
            summary["selected_correction_target"] = selected_correction_target
        candidate_tasks = self._compact_structured_reasoning_list_for_orchestrator(
            orchestrator_task.get("candidate_tasks", []),
            short=minimal,
        )
        if candidate_tasks:
            summary["candidate_tasks"] = candidate_tasks
        return summary

    def _summarize_node_snapshot_for_orchestrator(
        self,
        node: NodeSnapshot,
        *,
        minimal: bool,
    ) -> NodeSnapshot:
        reflection_tail = node.reflection_history[-1:] if minimal else node.reflection_history[-2:]
        return NodeSnapshot(
            id=node.id,
            parent_id=node.parent_id,
            thought_step=self._truncate_orchestrator_text(
                node.thought_step,
                self.ORCHESTRATOR_SHORT_TEXT_LIMIT if minimal else self.ORCHESTRATOR_TEXT_LIMIT,
            ),
            equations=self._compact_string_list_for_orchestrator(node.equations, short=minimal),
            known_vars=self._summarize_known_vars_for_orchestrator(node.known_vars, minimal=minimal),
            used_models=self._compact_string_list_for_orchestrator(node.used_models, short=minimal),
            quantities=self._compact_mapping_for_orchestrator(
                node.quantities,
                limit=self.ORCHESTRATOR_MAP_LIMIT,
                short=minimal,
            ),
            boundary_conditions=self._compact_mapping_for_orchestrator(
                node.boundary_conditions,
                limit=self.ORCHESTRATOR_MAP_LIMIT,
                short=minimal,
            ),
            status=node.status,
            fsm_state=node.fsm_state,
            score=node.score,
            reflection_history=self._compact_string_list_for_orchestrator(reflection_tail, short=True),
        )

    def _summarize_known_vars_for_orchestrator(
        self,
        known_vars: dict[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        if not isinstance(known_vars, dict) or not known_vars:
            return {}

        summary: dict[str, Any] = {}
        for key in (
            "evaluation_passed",
            "needs_deeper_reasoning",
            "low_score_reason",
            "hard_rule_violations",
            "recoverable_rule_violations",
            "expansion_priority",
            "selected_for_frontier",
            "scheduler_action",
            "route_family",
            "correction_mode",
            "correction_target",
            "distributed_reasoning_slot",
        ):
            if key not in known_vars:
                continue
            value = known_vars[key]
            if key in {"hard_rule_violations", "recoverable_rule_violations"}:
                summary[key] = self._compact_string_list_for_orchestrator(value, short=True)
            else:
                summary[key] = self._compact_scalar_for_orchestrator(value, short=minimal)

        orchestrator_task = known_vars.get("orchestrator_task")
        if isinstance(orchestrator_task, dict) and orchestrator_task:
            summary["orchestrator_task"] = self._summarize_orchestrator_task(
                orchestrator_task,
                minimal=True,
            )

        summary["known_var_keys"] = self._compact_string_list_for_orchestrator(
            list(known_vars.keys()),
            short=minimal,
        )
        return summary

    def _build_fallback_orchestrator_task(
        self,
        *,
        problem_context: dict[str, Any],
        latest_critique: str,
        target_stage: str,
    ) -> dict[str, Any]:
        meta_task = problem_context.get("meta_task")
        meta_task_progress = problem_context.get("meta_task_progress")
        current_step = ""
        current_guidance = ""
        remaining_steps: list[str] = []
        completion_signals: list[str] = []
        selected_route_family = ""
        selected_correction_mode = ""
        selected_correction_target = ""
        if isinstance(meta_task_progress, dict):
            current_step = self._truncate_orchestrator_text(
                meta_task_progress.get("current_step", ""),
                self.ORCHESTRATOR_TEXT_LIMIT,
            )
            current_guidance = self._truncate_orchestrator_text(
                meta_task_progress.get("current_step_guidance", ""),
                self.ORCHESTRATOR_TEXT_LIMIT,
            )
            remaining_steps = self._compact_string_list_for_orchestrator(
                meta_task_progress.get("remaining_steps", []),
                short=True,
            )
            selected_route_family = self._truncate_orchestrator_text(
                meta_task_progress.get("selected_route_family", ""),
                self.ORCHESTRATOR_TEXT_LIMIT,
            )
            selected_correction_mode = self._truncate_orchestrator_text(
                meta_task_progress.get("selected_correction_mode", ""),
                self.ORCHESTRATOR_TEXT_LIMIT,
            )
            selected_correction_target = self._truncate_orchestrator_text(
                meta_task_progress.get("selected_correction_target", ""),
                self.ORCHESTRATOR_TEXT_LIMIT,
            )
        if isinstance(meta_task, dict):
            completion_signals = self._compact_string_list_for_orchestrator(
                meta_task.get("completion_signals", []),
                short=True,
            )

        selected_task = self._truncate_orchestrator_text(
            latest_critique or current_guidance or current_step or f"Advance only the current {target_stage} task.",
            self.ORCHESTRATOR_TEXT_LIMIT,
        )
        guidance = self._truncate_orchestrator_text(
            f"Execute only this task: {selected_task}",
            self.ORCHESTRATOR_TEXT_LIMIT,
        )
        task_breakdown = [selected_task]
        if current_step and current_step != selected_task:
            task_breakdown.insert(0, current_step)
        if not completion_signals:
            completion_signals = ["one strictly scoped task completed"]

        candidate_tasks = _derive_orchestrator_candidate_tasks(
            {
                "selected_task": selected_task,
                "task_breakdown": task_breakdown,
                "deferred_tasks": remaining_steps,
                "selected_route_family": selected_route_family,
            }
        )
        for item in candidate_tasks:
            if str(item.get("status", "")).strip().lower() != "selected":
                continue
            if selected_correction_mode and not item.get("correction_mode"):
                item["correction_mode"] = selected_correction_mode
            if selected_correction_target and not item.get("correction_target"):
                item["correction_target"] = selected_correction_target
            break

        return {
            "step_focus": current_step or target_stage,
            "current_step_guidance": guidance,
            "task_breakdown": task_breakdown,
            "selected_task": selected_task,
            "deferred_tasks": remaining_steps,
            "completion_signals": completion_signals,
            "selected_route_family": selected_route_family,
            "candidate_tasks": candidate_tasks,
        }

    @staticmethod
    def _is_context_overflow_error(exc: ChatBackendTransportError) -> bool:
        details = f"{exc} {exc.response_body}".lower()
        return "context length" in details or "tokens to keep" in details

    def _compact_string_list_for_meta_analysis(self, value: Any, *, short: bool) -> list[str]:
        limit = self.META_ANALYSIS_SHORT_LIST_LIMIT if short else self.META_ANALYSIS_LIST_LIMIT
        text_limit = self.META_ANALYSIS_SHORT_TEXT_LIMIT if short else self.META_ANALYSIS_TEXT_LIMIT
        items = _coerce_string_list(value)
        return [self._truncate_meta_analysis_text(item, text_limit) for item in items[:limit] if item.strip()]

    def _compact_mapping_for_meta_analysis(
        self,
        value: Any,
        *,
        limit: int,
        short: bool,
    ) -> dict[str, Any]:
        mapping = _coerce_mapping(value)
        compact: dict[str, Any] = {}
        for key, raw_value in list(mapping.items())[:limit]:
            compact[str(key)] = self._compact_scalar_for_meta_analysis(raw_value, short=short)
        return compact

    def _compact_scalar_for_meta_analysis(self, value: Any, *, short: bool) -> Any:
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return self._compact_mapping_for_meta_analysis(
                value,
                limit=self.META_ANALYSIS_SHORT_LIST_LIMIT if short else self.META_ANALYSIS_MAP_LIMIT,
                short=True,
            )
        if isinstance(value, list):
            return self._compact_string_list_for_meta_analysis(value, short=True)
        text_limit = self.META_ANALYSIS_SHORT_TEXT_LIMIT if short else self.META_ANALYSIS_TEXT_LIMIT
        return self._truncate_meta_analysis_text(value, text_limit)

    def _truncate_meta_analysis_text(self, value: Any, max_chars: int) -> str:
        text = str(value).strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    def _compact_string_list_for_orchestrator(self, value: Any, *, short: bool) -> list[str]:
        limit = self.ORCHESTRATOR_SHORT_LIST_LIMIT if short else self.ORCHESTRATOR_LIST_LIMIT
        text_limit = self.ORCHESTRATOR_SHORT_TEXT_LIMIT if short else self.ORCHESTRATOR_TEXT_LIMIT
        items = _coerce_string_list(value)
        return [self._truncate_orchestrator_text(item, text_limit) for item in items[:limit] if item.strip()]

    def _compact_structured_reasoning_list_for_orchestrator(
        self,
        value: Any,
        *,
        short: bool,
    ) -> list[dict[str, Any]]:
        limit = self.ORCHESTRATOR_SHORT_LIST_LIMIT if short else self.ORCHESTRATOR_LIST_LIMIT
        items = _coerce_structured_reasoning_list(value)
        compact: list[dict[str, Any]] = []
        for item in items[:limit]:
            compact_item: dict[str, Any] = {}
            for key, raw_value in item.items():
                if key == "priority" and isinstance(raw_value, (int, float)):
                    compact_item[key] = raw_value
                    continue
                if isinstance(raw_value, list):
                    compact_item[key] = self._compact_string_list_for_orchestrator(raw_value, short=True)
                    continue
                compact_item[key] = self._compact_scalar_for_orchestrator(raw_value, short=True)
            if compact_item:
                compact.append(compact_item)
        return compact

    def _compact_mapping_for_orchestrator(
        self,
        value: Any,
        *,
        limit: int,
        short: bool,
    ) -> dict[str, Any]:
        mapping = _coerce_mapping(value)
        compact: dict[str, Any] = {}
        for key, raw_value in list(mapping.items())[:limit]:
            compact[str(key)] = self._compact_scalar_for_orchestrator(raw_value, short=short)
        return compact

    def _compact_scalar_for_orchestrator(self, value: Any, *, short: bool) -> Any:
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return self._compact_mapping_for_orchestrator(
                value,
                limit=self.ORCHESTRATOR_SHORT_LIST_LIMIT if short else self.ORCHESTRATOR_MAP_LIMIT,
                short=True,
            )
        if isinstance(value, list):
            return self._compact_string_list_for_orchestrator(value, short=True)
        text_limit = self.ORCHESTRATOR_SHORT_TEXT_LIMIT if short else self.ORCHESTRATOR_TEXT_LIMIT
        return self._truncate_orchestrator_text(value, text_limit)

    def _truncate_orchestrator_text(self, value: Any, max_chars: int) -> str:
        text = str(value).strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    def _compact_string_list_for_reasoning(self, value: Any, *, short: bool) -> list[str]:
        limit = self.REASONING_SHORT_LIST_LIMIT if short else self.REASONING_LIST_LIMIT
        text_limit = self.REASONING_SHORT_TEXT_LIMIT if short else self.REASONING_TEXT_LIMIT
        items = _coerce_string_list(value)
        return [self._truncate_reasoning_text(item, text_limit) for item in items[:limit] if item.strip()]

    def _compact_structured_reasoning_list_for_reasoning(
        self,
        value: Any,
        *,
        short: bool,
    ) -> list[dict[str, Any]]:
        limit = self.REASONING_SHORT_LIST_LIMIT if short else self.REASONING_LIST_LIMIT
        items = _coerce_structured_reasoning_list(value)
        compact: list[dict[str, Any]] = []
        for item in items[:limit]:
            compact_item: dict[str, Any] = {}
            for key, raw_value in item.items():
                if key == "priority" and isinstance(raw_value, (int, float)):
                    compact_item[key] = raw_value
                    continue
                if isinstance(raw_value, list):
                    compact_item[key] = self._compact_string_list_for_reasoning(raw_value, short=True)
                    continue
                compact_item[key] = self._compact_scalar_for_reasoning(raw_value, short=True)
            if compact_item:
                compact.append(compact_item)
        return compact

    def _compact_mapping_for_reasoning(self, value: Any, *, short: bool) -> dict[str, Any]:
        mapping = _coerce_mapping(value)
        compact: dict[str, Any] = {}
        limit = self.REASONING_SHORT_LIST_LIMIT if short else self.REASONING_MAP_LIMIT
        for key, raw_value in list(mapping.items())[:limit]:
            compact[str(key)] = self._compact_scalar_for_reasoning(raw_value, short=short)
        return compact

    def _compact_scalar_for_reasoning(self, value: Any, *, short: bool) -> Any:
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return self._compact_mapping_for_reasoning(value, short=True)
        if isinstance(value, list):
            return self._compact_string_list_for_reasoning(value, short=True)
        text_limit = self.REASONING_SHORT_TEXT_LIMIT if short else self.REASONING_TEXT_LIMIT
        return self._truncate_reasoning_text(value, text_limit)

    def _truncate_reasoning_text(self, value: Any, max_chars: int) -> str:
        text = str(value).strip()
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."

    def _call_chat_model(
        self,
        *,
        stage: str,
        model: str,
        system_prompt: str,
        input_payload: dict[str, Any],
        response_model: type[BaseModel],
    ) -> dict[str, Any]:
        serialized_input = json.dumps(input_payload, ensure_ascii=False, sort_keys=True)
        attempted_models: list[str] = []
        last_transport_exc: Optional[ChatBackendTransportError] = None

        for candidate_model in self._stage_model_candidates(stage=stage, requested_model=model):
            attempted_models.append(candidate_model)
            try:
                raw_response = self.client.chat(
                    model=candidate_model,
                    system_prompt=system_prompt,
                    input_text=serialized_input,
                )
                return self._validate_or_repair_payload(
                    stage=stage,
                    model=candidate_model,
                    response_model=response_model,
                    raw_response=raw_response,
                )
            except ChatBackendTransportError as exc:
                if self._is_model_not_found_error(exc):
                    last_transport_exc = exc
                    continue
                if self._should_retry_next_model_after_transport_error(
                    stage=stage,
                    requested_model=model,
                    candidate_model=candidate_model,
                    exc=exc,
                ):
                    last_transport_exc = exc
                    continue
                raise ChatBackendTransportError(
                    f"{stage} request to model '{candidate_model}' failed: {exc}",
                    status_code=exc.status_code,
                    response_body=exc.response_body,
                ) from exc
            except (ChatBackendResponseError, TypeError, ValueError) as exc:
                if self._should_retry_next_model_after_payload_error(
                    stage=stage,
                    requested_model=model,
                    candidate_model=candidate_model,
                ):
                    continue
                raise ChatBackendResponseError(
                    f"Invalid {stage} payload from model '{candidate_model}': {exc}"
                ) from exc

        if last_transport_exc is not None:
            fallback_tail = [candidate for candidate in attempted_models[1:] if candidate != model]
            fallback_suffix = f" and fallbacks {fallback_tail}" if fallback_tail else ""
            raise ChatBackendTransportError(
                f"{stage} request to model '{model}' failed{fallback_suffix}: {last_transport_exc}",
                status_code=last_transport_exc.status_code,
                response_body=last_transport_exc.response_body,
            ) from last_transport_exc

        raise ChatBackendTransportError(f"{stage} request to model '{model}' failed before any model call completed.")

    def _should_retry_next_model_after_payload_error(
        self,
        *,
        stage: str,
        requested_model: str,
        candidate_model: str,
    ) -> bool:
        if stage != "evaluation":
            return False
        normalized_requested = str(requested_model).strip()
        normalized_candidate = str(candidate_model).strip()
        normalized_review = str(self.review_model).strip()
        return bool(
            normalized_requested
            and normalized_requested != normalized_review
            and normalized_candidate != normalized_review
        )

    def _should_retry_next_model_after_transport_error(
        self,
        *,
        stage: str,
        requested_model: str,
        candidate_model: str,
        exc: ChatBackendTransportError,
    ) -> bool:
        if exc.status_code not in (None, *TRANSIENT_HTTP_STATUS_CODES):
            return False
        normalized_requested = str(requested_model).strip()
        normalized_candidate = str(candidate_model).strip()
        normalized_review = str(self.review_model).strip()
        normalized_modeling = str(self.modeling_model).strip()
        return bool(
            stage in {"proposal", "reflection"}
            and normalized_requested
            and normalized_requested == normalized_review
            and normalized_requested != normalized_modeling
            and normalized_candidate == normalized_requested
        ) or bool(
            stage == "evaluation"
            and normalized_requested
            and normalized_requested != normalized_review
            and normalized_candidate != normalized_review
        )

    def _stage_model_candidates(self, *, stage: str, requested_model: str) -> list[str]:
        candidates: list[str] = []
        if stage == "evaluation":
            requested = str(requested_model).strip()
            review = str(self.review_model).strip()
            for candidate in (requested, review if requested != review else ""):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)
            return candidates

        for candidate in [requested_model, *STAGE_MODEL_FALLBACKS.get(stage, [])]:
            normalized = str(candidate).strip()
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return candidates

    @staticmethod
    def _is_model_not_found_error(exc: ChatBackendTransportError) -> bool:
        if exc.status_code != 404:
            return False
        details = f"{exc} {exc.response_body}".lower()
        return "invalid model identifier" in details or "model_not_found" in details

    @classmethod
    def _stage_output_contract(cls, stage: str) -> dict[str, Any]:
        del cls
        contract = _load_stage_prompt_contract(stage)
        required_keys = [str(item) for item in contract.get("required_keys", []) if str(item).strip()]
        optional_keys = [str(item) for item in contract.get("optional_keys", []) if str(item).strip()]
        allowed_keys = required_keys + [key for key in optional_keys if key not in required_keys]
        return {
            "required_keys": required_keys,
            "optional_keys": optional_keys,
            "list_fields": [
                key
                for key in allowed_keys
                if key
                in {
                    "equations",
                    "used_models",
                    "hard_rule_violations",
                    "givens",
                    "unknowns",
                    "minimal_subproblems",
                    "step_ordering",
                    "completion_signals",
                    "task_breakdown",
                    "deferred_tasks",
                    "route_options",
                    "step_blueprints",
                    "candidate_tasks",
                }
            ],
            "map_fields": [
                key for key in allowed_keys if key in {"known_vars", "quantities", "boundary_conditions"}
            ],
            "string_fields": [
                key
                for key in allowed_keys
                if key
                in {
                    "thought_step",
                    "reason",
                    "objective",
                    "first_step",
                    "step_focus",
                    "current_step_guidance",
                    "selected_task",
                    "risk_level",
                    "selected_route_family",
                }
            ],
            "numeric_fields": [
                key
                for key in allowed_keys
                if key in {"physical_consistency", "variable_grounding", "contextual_relevance", "simplicity_hint", "score"}
            ],
            "boolean_fields": [key for key in allowed_keys if key in {"approved"}],
        }

    @classmethod
    def _stage_string_value_budget(cls, stage: str) -> int:
        normalized_stage = str(stage).strip().lower()
        if normalized_stage == "meta-analysis":
            return 280
        if normalized_stage == "orchestrator":
            return 180
        if normalized_stage in {"proposal", "reflection", "evaluation"}:
            return 160
        return 120

    @classmethod
    def _stage_system_prompt(cls, stage: str) -> str:
        contract = _load_stage_prompt_contract(stage)
        prompt_fragment = str(contract.get("prompt_fragment", "Return only a JSON object. Do not use markdown."))
        output_contract = cls._stage_output_contract(stage)
        required_keys = output_contract["required_keys"]
        optional_keys = output_contract.get("optional_keys", [])
        if not required_keys:
            return prompt_fragment

        instructions = [
            "Output exactly one compact JSON object with no prose, no code fences, and no markdown.",
            f"Keep the top-level keys in this exact order: {', '.join(required_keys)}.",
            (
                f"You may append these optional top-level keys after the required keys when they help preserve structured route or task distribution context: {', '.join(optional_keys)}."
                if optional_keys
                else "Do not add any other top-level keys."
            ),
        ]
        string_value_budget = cls._stage_string_value_budget(stage)
        instructions.append(
            "Keep every prose value to a single short paragraph with no blank lines, bullets, or numbered lists."
        )
        instructions.append(
            f"Keep each prose string value under about {string_value_budget} characters and at most two short sentences; move extra detail into structured fields instead of longer paragraphs."
        )
        if output_contract["list_fields"]:
            instructions.append(
                f"Use [] for empty list fields: {', '.join(output_contract['list_fields'])}."
            )
            instructions.append(
                f"Keep each item in list fields short: one phrase or sentence only, never a paragraph."
            )
        if output_contract["map_fields"]:
            instructions.append(
                f"Use {{}} for empty object fields: {', '.join(output_contract['map_fields'])}."
            )
        if output_contract["string_fields"]:
            instructions.append(
                f"Use \"\" for empty string fields: {', '.join(output_contract['string_fields'])}."
            )
        if output_contract["numeric_fields"]:
            instructions.append(
                f"Use JSON numbers, not strings, for numeric fields: {', '.join(output_contract['numeric_fields'])}."
            )
        if output_contract["boolean_fields"]:
            instructions.append(
                f"Use JSON booleans, not strings, for boolean fields: {', '.join(output_contract['boolean_fields'])}."
            )
        return f"{prompt_fragment} {' '.join(instructions)}"

    def _validate_or_repair_payload(
        self,
        *,
        stage: str,
        model: str,
        response_model: type[BaseModel],
        raw_response: Any,
    ) -> dict[str, Any]:
        try:
            return self._validate_payload(response_model, raw_response)
        except (TypeError, ValueError) as initial_exc:
            repaired_response = self.client.chat(
                model=model,
                system_prompt=self._repair_system_prompt(stage),
                input_text=json.dumps(
                    {
                        "stage": "repair",
                        "target_stage": stage,
                        "required_keys": _load_stage_prompt_contract(stage).get("required_keys", []),
                        "raw_response": _serialize_raw_response_for_repair(raw_response),
                        "error": str(initial_exc),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            try:
                return self._validate_payload(response_model, repaired_response)
            except (TypeError, ValueError) as repair_exc:
                raise ValueError(f"{initial_exc}; repair attempt failed: {repair_exc}") from repair_exc

    @staticmethod
    def _validate_payload(response_model: type[BaseModel], raw_response: Any) -> dict[str, Any]:
        normalized = _normalize_chat_payload(raw_response)
        validated = _build_model(response_model, _coerce_model_payload(response_model, normalized))
        return _model_dump(validated)

    @staticmethod
    def _repair_system_prompt(stage: str) -> str:
        contract = _load_stage_prompt_contract(stage)
        required_keys = contract.get("required_keys", [])
        optional_keys = contract.get("optional_keys", [])
        keys_text = ", ".join(required_keys) if required_keys else "the required stage schema"
        optional_text = f" Optional keys may be preserved when present and valid: {', '.join(optional_keys)}." if optional_keys else ""
        single_step_instruction = (
            " Preserve exactly one minimal next step only."
            if contract.get("single_step")
            else ""
        )
        return (
            f"You are repairing a malformed ToT {stage} response. Return only one valid JSON object with exactly these top-level keys: {keys_text}. "
            "Remove extra keys, strip commentary, and preserve the original intent when it is clear from the raw response. "
            "If the raw response contains multiple JSON-like fragments, extract the single fragment that best matches the required keys. "
            "Do not use markdown or explanation."
            f"{optional_text}"
            f"{single_step_instruction}"
        )


class LocalChatDeletionReviewAdapter(NodeDeletionReviewAdapter):
    """Deletion-review adapter that calls the local chat API with the review model."""

    name = "local-chat-delete-review"

    def __init__(
        self,
        *,
        client: Optional[LocalChatAPIClient] = None,
        base_url: str = DEFAULT_CHAT_API_URL,
        timeout: float = 30.0,
        max_retries: int = 1,
        retry_backoff_seconds: float = 0.25,
        requester: Optional[ChatRequester] = None,
        review_model: str = DEFAULT_REVIEW_MODEL,
    ) -> None:
        self.client = client or LocalChatAPIClient(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            requester=requester,
        )
        self.review_model = review_model

    def review_delete_node(
        self,
        request: DeleteNodeReviewRequest,
    ) -> dict[str, Any]:
        try:
            raw_response = self.client.chat(
                model=self.review_model,
                system_prompt=LocalChatDualModelBackendAdapter._stage_system_prompt("delete-review"),
                input_text=json.dumps(
                    {"stage": "delete-review", "request": _model_dump(request)},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
            normalized = _normalize_chat_payload(raw_response)
            validated = _build_model(DeleteNodeReviewDecision, normalized)
            return _model_dump(validated)
        except ChatBackendTransportError as exc:
            raise ChatBackendTransportError(
                f"Delete-review request to model '{self.review_model}' failed: {exc}",
                status_code=exc.status_code,
                response_body=exc.response_body,
            ) from exc
        except (ChatBackendResponseError, TypeError, ValueError) as exc:
            raise ChatBackendResponseError(
                f"Invalid delete-review payload from model '{self.review_model}': {exc}"
            ) from exc


def build_local_chat_adapter_bundle(
    *,
    base_url: str = DEFAULT_CHAT_API_URL,
    timeout: float = 30.0,
    max_retries: int = 1,
    retry_backoff_seconds: float = 0.25,
    requester: Optional[ChatRequester] = None,
    planning_model: str = DEFAULT_PLANNING_MODEL,
    modeling_model: str = DEFAULT_MODELING_MODEL,
    review_model: str = DEFAULT_REVIEW_MODEL,
    non_terminal_evaluation_model: str = DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
) -> tuple[Callable[[dict[str, Any]], ReasoningBackendAdapter], NodeDeletionReviewAdapter]:
    """Build the planning, modeling, and review adapters for the local ``/api/v1/chat`` endpoint."""

    client = LocalChatAPIClient(
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        requester=requester,
    )

    def backend_adapter_factory(problem_context: dict[str, Any]) -> ReasoningBackendAdapter:
        del problem_context
        return LocalChatDualModelBackendAdapter(
            client=client,
            planning_model=planning_model,
            modeling_model=modeling_model,
            review_model=review_model,
            non_terminal_evaluation_model=non_terminal_evaluation_model,
        )

    deletion_review_adapter = LocalChatDeletionReviewAdapter(
        client=client,
        review_model=review_model,
    )
    return backend_adapter_factory, deletion_review_adapter


__all__ = [
    "DEFAULT_CHAT_API_URL",
    "DEFAULT_MODELING_MODEL",
    "DEFAULT_PLANNING_MODEL",
    "DEFAULT_REVIEW_MODEL",
    "ChatBackendError",
    "ChatBackendResponseError",
    "ChatBackendTransportError",
    "DeleteNodeReviewDecision",
    "DeleteNodeReviewRequest",
    "DeterministicContextBackendAdapter",
    "EvaluationRequest",
    "LocalChatAPIClient",
    "LocalChatDeletionReviewAdapter",
    "LocalChatDualModelBackendAdapter",
    "NodeDeletionReviewAdapter",
    "ProposalRequest",
    "ReasoningBackendAdapter",
    "ReflectionRequest",
    "build_local_chat_adapter_bundle",
]