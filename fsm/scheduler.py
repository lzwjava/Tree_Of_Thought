"""Tree-level scheduler built on top of the single-node FSM."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .backend import (
    DeleteNodeReviewDecision,
    DeleteNodeReviewRequest,
    NodeDeletionReviewAdapter,
    ReasoningBackendAdapter,
)
from .builder import NodeBuilderFSM
from .models import NodeSnapshot, NodeStatus, ToTNode
from .utils import _deserialize_blob, _model_field_names, _serialize_blob, _stable_hash


class TreeSchedulerState(BaseModel):
    """Persistent scheduler snapshot.

    The state payload is stored as a trusted local pickle-backed blob so that
    SymPy expressions and recursive Pydantic node trees survive round-trips.
    """

    version: int = 1
    snapshot_blob: str


class ToTTreeScheduler:
    """Tree-level harness that expands and ranks sibling nodes.

    The scheduler reuses ``NodeBuilderFSM`` for each node and adds the missing
    tree-level mechanics on top:

    - sibling ranking
    - frontier management with BFS-like layer priority
    - global expansion-budget control
    - diversity control
    - state deduplication and loop suppression
    - duplicate-branch merge
    - persistence and resume support

    Child candidates are provided deterministically through the ``children`` key
    in each node problem context. Each child context may itself contain nested
    ``children`` entries for deeper expansion.
    """

    META_TASK_STRATEGY_SCAN_GUIDANCE = (
        "Analyze the next-step strategy space at planning level. "
        "Make one route-local planning claim only, keep all other routes deferred, and do not solve the final answer yet."
    )

    def __init__(
        self,
        root_problem_context: dict[str, Any],
        *,
        max_reflections: int = 2,
        expansion_budget: int = 8,
        max_frontier_size: int = 8,
        max_children_per_expansion: int = 4,
        max_frontier_per_diversity_key: int = 2,
        children_key: str = "children",
        backend_adapter_factory: Optional[Callable[[dict[str, Any]], ReasoningBackendAdapter]] = None,
        deletion_review_adapter: Optional[NodeDeletionReviewAdapter] = None,
    ) -> None:
        if expansion_budget < 0:
            raise ValueError("expansion_budget must be non-negative.")
        if max_frontier_size < 1:
            raise ValueError("max_frontier_size must be at least 1.")
        if max_children_per_expansion < 1:
            raise ValueError("max_children_per_expansion must be at least 1.")
        if max_frontier_per_diversity_key < 1:
            raise ValueError("max_frontier_per_diversity_key must be at least 1.")

        self.root_problem_context = root_problem_context
        self.max_reflections = max_reflections
        self.expansion_budget = expansion_budget
        self.max_frontier_size = max_frontier_size
        self.max_children_per_expansion = max_children_per_expansion
        self.max_frontier_per_diversity_key = max_frontier_per_diversity_key
        self.children_key = children_key
        self.backend_adapter_factory = backend_adapter_factory
        self.deletion_review_adapter = deletion_review_adapter

        self.root_node: Optional[ToTNode] = None
        self._frontier: list[dict[str, Any]] = []
        self._expansion_log: list[dict[str, Any]] = []
        self._expanded_node_ids: list[str] = []
        self._node_index: dict[str, ToTNode] = {}
        self._signature_registry: dict[str, str] = {}
        self._problem_context_prepared = False
        self.target_expansion_budget = expansion_budget
        self.run_status = "idle"
        self.run_phase = "created"
        self.last_error = ""
        self.auto_run_requested = False

    def _set_run_state(
        self,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        if status is not None:
            self.run_status = str(status)
        if phase is not None:
            self.run_phase = str(phase)
        if last_error is not None:
            self.last_error = str(last_error)

    def _prepare_root_problem_context_if_needed(self) -> None:
        if self._problem_context_prepared:
            return
        if self.backend_adapter_factory is None:
            self._problem_context_prepared = True
            return

        self._set_run_state(status="busy", phase="preparing-meta-task", last_error="")
        backend_adapter = self.backend_adapter_factory(self.root_problem_context)
        self.root_problem_context = backend_adapter.prepare_problem_context(deepcopy(self.root_problem_context))
        self._problem_context_prepared = True

    def _final_run_phase(self) -> str:
        if self.run_status == "error":
            return "error"
        if self.root_node is None:
            return "created"
        if not self._frontier:
            return "frontier-empty"
        if len(self._expanded_node_ids) >= self.expansion_budget:
            return "awaiting-next-step"
        return "idle"

    def run(self) -> dict[str, Any]:
        """Build the root node and expand the tree under scheduler constraints."""
        self._set_run_state(status="busy", phase="preparing-meta-task", last_error="")

        try:
            self._prepare_root_problem_context_if_needed()

            if self.root_node is None:
                self._set_run_state(status="busy", phase="building-root")
                root_node = self._build_node(parent_node=None, problem_context=self.root_problem_context)
                self.root_node = root_node
                self._initialize_root_node(root_node)
                if self._is_expandable(root_node, self.root_problem_context):
                    self._frontier.append(
                        {
                            "node": root_node,
                            "problem_context": self.root_problem_context,
                            "depth": 0,
                        }
                    )
                    retained_ids = self._rebalance_frontier()
                    root_node.known_vars["selected_for_frontier"] = root_node.id in retained_ids

            while self._frontier and len(self._expanded_node_ids) < self.expansion_budget:
                self._set_run_state(status="busy", phase="expanding-frontier")
                current = self._frontier.pop(0)
                parent_node = current["node"]
                problem_context = current["problem_context"]
                depth = current["depth"]

                child_contexts = self._extract_child_contexts(problem_context)
                if not child_contexts:
                    continue

                self._expanded_node_ids.append(parent_node.id)
                built_children: list[tuple[ToTNode, dict[str, Any]]] = []
                for child_context in child_contexts:
                    child_node = self._build_node(parent_node=parent_node, problem_context=child_context)
                    built_children.append((child_node, child_context))

                ranked_children = sorted(
                    built_children,
                    key=lambda item: self._node_ranking_key(item[0]),
                )

                expandable_candidates: list[tuple[ToTNode, dict[str, Any]]] = []
                sibling_ranking: list[dict[str, Any]] = []

                for rank, (child_node, child_context) in enumerate(ranked_children, start=1):
                    scheduler_action = self._apply_scheduler_controls(child_node, parent_node)
                    priority = self._node_priority(child_node)
                    expandable = scheduler_action is None and self._is_expandable(child_node, child_context)
                    child_node.known_vars["sibling_rank"] = rank
                    child_node.known_vars["sibling_priority"] = priority
                    child_node.known_vars["scheduler_action"] = scheduler_action or "expanded"
                    child_node.known_vars["expandable"] = expandable
                    child_node.known_vars["selected_for_frontier"] = False
                    sibling_ranking.append(
                        {
                            "node_id": child_node.id,
                            "rank": rank,
                            "status": child_node.status.value,
                            "priority": priority,
                            "score": child_node.score,
                            "expandable": expandable,
                            "scheduler_action": scheduler_action or "expanded",
                            "diversity_key": child_node.known_vars.get("diversity_key"),
                        }
                    )
                    if expandable:
                        expandable_candidates.append((child_node, child_context))

                parent_node.known_vars["sibling_ranking"] = sibling_ranking

                frontier_candidate_budget = self._frontier_candidate_budget(problem_context)
                frontier_candidates = expandable_candidates[:frontier_candidate_budget]
                for child_node, child_context in frontier_candidates:
                    self._frontier.append(
                        {
                            "node": child_node,
                            "problem_context": child_context,
                            "depth": depth + 1,
                        }
                    )

                retained_ids = self._rebalance_frontier()
                for child_node, _ in ranked_children:
                    child_node.known_vars["selected_for_frontier"] = child_node.id in retained_ids

                parent_node.known_vars["selected_child_ids"] = [
                    child_node.id
                    for child_node, _ in ranked_children
                    if child_node.id in retained_ids
                ]
                self._expansion_log.append(
                    {
                        "parent_id": parent_node.id,
                        "depth": depth,
                        "expanded": True,
                        "child_ids": [child.id for child, _ in ranked_children],
                        "frontier_candidate_ids": [
                            child.id
                            for child, _ in frontier_candidates
                        ],
                        "retained_frontier_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.id in retained_ids
                        ],
                        "pruned_child_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.status != NodeStatus.ACTIVE
                        ],
                        "duplicate_child_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.known_vars.get("scheduler_action") == "merged-duplicate"
                        ],
                        "loop_suppressed_child_ids": [
                            child.id
                            for child, _ in ranked_children
                            if child.known_vars.get("scheduler_action") == "suppressed-loop"
                        ],
                        "budget_remaining": self.expansion_budget - len(self._expanded_node_ids),
                        "frontier_size_after": len(self._frontier),
                    }
                )
        except Exception as exc:
            self._set_run_state(status="error", phase="error", last_error=str(exc))
            raise

        self._set_run_state(status="ready", phase=self._final_run_phase(), last_error="")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return the current scheduler state without mutating the tree."""

        meta_task = self.root_problem_context.get("meta_task", {})
        return {
            "root": self.root_node,
            "meta_task": dict(meta_task) if isinstance(meta_task, dict) else {},
            "frontier": self._frontier_snapshot(),
            "expansion_log": list(self._expansion_log),
            "expanded_node_ids": list(self._expanded_node_ids),
            "expansions_used": len(self._expanded_node_ids),
            "expansion_budget": self.expansion_budget,
            "target_expansion_budget": self.target_expansion_budget,
            "remaining_budget": self.expansion_budget - len(self._expanded_node_ids),
            "run_state": {
                "status": self.run_status,
                "phase": self.run_phase,
                "problem_context_prepared": self._problem_context_prepared,
                "auto_run_requested": self.auto_run_requested,
                "target_expansion_budget": self.target_expansion_budget,
                "last_error": self.last_error,
            },
        }

    def delete_node(
        self,
        node_id: str,
        *,
        reason: str,
        requested_by: str = "frontend",
        review_adapter: Optional[NodeDeletionReviewAdapter] = None,
    ) -> dict[str, Any]:
        """Delete a node subtree only after an AI review approves the operation."""

        if self.root_node is None:
            raise RuntimeError("Cannot delete a node before the tree has been built.")

        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("Deletion reason must be provided for AI review.")

        target_node = self._node_index.get(node_id)
        if target_node is None:
            raise KeyError(f"Unknown node id: {node_id}")
        if target_node.parent_id is None:
            raise ValueError("Deleting the root node is not supported.")

        effective_review_adapter = review_adapter or self.deletion_review_adapter
        if effective_review_adapter is None:
            raise ValueError("AI deletion review adapter is required before deleting a node.")

        parent_node = self._node_index.get(target_node.parent_id)
        if parent_node is None:
            raise RuntimeError("Failed to resolve the parent node for deletion.")

        review_request = DeleteNodeReviewRequest(
            requested_by=requested_by,
            reason=normalized_reason,
            current_root_id=self.root_node.id,
            current_frontier_size=len(self._frontier),
            target_node=self._node_snapshot(target_node),
            parent_node=self._node_snapshot(parent_node),
            descendant_count=max(0, len(self._collect_subtree_nodes(target_node)) - 1),
            is_frontier_node=any(entry["node"].id == node_id for entry in self._frontier),
            is_expanded_node=node_id in self._expanded_node_ids,
        )
        review = self._build_model(
            DeleteNodeReviewDecision,
            self._normalize_review_payload(
                effective_review_adapter.review_delete_node(review_request)
            ),
        )

        if not review.approved:
            return {
                "deleted": False,
                "node_id": node_id,
                "deleted_node_ids": [],
                "review": self._model_dump(review),
                "frontier": self._frontier_snapshot(),
            }

        deleted_nodes = self._collect_subtree_nodes(target_node)
        deleted_node_ids = [node.id for node in deleted_nodes]
        parent_node.children = [child for child in parent_node.children if child.id != node_id]

        self._frontier = [
            entry for entry in self._frontier if entry["node"].id not in deleted_node_ids
        ]
        self._expanded_node_ids = [
            expanded_id for expanded_id in self._expanded_node_ids if expanded_id not in deleted_node_ids
        ]

        self._resync_runtime_state()
        self._expansion_log.append(
            {
                "event": "delete-node",
                "requested_by": requested_by,
                "node_id": node_id,
                "deleted_node_ids": deleted_node_ids,
                "review": self._model_dump(review),
                "frontier_size_after": len(self._frontier),
            }
        )

        return {
            "deleted": True,
            "node_id": node_id,
            "deleted_node_ids": deleted_node_ids,
            "review": self._model_dump(review),
            "frontier": self._frontier_snapshot(),
        }

    def _build_node(self, parent_node: Optional[ToTNode], problem_context: dict[str, Any]) -> ToTNode:
        fsm = NodeBuilderFSM(
            parent_node=parent_node,
            problem_context=problem_context,
            max_reflections=self.max_reflections,
            backend_adapter=self._make_backend_adapter(problem_context),
        )
        node = fsm.run()
        route_focus = problem_context.get("route_focus")
        if isinstance(route_focus, dict):
            route_family = str(route_focus.get("route_family") or route_focus.get("label") or "").strip()
            if route_family and not str(node.known_vars.get("route_family", "")).strip():
                node.known_vars["route_family"] = route_family
            for key in ("correction_mode", "correction_target"):
                value = str(route_focus.get(key, "")).strip()
                if value and not str(node.known_vars.get(key, "")).strip():
                    node.known_vars[key] = value
            slot = str(route_focus.get("slot", "")).strip()
            if slot and not str(node.known_vars.get("distributed_reasoning_slot", "")).strip():
                node.known_vars["distributed_reasoning_slot"] = slot
        meta_task_progress = problem_context.get("meta_task_progress")
        if isinstance(meta_task_progress, dict):
            selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
            if selected_route_family and not str(node.known_vars.get("route_family", "")).strip():
                node.known_vars["route_family"] = selected_route_family
            for progress_key, known_var_key in (
                ("selected_correction_mode", "correction_mode"),
                ("selected_correction_target", "correction_target"),
            ):
                value = str(meta_task_progress.get(progress_key, "")).strip()
                if value and not str(node.known_vars.get(known_var_key, "")).strip():
                    node.known_vars[known_var_key] = value
        self._node_index[node.id] = node
        self._append_node_events(node)
        return node

    def _append_node_events(self, node: ToTNode) -> None:
        raw_events = node.known_vars.get("node_event_log")
        if not isinstance(raw_events, list):
            return

        for item in raw_events:
            if not isinstance(item, dict):
                continue
            entry = dict(item)
            entry.setdefault("node_id", node.id)
            entry.setdefault("parent_id", node.parent_id)
            self._expansion_log.append(entry)

    def _make_backend_adapter(self, problem_context: dict[str, Any]) -> Optional[ReasoningBackendAdapter]:
        if self.backend_adapter_factory is None:
            return None
        return self.backend_adapter_factory(problem_context)

    def _initialize_root_node(self, root_node: ToTNode) -> None:
        signature = self._compute_state_signature(root_node)
        root_node.known_vars["state_signature"] = signature
        root_node.known_vars["diversity_key"] = self._compute_diversity_key(root_node)
        root_node.known_vars.setdefault("scheduler_action", "root")
        if root_node.status == NodeStatus.ACTIVE:
            self._signature_registry[signature] = root_node.id

    def _extract_child_contexts(self, problem_context: dict[str, Any]) -> list[dict[str, Any]]:
        payload = problem_context.get(self.children_key, [])
        if not isinstance(payload, list):
            payload = []
        explicit_children = [dict(item) for item in payload if isinstance(item, dict)]
        if explicit_children:
            return explicit_children
        return self._synthesize_meta_task_child_contexts(problem_context)

    def _synthesize_meta_task_child_contexts(self, problem_context: dict[str, Any]) -> list[dict[str, Any]]:
        if any(key in problem_context for key in ("proposal", "calculation", "evaluation", "reflection")):
            return []

        problem_statement = str(problem_context.get("problem_statement", "")).strip()
        meta_task = problem_context.get("meta_task")
        meta_task_progress = problem_context.get("meta_task_progress")
        if not problem_statement:
            return []
        if not isinstance(meta_task, dict) or not meta_task:
            return []
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return []

        step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
        if not step_ordering:
            return []

        try:
            current_step_index = int(meta_task_progress.get("current_step_index", 0))
        except (TypeError, ValueError):
            current_step_index = 0
        current_step_index = max(0, min(current_step_index, len(step_ordering) - 1))
        if current_step_index >= len(step_ordering) - 1:
            return []

        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        route_focus = problem_context.get("route_focus")
        if not selected_route_family and isinstance(route_focus, dict):
            selected_route_family = str(
                route_focus.get("route_family") or route_focus.get("label") or ""
            ).strip()

        raw_route_options = meta_task.get("route_options", [])
        if current_step_index == 0 and not selected_route_family and isinstance(raw_route_options, list):
            route_options = [dict(item) for item in raw_route_options if isinstance(item, dict) and item]
            if route_options:
                child_contexts: list[dict[str, Any]] = []
                route_surface_budget = self._route_surface_budget(problem_context)
                for index, route_option in enumerate(route_options[:route_surface_budget]):
                    child_context = deepcopy(problem_context)
                    child_context.pop(self.children_key, None)
                    child_context["meta_task"] = dict(meta_task)
                    child_context["meta_task_progress"] = self._build_meta_task_progress(
                        meta_task,
                        step_index=current_step_index,
                    )
                    route_family = str(route_option.get("route_family") or route_option.get("label") or "").strip()
                    route_label = str(route_option.get("label") or route_family or f"route {index + 1}").strip()
                    route_guidance = str(route_option.get("guidance", "")).strip()
                    route_focus = dict(route_option)
                    route_focus.setdefault("label", route_label)
                    route_focus.setdefault("route_family", route_family or route_label)
                    route_focus.setdefault("slot", str(index))
                    child_context["route_focus"] = route_focus
                    child_progress = dict(child_context["meta_task_progress"])
                    child_progress["selected_route_family"] = route_focus["route_family"]
                    child_progress["distributed_reasoning_slot"] = str(index)
                    child_progress["current_step"] = f"route-local scan: {route_focus['route_family']}"
                    correction_mode = str(route_focus.get("correction_mode", "")).strip()
                    if correction_mode:
                        child_progress["selected_correction_mode"] = correction_mode
                    correction_target = str(route_focus.get("correction_target", "")).strip()
                    if correction_target:
                        child_progress["selected_correction_target"] = correction_target
                    refined_guidance = self._build_route_strategy_scan_guidance(route_focus)
                    if route_guidance:
                        refined_guidance = f"{refined_guidance} Route-specific focus: {route_guidance}".strip()
                    correction_guidance_parts: list[str] = []
                    if correction_mode:
                        correction_guidance_parts.append(
                            f"use the {correction_mode} correction framing"
                        )
                    if correction_target:
                        correction_guidance_parts.append(
                            f"treat {correction_target} as the active correction quantity"
                        )
                    if correction_guidance_parts:
                        refined_guidance = (
                            f"{refined_guidance} Correction-specific focus: {'; '.join(correction_guidance_parts)}."
                        ).strip()
                    child_progress["current_step_guidance"] = refined_guidance
                    child_context["meta_task_progress"] = child_progress
                    child_context["auto_generated_child"] = True
                    child_contexts.append(child_context)
                if child_contexts:
                    return child_contexts

        child_context = deepcopy(problem_context)
        child_context.pop(self.children_key, None)
        child_context["meta_task"] = dict(meta_task)
        child_context["meta_task_progress"] = self._build_meta_task_progress(
            meta_task,
            step_index=current_step_index + 1,
        )
        inherited_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        if inherited_route_family:
            child_context["meta_task_progress"]["selected_route_family"] = inherited_route_family
        inherited_slot = str(meta_task_progress.get("distributed_reasoning_slot", "")).strip()
        if inherited_slot:
            child_context["meta_task_progress"]["distributed_reasoning_slot"] = inherited_slot
        inherited_correction_mode = str(meta_task_progress.get("selected_correction_mode", "")).strip()
        if inherited_correction_mode:
            child_context["meta_task_progress"]["selected_correction_mode"] = inherited_correction_mode
        inherited_correction_target = str(meta_task_progress.get("selected_correction_target", "")).strip()
        if inherited_correction_target:
            child_context["meta_task_progress"]["selected_correction_target"] = inherited_correction_target
        child_context["auto_generated_child"] = True
        return [child_context]

    def _build_route_strategy_scan_guidance(self, route_focus: dict[str, Any]) -> str:
        route_family = str(route_focus.get("route_family") or route_focus.get("label") or "this").strip()
        return (
            f"Stay at planning level and look only at the {route_family} route. "
            "Make one tiny route-local step only, keep every other route deferred, and do not solve the final answer yet."
        )

    def _build_meta_task_progress(self, meta_task: dict[str, Any], *, step_index: int) -> dict[str, Any]:
        step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
        first_step = str(meta_task.get("first_step", "")).strip()
        if step_ordering:
            normalized_index = max(0, min(step_index, len(step_ordering) - 1))
            current_step = step_ordering[normalized_index]
            previous_steps = step_ordering[:normalized_index]
            remaining_steps = step_ordering[normalized_index + 1 :]
        else:
            normalized_index = max(0, step_index)
            current_step = first_step
            previous_steps = []
            remaining_steps = []

        phase = "strategy_scan" if normalized_index == 0 else "incremental_refinement"
        if phase == "strategy_scan":
            current_step_guidance = self.META_TASK_STRATEGY_SCAN_GUIDANCE
        else:
            current_step_guidance = (
                f"Refine only the current subproblem: {current_step}. "
                "Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pending fixes deferred."
            )

        return {
            "current_step_index": normalized_index,
            "current_step": current_step,
            "current_step_guidance": current_step_guidance,
            "previous_steps": previous_steps,
            "remaining_steps": remaining_steps,
            "total_steps": len(step_ordering),
            "phase": phase,
            "is_terminal_step": bool(step_ordering) and normalized_index >= len(step_ordering) - 1,
            "route_options": [dict(item) for item in meta_task.get("route_options", []) if isinstance(item, dict)],
            "step_blueprints": [dict(item) for item in meta_task.get("step_blueprints", []) if isinstance(item, dict)],
        }

    def _apply_scheduler_controls(self, node: ToTNode, parent_node: ToTNode) -> Optional[str]:
        signature = self._compute_state_signature(node)
        node.known_vars["state_signature"] = signature
        node.known_vars["diversity_key"] = self._compute_diversity_key(node)

        ancestor_signature_map = self._ancestor_signature_map(parent_node)
        if signature in ancestor_signature_map:
            node.known_vars["merged_into_node_id"] = ancestor_signature_map[signature]
            node.known_vars["suppressed_by_scheduler"] = "loop"
            return "suppressed-loop"

        if node.status != NodeStatus.ACTIVE:
            return None

        canonical_id = self._signature_registry.get(signature)
        if canonical_id is not None and canonical_id != node.id:
            canonical_node = self._node_index.get(canonical_id)
            node.known_vars["merged_into_node_id"] = canonical_id
            node.known_vars["suppressed_by_scheduler"] = "duplicate"
            if canonical_node is not None:
                canonical_node.known_vars.setdefault("merged_duplicate_node_ids", []).append(node.id)
                canonical_node.known_vars.setdefault("merged_duplicate_parent_ids", []).append(parent_node.id)
                canonical_node.known_vars["merged_duplicate_count"] = len(
                    canonical_node.known_vars.get("merged_duplicate_node_ids", [])
                )
                canonical_node.score = max(canonical_node.score, node.score)
                canonical_priority = self._node_priority(canonical_node)
                duplicate_priority = self._node_priority(node)
                canonical_node.known_vars["expansion_priority"] = max(
                    canonical_priority,
                    duplicate_priority,
                )
            return "merged-duplicate"

        self._signature_registry[signature] = node.id
        return None

    def _compute_state_signature(self, node: ToTNode) -> str:
        signature_payload = {
            "equations": sorted(str(item) for item in node.equations),
            "used_models": sorted(str(item) for item in node.used_models),
            "quantities": {str(key): value for key, value in node.quantities.items()},
            "boundary_conditions": {
                str(key): value for key, value in node.boundary_conditions.items()
            },
        }
        route_family = str(node.known_vars.get("route_family", "")).strip()
        correction_mode = str(node.known_vars.get("correction_mode", "")).strip()
        correction_target = str(node.known_vars.get("correction_target", "")).strip()
        distributed_reasoning_slot = str(node.known_vars.get("distributed_reasoning_slot", "")).strip()
        if route_family or correction_mode or correction_target or distributed_reasoning_slot:
            signature_payload["route_family"] = route_family
            signature_payload["correction_mode"] = correction_mode
            signature_payload["correction_target"] = correction_target
            signature_payload["distributed_reasoning_slot"] = distributed_reasoning_slot
        return _stable_hash(signature_payload)

    def _compute_diversity_key(self, node: ToTNode) -> str:
        route_family = str(node.known_vars.get("route_family", "")).strip()
        correction_mode = str(node.known_vars.get("correction_mode", "")).strip()
        correction_target = str(node.known_vars.get("correction_target", "")).strip()
        distributed_reasoning_slot = str(node.known_vars.get("distributed_reasoning_slot", "")).strip()
        boundary_axes = []
        for key in node.boundary_conditions:
            text = str(key)
            boundary_axes.append(text.split("=", 1)[0].strip() if "=" in text else text)
        used_models = sorted(str(item) for item in node.used_models)
        if route_family or correction_mode or correction_target:
            diversity_payload = {
                "route_family": route_family,
                "correction_mode": correction_mode,
                "correction_target": correction_target,
                "slot": distributed_reasoning_slot,
                "used_models": used_models,
            }
        elif used_models or boundary_axes:
            diversity_payload = {
                "used_models": used_models,
                "boundary_axes": sorted(boundary_axes),
            }
        else:
            diversity_payload = {
                "equation_heads": [str(item)[:80] for item in node.equations[:2]],
            }
        return _stable_hash(diversity_payload)[:16]

    def _ancestor_signature_map(self, node: ToTNode) -> dict[str, str]:
        out: dict[str, str] = {}
        current = node
        while current is not None:
            signature = current.known_vars.get("state_signature")
            if signature is not None:
                out[str(signature)] = current.id
            if current.parent_id is None:
                break
            current = self._node_index.get(current.parent_id)
        return out

    def _is_expandable(self, node: ToTNode, problem_context: dict[str, Any]) -> bool:
        return node.status == NodeStatus.ACTIVE and bool(self._extract_child_contexts(problem_context))

    def _node_priority(self, node: ToTNode) -> float:
        if node.status != NodeStatus.ACTIVE:
            return 0.0
        raw_priority = node.known_vars.get("expansion_priority", 1.0)
        try:
            return round(float(raw_priority), 4)
        except (TypeError, ValueError):
            return 0.0

    def _node_ranking_key(self, node: ToTNode) -> tuple[Any, ...]:
        return (
            0 if node.status == NodeStatus.ACTIVE else 1,
            -self._node_priority(node),
            -float(node.score),
            len(node.reflection_history),
            len(node.equations),
            node.id,
        )

    def _frontier_entry_key(self, entry: dict[str, Any]) -> tuple[Any, ...]:
        return (int(entry["depth"]), *self._node_ranking_key(entry["node"]))

    def _route_surface_budget(self, problem_context: Optional[dict[str, Any]] = None) -> int:
        remaining_expansion_capacity = max(0, self.expansion_budget - len(self._expanded_node_ids))
        if remaining_expansion_capacity <= 0:
            return 0
        if self._is_root_strategy_scan_route_surface(problem_context):
            return max(1, self.max_frontier_size)
        return max(
            1,
            min(
                self.max_children_per_expansion,
                self.max_frontier_size,
                remaining_expansion_capacity,
            ),
        )

    def _frontier_candidate_budget(self, problem_context: Optional[dict[str, Any]] = None) -> int:
        if self._is_root_strategy_scan_route_surface(problem_context):
            return max(1, self.max_frontier_size)
        # Cap only the sibling slice for this expansion. Frontier capacity is
        # enforced later in _rebalance_frontier so lower-scored but distinct
        # route families or correction modes can still compete for retention.
        return max(1, self.max_children_per_expansion)

    def _is_root_strategy_scan_route_surface(self, problem_context: Optional[dict[str, Any]]) -> bool:
        if not isinstance(problem_context, dict) or not problem_context:
            return False
        meta_task = problem_context.get("meta_task")
        meta_task_progress = problem_context.get("meta_task_progress")
        if not isinstance(meta_task, dict) or not isinstance(meta_task_progress, dict):
            return False
        raw_route_options = meta_task.get("route_options", [])
        if not isinstance(raw_route_options, list) or not any(isinstance(item, dict) and item for item in raw_route_options):
            return False
        try:
            current_step_index = int(meta_task_progress.get("current_step_index", 0))
        except (TypeError, ValueError):
            current_step_index = 0
        if current_step_index != 0:
            return False
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        if selected_route_family:
            return False
        route_focus = problem_context.get("route_focus")
        if isinstance(route_focus, dict) and any(str(route_focus.get(key, "")).strip() for key in ("route_family", "label")):
            return False
        return True

    def _try_retain_frontier_entry(
        self,
        entry: dict[str, Any],
        retained: list[dict[str, Any]],
        diversity_counts: dict[str, int],
    ) -> bool:
        diversity_key = str(entry["node"].known_vars.get("diversity_key", entry["node"].id))
        if diversity_counts.get(diversity_key, 0) >= self.max_frontier_per_diversity_key:
            entry["node"].known_vars["selected_for_frontier"] = False
            entry["node"].known_vars["suppressed_by_scheduler"] = "diversity-cap"
            return False
        if len(retained) >= self.max_frontier_size:
            entry["node"].known_vars["selected_for_frontier"] = False
            return False
        retained.append(entry)
        diversity_counts[diversity_key] = diversity_counts.get(diversity_key, 0) + 1
        entry["node"].known_vars["suppressed_by_scheduler"] = ""
        return True

    def _rebalance_frontier(self) -> set[str]:
        self._frontier.sort(key=self._frontier_entry_key)
        retained: list[dict[str, Any]] = []
        diversity_counts: dict[str, int] = {}
        deferred_entries: list[dict[str, Any]] = []

        for entry in self._frontier:
            diversity_key = str(entry["node"].known_vars.get("diversity_key", entry["node"].id))
            if diversity_counts.get(diversity_key, 0) == 0 and len(retained) < self.max_frontier_size:
                self._try_retain_frontier_entry(entry, retained, diversity_counts)
                continue
            deferred_entries.append(entry)

        for entry in deferred_entries:
            if len(retained) >= self.max_frontier_size:
                entry["node"].known_vars["selected_for_frontier"] = False
                continue
            self._try_retain_frontier_entry(entry, retained, diversity_counts)

        self._frontier = retained
        return {entry["node"].id for entry in self._frontier}

    def _frontier_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "node_id": entry["node"].id,
                "parent_id": entry["node"].parent_id,
                "depth": entry["depth"],
                "priority": self._node_priority(entry["node"]),
                "score": entry["node"].score,
                "status": entry["node"].status.value,
                "state_signature": entry["node"].known_vars.get("state_signature"),
                "diversity_key": entry["node"].known_vars.get("diversity_key"),
                "route_family": entry["node"].known_vars.get("route_family"),
                "correction_mode": entry["node"].known_vars.get("correction_mode"),
                "correction_target": entry["node"].known_vars.get("correction_target"),
                "distributed_reasoning_slot": entry["node"].known_vars.get("distributed_reasoning_slot"),
                "needs_deeper_reasoning": bool(
                    entry["node"].known_vars.get("needs_deeper_reasoning", False)
                ),
                "child_context_count": len(self._extract_child_contexts(entry["problem_context"])),
            }
            for entry in self._frontier
        ]

    def save_state(self, file_path: str) -> None:
        """Persist scheduler state for trusted local resume."""

        snapshot = {
            "root_problem_context": self.root_problem_context,
            "root_node": self.root_node,
            "frontier": self._frontier,
            "expansion_log": self._expansion_log,
            "expanded_node_ids": self._expanded_node_ids,
            "signature_registry": self._signature_registry,
            "max_reflections": self.max_reflections,
            "expansion_budget": self.expansion_budget,
            "max_frontier_size": self.max_frontier_size,
            "max_children_per_expansion": self.max_children_per_expansion,
            "max_frontier_per_diversity_key": self.max_frontier_per_diversity_key,
            "children_key": self.children_key,
        }
        state = TreeSchedulerState(snapshot_blob=_serialize_blob(snapshot))
        path = Path(file_path)
        try:
            payload = state.model_dump_json(indent=2)
        except AttributeError:
            payload = state.json(indent=2)
        path.write_text(payload, encoding="utf-8")

    @classmethod
    def from_state_file(
        cls,
        file_path: str,
        *,
        backend_adapter_factory: Optional[Callable[[dict[str, Any]], ReasoningBackendAdapter]] = None,
        deletion_review_adapter: Optional[NodeDeletionReviewAdapter] = None,
    ) -> "ToTTreeScheduler":
        """Restore a scheduler from a saved state snapshot."""

        text = Path(file_path).read_text(encoding="utf-8")
        try:
            state = TreeSchedulerState.model_validate_json(text)
        except AttributeError:
            state = TreeSchedulerState.parse_raw(text)

        snapshot = _deserialize_blob(state.snapshot_blob)
        scheduler = cls(
            root_problem_context=snapshot["root_problem_context"],
            max_reflections=snapshot["max_reflections"],
            expansion_budget=snapshot["expansion_budget"],
            max_frontier_size=snapshot["max_frontier_size"],
            max_children_per_expansion=snapshot["max_children_per_expansion"],
            max_frontier_per_diversity_key=snapshot["max_frontier_per_diversity_key"],
            children_key=snapshot["children_key"],
            backend_adapter_factory=backend_adapter_factory,
            deletion_review_adapter=deletion_review_adapter,
        )
        scheduler.root_node = snapshot["root_node"]
        scheduler._frontier = snapshot["frontier"]
        scheduler._expansion_log = snapshot["expansion_log"]
        scheduler._expanded_node_ids = snapshot["expanded_node_ids"]
        scheduler._signature_registry = snapshot["signature_registry"]
        if scheduler.root_node is not None:
            scheduler._rebuild_node_index(scheduler.root_node)
        return scheduler

    def _rebuild_node_index(self, node: ToTNode) -> None:
        self._node_index[node.id] = node
        for child in node.children:
            self._rebuild_node_index(child)

    def _node_snapshot(self, node: ToTNode) -> NodeSnapshot:
        return NodeSnapshot(
            id=node.id,
            parent_id=node.parent_id,
            thought_step=node.thought_step,
            equations=list(node.equations),
            known_vars=dict(node.known_vars),
            used_models=list(node.used_models),
            quantities=dict(node.quantities),
            boundary_conditions=dict(node.boundary_conditions),
            status=node.status,
            fsm_state=node.fsm_state,
            result_state=node.result_state,
            score=node.score,
            reflection_history=list(node.reflection_history),
        )

    def _normalize_review_payload(
        self,
        payload: BaseModel | dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(payload, BaseModel):
            return self._model_dump(payload)
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError("Deletion review adapter must return a Pydantic model or a dictionary payload.")

    def _build_model(self, model_type: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
        unexpected = set(payload) - _model_field_names(model_type)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Unexpected fields for {model_type.__name__}: {names}")
        try:
            return model_type.model_validate(payload)
        except AttributeError:
            return model_type.parse_obj(payload)

    def _model_dump(self, model: BaseModel) -> dict[str, Any]:
        try:
            return model.model_dump()
        except AttributeError:
            return model.dict()

    def _collect_subtree_nodes(self, node: ToTNode) -> list[ToTNode]:
        collected = [node]
        for child in node.children:
            collected.extend(self._collect_subtree_nodes(child))
        return collected

    def _resync_runtime_state(self) -> None:
        self._node_index = {}
        self._signature_registry = {}
        if self.root_node is None:
            self._frontier = []
            return

        self._clear_scheduler_metadata(self.root_node)
        self._sync_existing_tree(self.root_node, parent_node=None)
        retained_ids = self._rebalance_frontier()
        self._refresh_selection_metadata(self.root_node, retained_ids)

    def _clear_scheduler_metadata(self, node: ToTNode) -> None:
        for key in [
            "diversity_key",
            "expandable",
            "merged_duplicate_count",
            "merged_duplicate_node_ids",
            "merged_duplicate_parent_ids",
            "merged_into_node_id",
            "scheduler_action",
            "selected_child_ids",
            "selected_for_frontier",
            "sibling_priority",
            "sibling_ranking",
            "state_signature",
            "suppressed_by_scheduler",
        ]:
            node.known_vars.pop(key, None)
        for child in node.children:
            self._clear_scheduler_metadata(child)

    def _sync_existing_tree(
        self,
        node: ToTNode,
        parent_node: Optional[ToTNode],
    ) -> None:
        self._node_index[node.id] = node
        node.known_vars["state_signature"] = self._compute_state_signature(node)
        node.known_vars["diversity_key"] = self._compute_diversity_key(node)

        if parent_node is None:
            node.known_vars["scheduler_action"] = "root"
            if node.status == NodeStatus.ACTIVE:
                self._signature_registry[node.known_vars["state_signature"]] = node.id
        else:
            scheduler_action = self._apply_scheduler_controls(node, parent_node)
            node.known_vars["scheduler_action"] = scheduler_action or "expanded"

        ranked_children = sorted(node.children, key=self._node_ranking_key)
        sibling_ranking: list[dict[str, Any]] = []
        for rank, child in enumerate(ranked_children, start=1):
            child.known_vars["sibling_rank"] = rank
            child.known_vars["sibling_priority"] = self._node_priority(child)
            self._sync_existing_tree(child, parent_node=node)
            sibling_ranking.append(
                {
                    "node_id": child.id,
                    "rank": rank,
                    "status": child.status.value,
                    "priority": self._node_priority(child),
                    "score": child.score,
                    "expandable": False,
                    "scheduler_action": child.known_vars.get("scheduler_action", "expanded"),
                    "diversity_key": child.known_vars.get("diversity_key"),
                }
            )
        if sibling_ranking:
            node.known_vars["sibling_ranking"] = sibling_ranking

    def _refresh_selection_metadata(self, node: ToTNode, frontier_ids: set[str]) -> None:
        node.known_vars["selected_for_frontier"] = node.id in frontier_ids
        if node.children:
            node.known_vars["selected_child_ids"] = [
                child.id for child in node.children if child.id in frontier_ids
            ]
            sibling_ranking = node.known_vars.get("sibling_ranking", [])
            refreshed_sibling_ranking: list[dict[str, Any]] = []
            for entry in sibling_ranking:
                child = self._node_index.get(entry.get("node_id"))
                if child is None:
                    continue
                updated = dict(entry)
                updated["priority"] = self._node_priority(child)
                updated["score"] = child.score
                updated["scheduler_action"] = child.known_vars.get("scheduler_action", updated.get("scheduler_action", "expanded"))
                updated["expandable"] = child.id in frontier_ids
                refreshed_sibling_ranking.append(updated)
            node.known_vars["sibling_ranking"] = refreshed_sibling_ranking
        for child in node.children:
            self._refresh_selection_metadata(child, frontier_ids)


__all__ = ["ToTTreeScheduler", "TreeSchedulerState"]