"""Single-node finite-state machine for building ToT nodes."""

from __future__ import annotations

from difflib import SequenceMatcher
from importlib import import_module
from typing import Any, Optional

from pydantic import BaseModel

from .backend import (
    DeterministicContextBackendAdapter,
    EvaluationRequest,
    ProposalRequest,
    ReasoningBackendAdapter,
    ReflectionRequest,
)
from .models import (
    CalculationPayload,
    EvaluationBreakdown,
    EvaluationPayload,
    FSMState,
    NodeResultState,
    NodeSnapshot,
    NodeStatus,
    ProposalPayload,
    ReflectionPayload,
    ToTNode,
)
from .utils import _model_field_names


class NodeBuilderFSM:
    """Finite-state machine that manages the lifecycle of one ToT node.

    The FSM owns exactly one working node. A parent may be supplied so the built
    node can be attached into a larger tree after execution completes.

    Notes
    -----
    ``problem_context`` is also used as a deterministic mock input source so the
    FSM can be tested without any live model calls. Each state handler reads from
    one optional context key:

    - ``proposal``: dict or list[dict]
    - ``calculation``: dict or list[dict]
    - ``evaluation``: dict or list[dict]
    - ``reflection``: dict or list[dict]

    In the calculation stage, the payload may additionally contain
    ``skill_params``. These parameters are forwarded to the hard-rule checking
    skill in ``skills.py`` together with the current node equations and known
    variables.

    When a list is provided, the FSM selects the item that corresponds to the
    current attempt index, clamping to the last item if the list is shorter than
    the number of attempts.

    Scoring policy
    --------------
    The evaluation stage uses a fixed weighted rubric instead of accepting an
    arbitrary total score:

    - physical consistency: 50%
    - variable grounding: 25%
    - contextual relevance: 10%
    - simplicity (Occam's razor): 15%

    Physical hard-rule violations have veto power for non-root nodes. If a
    violation is found during either calculation or evaluation, child nodes are
    pruned by rule immediately and no soft score can rescue them. Root nodes are
    special-cased so one bad first draft is sent back through reflection instead
    of collapsing the whole session. Missing required context and other
    recoverable deterministic issues are also sent back through reflection
    instead of being pruned. Meta-task scope drift is recorded as diagnostics
    only and does not block the branch. Low scores never prune the node; they
    only reduce downstream expansion priority.
    """

    SCORE_THRESHOLD = 6.0
    LOW_SCORE_CONTINUE_RATIO = 0.66
    LOCAL_SCORE_CAP_MARGIN = 0.25
    PHYSICAL_WEIGHT = 0.50
    VARIABLE_WEIGHT = 0.25
    RELEVANCE_WEIGHT = 0.10
    SIMPLICITY_WEIGHT = 0.15
    STRATEGY_SCAN_STEP_GUIDANCE = (
        "Analyze the next-step strategy space at planning level. "
        "Compare plausible governing routes briefly, keep one route-local claim per node, and do not solve the final answer yet."
    )
    RECOVERABLE_RULE_PREFIXES = (
        "No candidate equations were provided for hard-rule checking.",
        "Missing required variable:",
        "No equation matches required pattern:",
        "Missing required model:",
        "No model matches required pattern:",
        "Missing required boundary condition key:",
        "No boundary condition matches required pattern:",
        "Missing required boundary condition:",
    )
    BOUNDARY_GROUNDING_VIOLATION_PREFIX = "Boundary condition key is not grounded in equations or known variables:"
    SEMANTIC_DELTA_TEXT_SIMILARITY = 0.72
    SEMANTIC_DELTA_METADATA_KEYS = frozenset(
        {
            "route_family",
            "correction_mode",
            "correction_target",
            "distributed_reasoning_slot",
            "orchestrator_task",
            "node_event_log",
            "hard_rule_check",
            "hard_rule_violations",
            "recoverable_rule_violations",
            "ignored_review_rule_violations",
            "recoverable_rule_retry_pending",
            "evaluation_breakdown",
            "evaluation_passed",
            "needs_deeper_reasoning",
            "low_score_reason",
            "continue_on_low_score",
            "expansion_priority",
            "top66_low_score_candidate",
            "semantic_delta_critique",
        }
    )

    def __init__(
        self,
        parent_node: Optional[ToTNode],
        problem_context: dict[str, Any],
        max_reflections: int = 2,
        backend_adapter: Optional[ReasoningBackendAdapter] = None,
    ) -> None:
        """Initialize the node builder.

        Parameters
        ----------
        parent_node
            Parent node in the ToT tree. ``None`` is valid for a root-level node.
        problem_context
            Arbitrary context payload for the problem plus deterministic mock
            outputs used by the handlers.
        max_reflections
            Maximum number of reflection retries allowed before pruning by SLM.
        """

        self.parent_node = parent_node
        self.problem_context = problem_context
        self.max_reflections = max_reflections
        self.backend_adapter = backend_adapter or DeterministicContextBackendAdapter(problem_context)

        self.node = ToTNode(parent_id=parent_node.id if parent_node else None)

        self._reflection_count = 0
        self._calculation_attempt = 0
        self._evaluation_attempt = 0
        self._attached_to_parent = False

    def run(self) -> ToTNode:
        """Drive the FSM until the node reaches ``FINALIZED``.

        Returns
        -------
        ToTNode
            The finalized node after all state transitions have completed.
        """

        while self.node.fsm_state != FSMState.FINALIZED:
            if self.node.fsm_state == FSMState.PROPOSE:
                self._handle_propose()
            elif self.node.fsm_state == FSMState.CALCULATE:
                self._handle_calculate()
            elif self.node.fsm_state == FSMState.EVALUATE:
                self._handle_evaluate()
            elif self.node.fsm_state == FSMState.REFLECT:
                self._handle_reflect()
            else:
                raise ValueError(f"Unsupported FSM state: {self.node.fsm_state}")

        self._attach_to_parent_once()
        return self.node

    def _handle_propose(self) -> None:
        """Mock the proposal step and move unconditionally to ``CALCULATE``.

        State transition
        ----------------
        PROPOSE -> CALCULATE
        """

        request = ProposalRequest(
            attempt_index=0,
            problem_context=self.problem_context,
            current_node=self._node_snapshot(self.node),
            parent_node=self._node_snapshot(self.parent_node) if self.parent_node else None,
        )
        proposal = self._build_model(
            ProposalPayload,
            self._normalize_backend_payload(self.backend_adapter.propose(request)),
        )
        top_level_models = self.problem_context.get("used_models", [])
        top_level_quantities = self.problem_context.get("quantities", {})
        top_level_boundary_conditions = self.problem_context.get("boundary_conditions", {})

        self.node.thought_step = str(
            proposal.thought_step
            or self.problem_context.get(
                "thought_step",
                self.problem_context.get(
                    "problem_statement",
                    "Propose a physically meaningful next step.",
                ),
            )
        )
        proposal_equations = self._flatten_string_items(proposal.equations)
        if proposal_equations or not self._is_equation_optional_meta_task_context():
            self.node.equations = proposal_equations or ["F = m * a"]
        else:
            self.node.equations = []
        self.node.known_vars.update(dict(proposal.known_vars))
        self._apply_orchestrator_task_to_problem_context(proposal.known_vars.get("orchestrator_task"))
        self._inherit_route_metadata_from_problem_context()
        self._apply_node_physics_fields(
            used_models=(
                proposal.used_models
                if proposal.used_models
                else list(top_level_models) if isinstance(top_level_models, list) else []
            ),
            quantities=(
                proposal.quantities
                if proposal.quantities
                else dict(top_level_quantities) if isinstance(top_level_quantities, dict) else {}
            ),
            boundary_conditions=(
                proposal.boundary_conditions
                if proposal.boundary_conditions
                else dict(top_level_boundary_conditions)
                if isinstance(top_level_boundary_conditions, dict)
                else {}
            ),
        )
        if self._enforce_nonterminal_semantic_delta(source_state="propose"):
            return
        self.node.fsm_state = FSMState.CALCULATE

    def _handle_calculate(self) -> None:
        """Run deterministic calculation checks and the hard-rule skill.

        Branches
        --------
        A. Hard error
            status = ``PRUNED_BY_RULE`` and state -> ``FINALIZED``
        B. Calculation passes
            known_vars updated and state -> ``EVALUATE``
        """

        payload = self._select_payload("calculation", attempt_index=self._calculation_attempt)
        calculation = self._build_model(CalculationPayload, payload)
        self._calculation_attempt += 1

        candidate_known_vars = dict(self.node.quantities)
        candidate_known_vars.update(self.node.known_vars)
        candidate_known_vars.update(calculation.known_vars)
        candidate_equations = self._flatten_string_items(calculation.equations or self.node.equations)
        candidate_quantities = dict(self.node.quantities)
        candidate_quantities.update(calculation.quantities)
        candidate_boundary_conditions = dict(self.node.boundary_conditions)
        candidate_boundary_conditions.update(calculation.boundary_conditions)
        candidate_used_models = self._merge_unique_strings(self.node.used_models, calculation.used_models)
        candidate_known_vars.update(candidate_quantities)

        hard_rule_result = self._run_hard_rule_check_skill(
            calculation=calculation,
            equations=candidate_equations,
            known_vars=candidate_known_vars,
            used_models=candidate_used_models,
            boundary_conditions=candidate_boundary_conditions,
        )
        self.node.known_vars["hard_rule_check"] = hard_rule_result

        if not hard_rule_result["passed"]:
            physical_violations = list(hard_rule_result.get("physical_violations", []))
            recoverable_violations = list(hard_rule_result.get("recoverable_violations", []))
            if recoverable_violations:
                self.node.known_vars["recoverable_rule_violations"] = recoverable_violations
            else:
                self.node.known_vars.pop("recoverable_rule_violations", None)

            if physical_violations:
                if self._bounce_root_rule_violations(
                    physical_violations,
                    source_state="calculate",
                ):
                    return
                self.node.known_vars["hard_rule_violations"] = physical_violations
                self.node.status = NodeStatus.PRUNED_BY_RULE
                self.node.result_state = NodeResultState.DROP
                self.node.fsm_state = FSMState.FINALIZED
                self._record_node_event(
                    "pruned-by-rule",
                    source_state="calculate",
                    violations=physical_violations,
                )
                return

            self.node.known_vars.pop("hard_rule_violations", None)
            critique = "; ".join(recoverable_violations) or "Recoverable rule violation detected."
            self.node.reflection_history.append(critique)
            self.node.known_vars["recoverable_rule_retry_pending"] = True
            if self._reflection_count < self._allowed_reflection_attempts(recoverable_rule_retry_pending=True):
                self.node.fsm_state = FSMState.REFLECT
                self._record_node_event(
                    "sent-to-reflection",
                    source_state="calculate",
                    trigger="recoverable-rule-violation",
                    critique=critique,
                    violations=recoverable_violations,
                )
                return

            self._finalize_recoverable_rule_violation(recoverable_violations)
            return

        if calculation.known_vars:
            self.node.known_vars.update(calculation.known_vars)
        else:
            self.node.known_vars["validation_passed"] = True
        self.node.known_vars.pop("hard_rule_violations", None)
        self.node.known_vars.pop("recoverable_rule_violations", None)
        self.node.known_vars.pop("recoverable_rule_retry_pending", None)
        if candidate_equations:
            self.node.equations = self._flatten_string_items(candidate_equations)
        self._apply_node_physics_fields(
            used_models=calculation.used_models,
            quantities=calculation.quantities,
            boundary_conditions=calculation.boundary_conditions,
        )
        self.node.fsm_state = FSMState.EVALUATE

    def _handle_evaluate(self) -> None:
        """Mock an SLM evaluation pass over the current node.

        Branches
        --------
        A. Hard-rule violation
            status = ``PRUNED_BY_RULE`` and state -> ``FINALIZED``
        A. score >= 6.0
            status = ``ACTIVE`` and state -> ``FINALIZED``
        B. score < 6.0
            critique appended to ``reflection_history`` and the node either
            reflects locally or remains ``ACTIVE`` for deeper downstream
            reasoning
        """

        request = EvaluationRequest(
            attempt_index=self._evaluation_attempt,
            problem_context=self.problem_context,
            current_node=self._node_snapshot(self.node),
        )
        evaluation = self._build_model(
            EvaluationPayload,
            self._normalize_backend_payload(self.backend_adapter.evaluate(request)),
        )
        self._evaluation_attempt += 1

        filtered_eval_violations = self._filter_review_hard_rule_violations(
            list(evaluation.hard_rule_violations)
        )
        if filtered_eval_violations != list(evaluation.hard_rule_violations):
            evaluation_payload = self._normalize_backend_payload(evaluation)
            evaluation_payload["hard_rule_violations"] = filtered_eval_violations
            evaluation = self._build_model(EvaluationPayload, evaluation_payload)

        physical_eval_violations, recoverable_eval_violations = self._categorize_rule_violations(
            list(evaluation.hard_rule_violations)
        )
        if physical_eval_violations != list(evaluation.hard_rule_violations):
            evaluation_payload = self._normalize_backend_payload(evaluation)
            evaluation_payload["hard_rule_violations"] = physical_eval_violations
            evaluation = self._build_model(EvaluationPayload, evaluation_payload)

        breakdown = self._score_evaluation(evaluation)
        breakdown = self._apply_local_score_caps(
            breakdown,
            recoverable_rule_violations=recoverable_eval_violations,
        )
        self.node.score = breakdown.weighted_score
        self.node.known_vars["evaluation_breakdown"] = self._model_dump(breakdown)

        if breakdown.hard_rule_violations:
            if self._bounce_root_rule_violations(
                list(breakdown.hard_rule_violations),
                source_state="evaluate",
            ):
                if recoverable_eval_violations:
                    self.node.known_vars["recoverable_rule_violations"] = recoverable_eval_violations
                return
            self.node.known_vars["hard_rule_violations"] = list(breakdown.hard_rule_violations)
            if recoverable_eval_violations:
                self.node.known_vars["recoverable_rule_violations"] = recoverable_eval_violations
            self.node.status = NodeStatus.PRUNED_BY_RULE
            self.node.result_state = NodeResultState.DROP
            self.node.fsm_state = FSMState.FINALIZED
            self._record_node_event(
                "pruned-by-rule",
                source_state="evaluate",
                violations=list(breakdown.hard_rule_violations),
                recoverable_violations=recoverable_eval_violations,
            )
            return

        if recoverable_eval_violations:
            self.node.known_vars["recoverable_rule_violations"] = recoverable_eval_violations
            self.node.known_vars["recoverable_rule_retry_pending"] = True
        else:
            self.node.known_vars.pop("recoverable_rule_retry_pending", None)

        self.node.known_vars["evaluation_passed"] = breakdown.passed

        if breakdown.passed:
            self.node.known_vars["needs_deeper_reasoning"] = False
            self.node.known_vars["expansion_priority"] = 1.0
            self.node.known_vars.pop("low_score_reason", None)
            self.node.known_vars.pop("continue_on_low_score", None)
            self.node.known_vars.pop("top66_low_score_candidate", None)
            self.node.known_vars.pop("hard_rule_violations", None)
            self.node.known_vars.pop("recoverable_rule_retry_pending", None)
            if self._should_mark_final_result_on_pass():
                self.node.status = NodeStatus.SOLVED
                self.node.result_state = NodeResultState.FINALIZE
            else:
                self.node.status = NodeStatus.ACTIVE
                self.node.result_state = NodeResultState.PASS
            self.node.fsm_state = FSMState.FINALIZED
            return

        self.node.reflection_history.append(breakdown.reason)
        if self._reflection_count < self._allowed_reflection_attempts(
            recoverable_rule_retry_pending=bool(recoverable_eval_violations)
        ):
            self.node.fsm_state = FSMState.REFLECT
            self._record_node_event(
                "sent-to-reflection",
                source_state="evaluate",
                trigger="low-score" if not recoverable_eval_violations else "recoverable-rule-violation",
                critique=breakdown.reason,
                violations=recoverable_eval_violations,
            )
            return

        self._finalize_for_deeper_reasoning(breakdown)

    def _handle_reflect(self) -> None:
        """Mock reflection-driven revision before a retry.

        Branches
        --------
        A. Reflection limit reached
            status = ``ACTIVE`` and state -> ``FINALIZED`` with low-score
            continuation metadata
        B. Retry still allowed
            reflection count incremented, equations revised, state -> ``CALCULATE``
        """

        recoverable_rule_retry_pending = bool(self.node.known_vars.pop("recoverable_rule_retry_pending", False))
        if self._reflection_count >= self._allowed_reflection_attempts(
            recoverable_rule_retry_pending=recoverable_rule_retry_pending
        ):
            if self.node.known_vars.get("recoverable_rule_violations"):
                self._finalize_recoverable_rule_violation(
                    list(self.node.known_vars.get("recoverable_rule_violations", []))
                )
                return
            self._finalize_for_deeper_reasoning()
            return

        request = ReflectionRequest(
            attempt_index=self._reflection_count,
            problem_context=self.problem_context,
            current_node=self._node_snapshot(self.node),
            latest_critique=self.node.reflection_history[-1] if self.node.reflection_history else "",
        )
        reflection = self._build_model(
            ReflectionPayload,
            self._normalize_backend_payload(self.backend_adapter.reflect(request)),
        )
        self._reflection_count += 1

        revised_thought = reflection.thought_step or f"{self.node.thought_step} | reflected attempt {self._reflection_count}"
        revised_equations = self._flatten_string_items(reflection.equations) or (
            [*self._flatten_string_items(self.node.equations), f"refined_attempt_{self._reflection_count}"]
            if self.node.equations
            else [f"refined_attempt_{self._reflection_count}"],
        )

        self.node.thought_step = str(revised_thought)
        self.node.equations = self._flatten_string_items(revised_equations)
        if reflection.known_vars:
            self.node.known_vars.update(dict(reflection.known_vars))
        self._apply_orchestrator_task_to_problem_context(reflection.known_vars.get("orchestrator_task"))
        self._inherit_route_metadata_from_problem_context()
        self._apply_node_physics_fields(
            used_models=reflection.used_models,
            quantities=reflection.quantities,
            boundary_conditions=reflection.boundary_conditions,
        )
        if self._enforce_nonterminal_semantic_delta(source_state="reflect"):
            return
        self.node.fsm_state = FSMState.CALCULATE

    def _enforce_nonterminal_semantic_delta(self, *, source_state: str) -> bool:
        critique = self._missing_nonterminal_semantic_delta_critique()
        if not critique:
            self.node.known_vars.pop("semantic_delta_critique", None)
            return False

        self.node.known_vars["semantic_delta_critique"] = critique
        self.node.reflection_history.append(critique)

        if self._reflection_count < self._allowed_reflection_attempts():
            self.node.fsm_state = FSMState.REFLECT
            self._record_node_event(
                "sent-to-reflection",
                source_state=source_state,
                trigger="missing-semantic-delta",
                critique=critique,
                route_family=self.node.known_vars.get("route_family"),
            )
            return True

        self._finalize_semantic_delta_violation(critique)
        return True

    def _missing_nonterminal_semantic_delta_critique(self) -> str:
        if not self._should_enforce_parent_semantic_delta():
            return ""
        has_explicit_delta = self._has_explicit_semantic_delta_vs_parent()
        if not self._is_semantically_redundant_with_parent():
            return ""
        if has_explicit_delta:
            return (
                "Non-terminal child hid its new local delta only in structured fields while repeating the parent thought_step. "
                "Rewrite thought_step so it explicitly names exactly one new correction, boundary condition, or control parameter."
            )
        return (
            "Non-terminal proposal repeated its parent without adding one explicit correction, "
            "boundary condition, or control parameter. Revise the branch by introducing exactly one "
            "new local delta for the selected route."
        )

    def _should_enforce_parent_semantic_delta(self) -> bool:
        if self.parent_node is None:
            return False

        meta_task_progress = self.problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False
        if str(meta_task_progress.get("phase", "")).strip() != "incremental_refinement":
            return False
        if bool(meta_task_progress.get("is_terminal_step")):
            return False

        route_family = str(
            self.node.known_vars.get("route_family")
            or meta_task_progress.get("selected_route_family")
            or ""
        ).strip()
        return bool(route_family)

    def _has_explicit_semantic_delta_vs_parent(self) -> bool:
        if self.parent_node is None:
            return False
        if self._new_normalized_items(self.node.equations, self.parent_node.equations):
            return True
        if self._new_mapping_items(self.node.boundary_conditions, self.parent_node.boundary_conditions):
            return True
        if self._new_mapping_items(self.node.quantities, self.parent_node.quantities):
            return True
        if self._new_mapping_items(
            self._semantic_known_vars(self.node.known_vars),
            self._semantic_known_vars(self.parent_node.known_vars),
        ):
            return True
        return False

    def _is_semantically_redundant_with_parent(self) -> bool:
        if self.parent_node is None:
            return False
        parent_text = self._normalize_semantic_text(self.parent_node.thought_step)
        child_text = self._normalize_semantic_text(self.node.thought_step)
        if not parent_text or not child_text:
            return False
        if parent_text == child_text:
            return True
        return SequenceMatcher(None, parent_text, child_text).ratio() >= self.SEMANTIC_DELTA_TEXT_SIMILARITY

    def _semantic_known_vars(self, mapping: dict[str, Any]) -> dict[str, Any]:
        semantic_items: dict[str, Any] = {}
        for key, value in mapping.items():
            normalized_key = str(key).strip()
            if not normalized_key or normalized_key in self.SEMANTIC_DELTA_METADATA_KEYS:
                continue
            if isinstance(value, (dict, list, tuple, set)):
                continue
            semantic_items[normalized_key] = value
        return semantic_items

    def _new_mapping_items(self, current: dict[str, Any], parent: dict[str, Any]) -> bool:
        for key, value in current.items():
            if key not in parent or parent.get(key) != value:
                return True
        return False

    def _new_normalized_items(self, current: list[str], parent: list[str]) -> bool:
        parent_items = {self._normalize_semantic_text(item) for item in parent if self._normalize_semantic_text(item)}
        for item in current:
            normalized = self._normalize_semantic_text(item)
            if normalized and normalized not in parent_items:
                return True
        return False

    def _normalize_semantic_text(self, value: Any) -> str:
        text = str(value).strip().lower()
        if not text:
            return ""
        normalized_chars: list[str] = []
        previous_was_space = False
        for char in text:
            if char.isalnum():
                normalized_chars.append(char)
                previous_was_space = False
                continue
            if previous_was_space:
                continue
            normalized_chars.append(" ")
            previous_was_space = True
        return " ".join("".join(normalized_chars).split())

    def _inherit_route_metadata_from_problem_context(self) -> None:
        route_focus = self.problem_context.get("route_focus")
        meta_task_progress = self.problem_context.get("meta_task_progress")

        route_family = ""
        correction_mode = ""
        correction_target = ""
        distributed_reasoning_slot = ""

        if isinstance(route_focus, dict):
            route_family = str(route_focus.get("route_family") or route_focus.get("label") or "").strip()
            correction_mode = str(route_focus.get("correction_mode", "")).strip()
            correction_target = str(route_focus.get("correction_target", "")).strip()
            distributed_reasoning_slot = str(route_focus.get("slot", "")).strip()

        if isinstance(meta_task_progress, dict):
            route_family = route_family or str(meta_task_progress.get("selected_route_family", "")).strip()
            correction_mode = correction_mode or str(meta_task_progress.get("selected_correction_mode", "")).strip()
            correction_target = correction_target or str(meta_task_progress.get("selected_correction_target", "")).strip()
            distributed_reasoning_slot = (
                distributed_reasoning_slot
                or str(meta_task_progress.get("distributed_reasoning_slot", "")).strip()
            )

        if route_family:
            self.node.known_vars.setdefault("route_family", route_family)
        if correction_mode:
            self.node.known_vars.setdefault("correction_mode", correction_mode)
        if correction_target:
            self.node.known_vars.setdefault("correction_target", correction_target)
        if distributed_reasoning_slot:
            self.node.known_vars.setdefault("distributed_reasoning_slot", distributed_reasoning_slot)

    def _apply_orchestrator_task_to_problem_context(self, orchestrator_task: Any) -> None:
        """Persist orchestrator-selected guidance so downstream hard-rule checks use it."""

        if not isinstance(orchestrator_task, dict) or not orchestrator_task:
            return

        updated_context = dict(self.problem_context)
        updated_context["orchestrator_task"] = dict(orchestrator_task)
        meta_task_progress = updated_context.get("meta_task_progress")
        updated_progress = dict(meta_task_progress) if isinstance(meta_task_progress, dict) else {}

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

        if updated_progress:
            updated_context["meta_task_progress"] = updated_progress
        self.problem_context = updated_context

    def _should_mark_final_result_on_pass(self) -> bool:
        if bool(self.problem_context.get("final_result_candidate") or self.problem_context.get("finalize_on_pass")):
            return True

        meta_task_progress = self.problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False

        if bool(meta_task_progress.get("is_terminal_step")):
            return True

        try:
            current_step_index = int(meta_task_progress.get("current_step_index", -1))
            total_steps = int(meta_task_progress.get("total_steps", 0))
        except (TypeError, ValueError):
            return False
        return total_steps > 0 and current_step_index >= total_steps - 1

    def _finalize_for_deeper_reasoning(self, breakdown: Optional[EvaluationBreakdown] = None) -> None:
        """Keep a low-score non-rule node active for downstream expansion."""

        active_breakdown = breakdown
        if active_breakdown is None:
            breakdown_payload = self.node.known_vars.get("evaluation_breakdown", {})
            active_breakdown = self._build_model(EvaluationBreakdown, breakdown_payload)

        expansion_priority = self._compute_expansion_priority(active_breakdown.weighted_score)
        self.node.known_vars["evaluation_passed"] = False
        self.node.known_vars["needs_deeper_reasoning"] = True
        self.node.known_vars["low_score_reason"] = active_breakdown.reason
        self.node.known_vars["continue_on_low_score"] = True
        self.node.known_vars["expansion_priority"] = expansion_priority
        self.node.known_vars["top66_low_score_candidate"] = expansion_priority >= self._low_score_top66_cutoff()
        self.node.known_vars.pop("recoverable_rule_retry_pending", None)
        self.node.status = NodeStatus.ACTIVE
        self.node.result_state = NodeResultState.PASS
        self.node.fsm_state = FSMState.FINALIZED

    def _finalize_recoverable_rule_violation(self, violations: list[str]) -> None:
        """Keep non-physical rule issues out of pruning when retry budget is exhausted."""

        reason = "; ".join(str(item) for item in violations if str(item).strip())
        if not reason:
            reason = "Recoverable rule issue remained unresolved."
        self.node.known_vars["evaluation_passed"] = False
        self.node.known_vars["needs_deeper_reasoning"] = True
        self.node.known_vars["low_score_reason"] = reason
        self.node.known_vars["continue_on_low_score"] = True
        self.node.known_vars["expansion_priority"] = 0.0
        self.node.known_vars["top66_low_score_candidate"] = False
        self.node.known_vars.pop("hard_rule_violations", None)
        self.node.known_vars.pop("recoverable_rule_retry_pending", None)
        self.node.status = NodeStatus.ACTIVE
        self.node.result_state = NodeResultState.PASS
        self.node.fsm_state = FSMState.FINALIZED

    def _finalize_semantic_delta_violation(self, critique: str) -> None:
        """Drop unresolved parent-child semantic duplicates after retries are exhausted."""

        reason = str(critique).strip() or "Non-terminal semantic delta remained unresolved."
        self.node.known_vars["evaluation_passed"] = False
        self.node.known_vars["needs_deeper_reasoning"] = False
        self.node.known_vars["low_score_reason"] = reason
        self.node.known_vars["continue_on_low_score"] = False
        self.node.known_vars["expansion_priority"] = 0.0
        self.node.known_vars["top66_low_score_candidate"] = False
        self.node.known_vars.pop("recoverable_rule_retry_pending", None)
        self.node.status = NodeStatus.PRUNED_BY_SLM
        self.node.result_state = NodeResultState.DROP
        self.node.fsm_state = FSMState.FINALIZED

    def _bounce_root_rule_violations(self, violations: list[str], *, source_state: str) -> bool:
        """Send root-node rule failures through reflection before giving up.

        Root-node failures should not collapse the whole session on the first
        bad draft. Reuse the recoverable-rule path so the root gets at least one
        reflection attempt and, if still unresolved, remains active for the
        scheduler instead of being pruned away.
        """

        if self.parent_node is not None:
            return False

        critique = "; ".join(str(item) for item in violations if str(item).strip())
        if not critique:
            critique = "Root node violated a deterministic rule."

        self.node.known_vars["hard_rule_violations"] = list(violations)
        self.node.known_vars["recoverable_rule_violations"] = list(violations)
        self.node.reflection_history.append(critique)
        self.node.known_vars["recoverable_rule_retry_pending"] = True

        if self._reflection_count < self._allowed_reflection_attempts(
            recoverable_rule_retry_pending=True
        ):
            self.node.fsm_state = FSMState.REFLECT
            self._record_node_event(
                "bounce-to-reflection",
                source_state=source_state,
                trigger="root-rule-violation",
                critique=critique,
                violations=list(violations),
            )
            return True

        self._finalize_recoverable_rule_violation(list(violations))
        return True

    def _record_node_event(self, event: str, **fields: Any) -> None:
        raw_log = self.node.known_vars.get("node_event_log")
        event_log = raw_log if isinstance(raw_log, list) else []
        if raw_log is not event_log:
            self.node.known_vars["node_event_log"] = event_log

        entry: dict[str, Any] = {
            "event": str(event),
            "node_id": self.node.id,
            "parent_id": self.node.parent_id,
            "status": self.node.status.value,
            "fsm_state": self.node.fsm_state.value,
            "reflection_count": self._reflection_count,
        }
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, list):
                normalized = [item for item in value if item is not None]
                if normalized:
                    entry[key] = normalized
                continue
            if isinstance(value, dict):
                if value:
                    entry[key] = dict(value)
                continue
            normalized_value = str(value).strip() if isinstance(value, str) else value
            if normalized_value == "":
                continue
            entry[key] = normalized_value

        event_log.append(entry)

    def _compute_expansion_priority(self, weighted_score: float) -> float:
        """Map score to a downstream expansion priority in ``[0.0, 1.0]``."""

        if weighted_score >= self.SCORE_THRESHOLD:
            return 1.0
        return round(max(0.0, min(0.9999, weighted_score / self.SCORE_THRESHOLD)), 4)

    def _low_score_top66_cutoff(self) -> float:
        """Return the normalized cutoff for the top 66% low-score band."""

        return round(max(0.0, 1.0 - self.LOW_SCORE_CONTINUE_RATIO), 4)

    def _allowed_reflection_attempts(self, *, recoverable_rule_retry_pending: bool = False) -> int:
        """Return the retry budget for the current reflection transition."""

        if recoverable_rule_retry_pending:
            return max(1, self.max_reflections)
        return self.max_reflections

    def _select_payload(self, key: str, attempt_index: int) -> dict[str, Any]:
        """Select a deterministic mock payload for a state handler.

        Parameters
        ----------
        key
            Top-level key inside ``problem_context``.
        attempt_index
            Zero-based attempt number for retry-aware states.

        Returns
        -------
        dict[str, Any]
            The selected mock payload. Missing or malformed entries yield an
            empty dictionary.
        """

        payload = self.problem_context.get(key, {})
        if isinstance(payload, list):
            if not payload:
                return {}
            index = min(attempt_index, len(payload) - 1)
            selected = payload[index]
            return dict(selected) if isinstance(selected, dict) else {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _node_snapshot(self, node: Optional[ToTNode]) -> Optional[NodeSnapshot]:
        """Create a strict backend-facing node snapshot."""

        if node is None:
            return None
        return NodeSnapshot(
            id=node.id,
            parent_id=node.parent_id,
            thought_step=node.thought_step,
            equations=self._flatten_string_items(node.equations),
            known_vars=dict(node.known_vars),
            used_models=self._flatten_string_items(node.used_models),
            quantities=dict(node.quantities),
            boundary_conditions=dict(node.boundary_conditions),
            status=node.status,
            fsm_state=node.fsm_state,
            result_state=node.result_state,
            score=node.score,
            reflection_history=self._flatten_string_items(node.reflection_history),
        )

    def _normalize_backend_payload(self, payload: BaseModel | dict[str, Any]) -> dict[str, Any]:
        """Normalize backend output to a plain dictionary before schema validation."""

        if isinstance(payload, BaseModel):
            return self._model_dump(payload)
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError("Backend adapter must return a Pydantic model or a dictionary payload.")

    def _build_model(self, model_type: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
        """Construct a schema-locked Pydantic model with v1/v2 compatibility."""

        unexpected = set(payload) - _model_field_names(model_type)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Unexpected fields for {model_type.__name__}: {names}")

        try:
            return model_type.model_validate(payload)
        except AttributeError:
            return model_type.parse_obj(payload)

    def _model_dump(self, model: BaseModel) -> dict[str, Any]:
        """Serialize a Pydantic model with v1/v2 compatibility."""

        try:
            return model.model_dump()
        except AttributeError:
            return model.dict()

    def _apply_node_physics_fields(
        self,
        used_models: Optional[list[str]] = None,
        quantities: Optional[dict[str, Any]] = None,
        boundary_conditions: Optional[dict[str, Any]] = None,
    ) -> None:
        """Merge explicit physical-context fields into the current node."""

        if used_models:
            self.node.used_models = self._merge_unique_strings(self.node.used_models, used_models)
        if quantities:
            self.node.quantities.update(dict(quantities))
        if boundary_conditions:
            self.node.boundary_conditions.update(dict(boundary_conditions))

    def _flatten_string_items(self, value: Any) -> list[str]:
        """Return a flat list of non-empty strings from nested list-like data."""

        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            return [stripped] if stripped else []
        if isinstance(value, dict):
            return []
        if isinstance(value, (list, tuple, set)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(self._flatten_string_items(item))
            return flattened
        text = str(value).strip()
        return [text] if text else []

    def _merge_unique_strings(self, existing: list[str], new_values: list[str]) -> list[str]:
        """Return unique strings while preserving first-seen order."""

        merged = self._flatten_string_items(existing)
        seen = set(merged)
        for normalized in self._flatten_string_items(new_values):
            if normalized not in seen:
                merged.append(normalized)
                seen.add(normalized)
        return merged

    def _run_hard_rule_check_skill(
        self,
        calculation: CalculationPayload,
        equations: list[str],
        known_vars: dict[str, Any],
        used_models: list[str],
        boundary_conditions: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the hard-rule checking skill from ``skills.py``.

        The skill is the authoritative source for rule-based veto decisions in
        the calculation stage. Legacy payload flags are merged afterward only for
        backward compatibility.
        """

        try:
            skills_module = import_module("skills")
            invoke_skill = getattr(skills_module, "invoke_skill")
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Failed to load 'invoke_skill' from skills.py.") from exc

        skill_params = dict(calculation.skill_params)
        skill_params.setdefault("equations", self._flatten_string_items(equations))
        skill_params.setdefault("known_vars", dict(known_vars))
        skill_params.setdefault("used_models", list(used_models))
        skill_params.setdefault("boundary_conditions", dict(boundary_conditions))
        skill_params.setdefault("thought_step", str(self.node.thought_step))
        if self._is_equation_optional_meta_task_context() and not self._flatten_string_items(
            skill_params.get("equations")
        ):
            skill_params.setdefault("require_equations", False)

        meta_task = self.problem_context.get("meta_task")
        if isinstance(meta_task, dict) and meta_task:
            skill_params.setdefault("meta_task", dict(meta_task))
            skill_params.setdefault("enforce_meta_task_step_scope", True)

        meta_task_progress = self.problem_context.get("meta_task_progress")
        if isinstance(meta_task, dict) and meta_task:
            normalized_progress = self._normalize_meta_task_progress(
                meta_task=meta_task,
                meta_task_progress=meta_task_progress,
            )
            if normalized_progress:
                skill_params.setdefault("meta_task_progress", normalized_progress)

        invocation = invoke_skill(
            "tot_hard_rule_check",
            skill_params,
            include_trace=True,
        )
        result = invocation["result"]
        if not isinstance(result, dict):
            raise TypeError("Hard-rule checking skill must return a dictionary.")

        violations = list(result.get("violations", []))
        if calculation.hard_error:
            violations.append("Legacy hard_error flag set.")
        violations.extend(list(calculation.hard_rule_violations))

        merged_result = dict(result)
        effective_violations, ignored_violations = self._filter_ignorable_rule_violations(
            violations,
            checked=merged_result.get("checked"),
        )
        merged_result["violations"] = effective_violations
        merged_result["ignored_violations"] = ignored_violations
        physical_violations, recoverable_violations = self._categorize_rule_violations(
            effective_violations,
            checked=merged_result.get("checked"),
        )
        merged_result["physical_violations"] = physical_violations
        merged_result["recoverable_violations"] = recoverable_violations
        merged_result["passed"] = not physical_violations and not recoverable_violations
        merged_result["trace"] = invocation["trace"]
        return merged_result

    def _filter_ignorable_rule_violations(
        self,
        violations: list[str],
        *,
        checked: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        effective_violations: list[str] = []
        ignored_violations: list[str] = []
        for item in violations:
            violation = str(item).strip()
            if not violation:
                continue
            if self._is_ignorable_route_local_boundary_condition_violation(
                violation,
                checked=checked,
            ):
                ignored_violations.append(violation)
                continue
            effective_violations.append(violation)
        return effective_violations, ignored_violations

    def _categorize_rule_violations(
        self,
        violations: list[str],
        *,
        checked: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], list[str]]:
        """Split rule failures into pruning physical errors and recoverable retry errors."""

        recoverable_messages = set()
        if isinstance(checked, dict):
            meta_task_scope = checked.get("meta_task_step_scope")
            if isinstance(meta_task_scope, dict):
                recoverable_messages.update(
                    str(item) for item in meta_task_scope.get("violations", []) if str(item).strip()
                )

        physical_violations: list[str] = []
        recoverable_violations: list[str] = []
        for item in violations:
            violation = str(item).strip()
            if not violation:
                continue
            if violation in recoverable_messages or self._is_recoverable_rule_violation(violation):
                recoverable_violations.append(violation)
                continue
            physical_violations.append(violation)
        return physical_violations, recoverable_violations

    def _is_ignorable_route_local_boundary_condition_violation(
        self,
        violation: str,
        *,
        checked: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not violation.startswith(self.BOUNDARY_GROUNDING_VIOLATION_PREFIX):
            return False
        if not (
            self._is_non_terminal_route_focused_correction_context()
            or self._is_route_local_strategy_scan_context()
        ):
            return False
        if not isinstance(checked, dict):
            return True
        semantic_violations = {
            str(item).strip()
            for item in checked.get("semantic_boundary_violations", [])
            if str(item).strip()
        }
        return not semantic_violations or violation in semantic_violations

    def _is_non_terminal_route_focused_correction_context(self) -> bool:
        meta_task_progress = self.problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False
        if str(meta_task_progress.get("phase", "")).strip().lower() != "incremental_refinement":
            return False
        if bool(meta_task_progress.get("is_terminal_step", False)):
            return False

        current_step = str(meta_task_progress.get("current_step", "")).strip().lower()
        current_step_guidance = str(meta_task_progress.get("current_step_guidance", "")).strip().lower()
        marker = "choose one active correction or closure"
        if marker not in current_step and marker not in current_step_guidance:
            return False

        route_focus = self.problem_context.get("route_focus")
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip().lower()
        focused_route_family = (
            str(route_focus.get("route_family", "")).strip().lower()
            if isinstance(route_focus, dict)
            else ""
        )
        return bool(selected_route_family or focused_route_family)

    def _is_route_local_strategy_scan_context(self) -> bool:
        meta_task_progress = self.problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False
        if str(meta_task_progress.get("phase", "")).strip().lower() != "strategy_scan":
            return False

        current_step = str(meta_task_progress.get("current_step", "")).strip().lower()
        current_step_guidance = str(meta_task_progress.get("current_step_guidance", "")).strip().lower()
        if "route-local scan" not in current_step and "route-local" not in current_step_guidance:
            return False

        route_focus = self.problem_context.get("route_focus")
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip().lower()
        focused_route_family = (
            str(route_focus.get("route_family", "")).strip().lower()
            if isinstance(route_focus, dict)
            else ""
        )
        return bool(selected_route_family or focused_route_family)

    def _is_equation_optional_meta_task_context(self) -> bool:
        meta_task_progress = self.problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return False

        phase = str(meta_task_progress.get("phase", "")).strip().lower()
        if phase == "strategy_scan":
            return True
        if phase != "incremental_refinement":
            return False

        current_step = str(meta_task_progress.get("current_step", "")).strip().lower()
        current_step_guidance = str(meta_task_progress.get("current_step_guidance", "")).strip().lower()
        marker = "choose one active correction or closure"
        return marker in current_step or marker in current_step_guidance

    def _filter_review_hard_rule_violations(self, violations: list[str]) -> list[str]:
        meta_task_progress = self.problem_context.get("meta_task_progress")
        if not isinstance(meta_task_progress, dict) or not meta_task_progress:
            return violations

        phase = str(meta_task_progress.get("phase", "")).strip().lower()
        selected_route_family = str(meta_task_progress.get("selected_route_family", "")).strip()
        route_focus = self.problem_context.get("route_focus")
        has_route_focus = bool(selected_route_family) or (isinstance(route_focus, dict) and bool(route_focus))
        if phase != "strategy_scan" and not has_route_focus:
            return violations
        if self.parent_node is not None and not has_route_focus:
            return violations

        hard_rule_check = self.node.known_vars.get("hard_rule_check")
        if isinstance(hard_rule_check, dict) and not bool(hard_rule_check.get("passed", False)):
            return violations

        ignored = [str(item).strip() for item in violations if str(item).strip()]
        if ignored:
            recorded = self.node.known_vars.get("ignored_review_rule_violations")
            existing = list(recorded) if isinstance(recorded, list) else []
            self.node.known_vars["ignored_review_rule_violations"] = [*existing, *ignored]
        return []

    def _is_recoverable_rule_violation(self, violation: str) -> bool:
        """Return whether a deterministic rule failure should trigger a retry instead of pruning."""

        return violation.startswith(self.RECOVERABLE_RULE_PREFIXES)

    def _normalize_meta_task_progress(
        self,
        *,
        meta_task: dict[str, Any],
        meta_task_progress: Any,
    ) -> dict[str, Any]:
        normalized_progress = dict(meta_task_progress) if isinstance(meta_task_progress, dict) else {}
        step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
        first_step = str(meta_task.get("first_step", "")).strip()
        total_steps = len(step_ordering)

        try:
            current_step_index = int(normalized_progress.get("current_step_index", 0))
        except (TypeError, ValueError):
            current_step_index = 0
        if total_steps:
            current_step_index = max(0, min(current_step_index, total_steps - 1))
        else:
            current_step_index = max(0, current_step_index)

        current_step = str(normalized_progress.get("current_step", "")).strip()
        if not current_step:
            current_step = step_ordering[current_step_index] if step_ordering else first_step

        phase = str(normalized_progress.get("phase", "")).strip()
        if not phase:
            phase = "strategy_scan" if current_step_index == 0 else "incremental_refinement"

        current_step_guidance = str(normalized_progress.get("current_step_guidance", "")).strip()
        if not current_step_guidance:
            current_step_guidance = (
                self.STRATEGY_SCAN_STEP_GUIDANCE
                if phase == "strategy_scan"
                else (
                    f"Refine only the current subproblem: {current_step}. "
                    "Add or correct exactly one quantity, relation, approximation, or correction term, and leave all other pending fixes deferred."
                )
            )

        remaining_steps = normalized_progress.get("remaining_steps")
        if not isinstance(remaining_steps, list):
            remaining_steps = step_ordering[current_step_index + 1 :] if step_ordering else []

        previous_steps = normalized_progress.get("previous_steps")
        if not isinstance(previous_steps, list):
            previous_steps = step_ordering[:current_step_index] if step_ordering else []

        return {
            "current_step_index": current_step_index,
            "current_step": current_step,
            "current_step_guidance": current_step_guidance,
            "previous_steps": [str(item) for item in previous_steps if str(item).strip()],
            "remaining_steps": [str(item) for item in remaining_steps if str(item).strip()],
            "total_steps": total_steps,
            "phase": phase,
            "is_terminal_step": bool(step_ordering) and current_step_index >= total_steps - 1,
        }

    def _score_evaluation(self, evaluation: EvaluationPayload) -> EvaluationBreakdown:
        """Compute a structured 10-point score with an Occam simplicity term.

        The weighted score is defined as:

        ``10 * (0.50 * physical + 0.25 * grounding + 0.10 * relevance + 0.15 * simplicity)``

        This keeps the original FSM acceptance threshold at ``6.0`` while making
        the score traceable and non-arbitrary.
        """

        if evaluation.hard_rule_violations:
            return EvaluationBreakdown(
                physical_consistency=0.0,
                variable_grounding=0.0,
                contextual_relevance=0.0,
                simplicity=0.0,
                weighted_score=0.0,
                threshold=self.SCORE_THRESHOLD,
                passed=False,
                reason="Physical hard-rule violation detected.",
                hard_rule_violations=list(evaluation.hard_rule_violations),
            )

        legacy_hint = None if evaluation.score is None else max(0.0, min(1.0, evaluation.score / 10.0))
        physical_consistency = self._select_quality_value(
            evaluation.physical_consistency,
            legacy_hint,
            default=0.70,
        )
        variable_grounding = self._select_quality_value(
            evaluation.variable_grounding,
            legacy_hint,
            default=0.70,
        )
        contextual_relevance = self._select_quality_value(
            evaluation.contextual_relevance,
            legacy_hint,
            default=0.70,
        )
        simplicity = self._compute_simplicity_score(evaluation.simplicity_hint)

        weighted_score = 10.0 * (
            self.PHYSICAL_WEIGHT * physical_consistency
            + self.VARIABLE_WEIGHT * variable_grounding
            + self.RELEVANCE_WEIGHT * contextual_relevance
            + self.SIMPLICITY_WEIGHT * simplicity
        )
        weighted_score = round(weighted_score, 2)
        passed = weighted_score >= self.SCORE_THRESHOLD

        critique_parts: list[str] = []
        if physical_consistency < 0.60:
            critique_parts.append("physical consistency is below the acceptance floor")
        if variable_grounding < 0.55:
            critique_parts.append("variables are insufficiently grounded")
        if contextual_relevance < 0.50:
            critique_parts.append("the branch is weakly aligned with the problem goal")
        if simplicity < 0.50:
            critique_parts.append("the branch is unnecessarily complex under Occam's razor")

        reason = evaluation.reason.strip()
        if critique_parts:
            prefix = "; ".join(critique_parts)
            reason = f"{prefix}. {reason}".strip() if reason else prefix
        if not reason:
            reason = (
                "Weighted evaluation passed the acceptance threshold."
                if passed
                else "Weighted evaluation is below the acceptance threshold."
            )

        return EvaluationBreakdown(
            physical_consistency=physical_consistency,
            variable_grounding=variable_grounding,
            contextual_relevance=contextual_relevance,
            simplicity=simplicity,
            weighted_score=weighted_score,
            threshold=self.SCORE_THRESHOLD,
            passed=passed,
            reason=reason,
        )

    def _apply_local_score_caps(
        self,
        breakdown: EvaluationBreakdown,
        *,
        recoverable_rule_violations: Optional[list[str]] = None,
    ) -> EvaluationBreakdown:
        cap_reasons: list[str] = []

        semantic_delta_critique = str(self.node.known_vars.get("semantic_delta_critique", "")).strip()
        if semantic_delta_critique:
            cap_reasons.append(semantic_delta_critique)

        merged_recoverable = [
            str(item).strip()
            for item in [
                *list(self.node.known_vars.get("recoverable_rule_violations", [])),
                *(recoverable_rule_violations or []),
            ]
            if str(item).strip()
        ]
        if merged_recoverable:
            cap_reasons.append("; ".join(dict.fromkeys(merged_recoverable)))

        if not cap_reasons:
            return breakdown

        breakdown.weighted_score = round(
            min(breakdown.weighted_score, max(0.0, self.SCORE_THRESHOLD - self.LOCAL_SCORE_CAP_MARGIN)),
            2,
        )
        breakdown.passed = False
        prefix = "; ".join(cap_reasons)
        breakdown.reason = f"{prefix}. {breakdown.reason}".strip() if breakdown.reason else prefix
        return breakdown

    def _select_quality_value(
        self,
        explicit_value: Optional[float],
        legacy_hint: Optional[float],
        default: float,
    ) -> float:
        """Select a normalized quality value with backward-compatible fallback."""

        if explicit_value is not None:
            return float(explicit_value)
        if legacy_hint is not None:
            return float(legacy_hint)
        return float(default)

    def _compute_simplicity_score(self, simplicity_hint: Optional[float]) -> float:
        """Quantify simplicity using Occam's razor.

        The simplicity term is computed from the representational complexity of
        the branch. More equations, more known variables, and more reflection
        retries all reduce the score. An optional external hint can only lower
        the internal simplicity score, never inflate it.
        """

        equation_penalty = 0.12 * max(len(self.node.equations) - 1, 0)
        variable_penalty = 0.05 * len(self.node.known_vars)
        reflection_penalty = 0.18 * self._reflection_count
        base_simplicity = max(0.0, 1.0 - equation_penalty - variable_penalty - reflection_penalty)
        if simplicity_hint is None:
            return round(base_simplicity, 4)
        return round(min(base_simplicity, float(simplicity_hint)), 4)

    def _attach_to_parent_once(self) -> None:
        """Attach the finalized node to its parent exactly once."""

        if self.parent_node is None or self._attached_to_parent:
            return
        self.parent_node.children.append(self.node)
        self._attached_to_parent = True


__all__ = ["NodeBuilderFSM"]