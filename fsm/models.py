"""Core enums and Pydantic models for the ToT harness."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    """Terminal or active status assigned to a ToT node.

    Values
    ------
    ACTIVE
        The node is currently viable and has not been pruned.
    PRUNED_BY_RULE
        The node was rejected by deterministic math or rule checks.
    PRUNED_BY_SLM
        Reserved for legacy or future soft-pruning policies. The current FSM
        keeps low-score nodes active unless a hard rule is violated.
    SOLVED
        Reserved for upstream orchestration when a node is promoted to a final
        solved branch.
    """

    ACTIVE = "ACTIVE"
    PRUNED_BY_RULE = "PRUNED_BY_RULE"
    PRUNED_BY_SLM = "PRUNED_BY_SLM"
    SOLVED = "SOLVED"


class FSMState(str, Enum):
    """Internal lifecycle state for building a single node.

    The FSM always starts at ``PROPOSE`` and terminates at ``FINALIZED``.
    """

    PROPOSE = "PROPOSE"
    CALCULATE = "CALCULATE"
    EVALUATE = "EVALUATE"
    REFLECT = "REFLECT"
    FINALIZED = "FINALIZED"


class NodeResultState(str, Enum):
    """User-facing result semantics for a node after the current build attempt."""

    PENDING = "PENDING"
    PASS = "PASS"
    DROP = "DROP"
    FINALIZE = "FINALIZE"


class ToTNode(BaseModel):
    """Recursive tree node used by the ToT builder.

    Parameters
    ----------
    id
        Eight-character prefix of a UUID4 hex string.
    parent_id
        Optional parent node identifier.
    thought_step
        Human-readable thought or reasoning step for the node.
    equations
        Candidate equations proposed for the node.
    known_vars
        Known or newly derived variables collected during validation.
    used_models
        Physical models or approximations used by the node.
    quantities
        Physical quantities carried by the node.
    boundary_conditions
        Initial, boundary, or constraint conditions attached to the node.
    status
        High-level viability status.
    fsm_state
        Current state inside the node builder FSM.
    score
        Most recent evaluation score.
    reflection_history
        History of SLM critique strings used to trigger reflection.
    children
        Recursive list of child nodes.
    """

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    parent_id: Optional[str] = None

    thought_step: str = ""
    equations: list[str] = Field(default_factory=list)
    known_vars: dict[str, Any] = Field(default_factory=dict)
    used_models: list[str] = Field(default_factory=list)
    quantities: dict[str, Any] = Field(default_factory=dict)
    boundary_conditions: dict[str, Any] = Field(default_factory=dict)

    status: NodeStatus = NodeStatus.ACTIVE
    fsm_state: FSMState = FSMState.PROPOSE
    result_state: NodeResultState = NodeResultState.PENDING
    score: float = 0.0

    reflection_history: list[str] = Field(default_factory=list)
    children: list["ToTNode"] = Field(default_factory=list)


class NodePhysicsPayload(BaseModel):
    """Shared physical-context payload fields for node-building states."""

    used_models: list[str] = Field(default_factory=list)
    quantities: dict[str, Any] = Field(default_factory=dict)
    boundary_conditions: dict[str, Any] = Field(default_factory=dict)


class ProposalPayload(NodePhysicsPayload):
    """Structured mock payload for the proposal state."""

    thought_step: Optional[str] = None
    equations: list[str] = Field(default_factory=list)
    known_vars: dict[str, Any] = Field(default_factory=dict)


class CalculationPayload(NodePhysicsPayload):
    """Structured mock payload for deterministic calculation checks.

    Notes
    -----
    The primary hard-error source is the dedicated hard-rule checking skill in
    ``skills.py``. ``hard_error`` and ``hard_rule_violations`` are retained only
    as backward-compatible overrides.
    """

    hard_error: bool = False
    hard_rule_violations: list[str] = Field(default_factory=list)
    skill_params: dict[str, Any] = Field(default_factory=dict)
    known_vars: dict[str, Any] = Field(default_factory=dict)
    equations: list[str] = Field(default_factory=list)


class EvaluationPayload(BaseModel):
    """Structured mock payload for the SLM evaluation stage.

    All quality inputs are normalized to ``[0.0, 1.0]`` before the final score is
    mapped to the required ``0.0`` to ``10.0`` range.

    Scoring dimensions
    ------------------
    physical_consistency
        Dominant term. Captures whether the branch obeys core physical laws and
        uses equations coherently.
    variable_grounding
        Measures whether variables are explicitly connected to known or derived
        quantities rather than left ambiguous.
    contextual_relevance
        Measures whether the branch is aligned with the stated problem context.
    simplicity_hint
        Optional external simplicity estimate. The final simplicity term uses
        Occam's razor and will never exceed the internal complexity-based score.
    score
        Legacy compatibility hint on a ``0`` to ``10`` scale. If provided
        without the structured fields above, it is used as a fallback quality
        hint rather than as a raw final score.
    hard_rule_violations
        Physical hard-rule failures discovered during evaluation. Any non-empty
        list triggers rule pruning immediately.
    """

    physical_consistency: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    variable_grounding: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contextual_relevance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    simplicity_hint: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    score: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    reason: str = ""
    hard_rule_violations: list[str] = Field(default_factory=list)


class ReflectionPayload(NodePhysicsPayload):
    """Structured mock payload for the reflection state."""

    thought_step: Optional[str] = None
    equations: list[str] = Field(default_factory=list)
    known_vars: dict[str, Any] = Field(default_factory=dict)


class MetaAnalysisPayload(BaseModel):
    """Structured session-level planning payload used to guide later node expansion."""

    objective: str = ""
    givens: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    minimal_subproblems: list[str] = Field(default_factory=list)
    step_ordering: list[str] = Field(default_factory=list)
    first_step: str = ""
    completion_signals: list[str] = Field(default_factory=list)
    route_options: list[dict[str, Any]] = Field(default_factory=list)
    step_blueprints: list[dict[str, Any]] = Field(default_factory=list)


class OrchestratorTaskPayload(BaseModel):
    """Strict per-step orchestration payload that selects exactly one modeling task."""

    step_focus: str = ""
    current_step_guidance: str = ""
    task_breakdown: list[str] = Field(default_factory=list)
    selected_task: str = ""
    deferred_tasks: list[str] = Field(default_factory=list)
    completion_signals: list[str] = Field(default_factory=list)
    selected_route_family: str = ""
    candidate_tasks: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationBreakdown(BaseModel):
    """Quantitative breakdown for the SLM scoring stage.

    The final score is defined on a fixed 10-point scale to preserve the
    originally required transition threshold of ``6.0``.
    """

    physical_consistency: float = Field(ge=0.0, le=1.0)
    variable_grounding: float = Field(ge=0.0, le=1.0)
    contextual_relevance: float = Field(ge=0.0, le=1.0)
    simplicity: float = Field(ge=0.0, le=1.0)
    weighted_score: float = Field(ge=0.0, le=10.0)
    threshold: float = 6.0
    passed: bool
    reason: str = ""
    hard_rule_violations: list[str] = Field(default_factory=list)


class NodeSnapshot(BaseModel):
    """Backend-facing immutable snapshot of a node state."""

    id: str
    parent_id: Optional[str] = None
    thought_step: str = ""
    equations: list[str] = Field(default_factory=list)
    known_vars: dict[str, Any] = Field(default_factory=dict)
    used_models: list[str] = Field(default_factory=list)
    quantities: dict[str, Any] = Field(default_factory=dict)
    boundary_conditions: dict[str, Any] = Field(default_factory=dict)
    status: NodeStatus = NodeStatus.ACTIVE
    fsm_state: FSMState = FSMState.PROPOSE
    result_state: NodeResultState = NodeResultState.PENDING
    score: float = 0.0
    reflection_history: list[str] = Field(default_factory=list)


try:
    ToTNode.model_rebuild()
except AttributeError:
    ToTNode.update_forward_refs()


__all__ = [
    "CalculationPayload",
    "EvaluationBreakdown",
    "EvaluationPayload",
    "FSMState",
    "MetaAnalysisPayload",
    "NodeResultState",
    "NodePhysicsPayload",
    "NodeSnapshot",
    "NodeStatus",
    "ProposalPayload",
    "ReflectionPayload",
    "ToTNode",
]