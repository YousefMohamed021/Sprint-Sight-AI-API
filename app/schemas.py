from pydantic import BaseModel, Field
from enum import Enum

# ── Request schema ─────────────────────────────────────────────────────────────

class SprintInput(BaseModel):
    """All fields required to predict sprint productivity and quality."""

    # ── Sprint-level features ──────────────────────────────────────────────────
    plan_duration_hours: float = Field(
        ..., gt=0,
        description="Planned sprint duration in hours (e.g. 336 for 2 weeks)",
        example=336.0
    )
    no_issues: int = Field(
        ..., ge=1,
        description="Number of issues committed to this sprint",
        example=12
    )
    no_team_members: int = Field(
        ..., ge=1,
        description="Number of developers assigned to this sprint",
        example=5
    )

    # ── Issue-level features (aggregated across all sprint issues) ─────────────
    no_components: float = Field(
        default=5.0, ge=0,
        description="Total number of components assigned across all issues in the sprint",
        example=18.0
    )
    no_comments: float = Field(
        default=10.0, ge=0,
        description="Total number of comments across all issues in the sprint",
        example=36.0
    )
    no_description_changes: float = Field(
        default=0.0, ge=0,
        description="Total number of times issue descriptions were changed across the sprint",
        example=4.0
    )
    no_priority_changes: float = Field(
        default=0.0, ge=0,
        description="Total number of times issue priorities were changed across the sprint",
        example=2.0
    )
    no_fix_version_changes: float = Field(
        default=0.0, ge=0,
        description="Total number of times fix versions were changed across the sprint",
        example=1.0
    )

    # ── Issue Type Counts ──
    type_bug_count: int = Field(
        default=0, ge=0, 
        description="Total number of Bug issues in the sprint",
        example=5
    )
    type_suggestion_count: int = Field(
        default=0, ge=0, 
        description="Total number of Suggestion issues in the sprint",
        example=4
    )
    type_support_request_count: int = Field(
        default=0, ge=0, 
        description="Total number of Support Request issues in the sprint",
        example=3
    )

    # ── Issue Priority Counts ──
    priority_blocker_count: int = Field(
        default=0, ge=0, 
        description="Total number of Blocker priority issues",
        example=0
    )
    priority_critical_count: int = Field(
        default=0, ge=0, 
        description="Total number of Critical priority issues",
        example=1
    )
    priority_high_count: int = Field(
        default=0, ge=0, 
        description="Total number of High priority issues",
        example=2
    )
    priority_highest_count: int = Field(
        default=0, ge=0, 
        description="Total number of Highest priority issues",
        example=1
    )
    priority_low_count: int = Field(
        default=0, ge=0, 
        description="Total number of Low priority issues",
        example=2
    )
    priority_major_count: int = Field(
        default=0, ge=0, 
        description="Total number of Major priority issues",
        example=4
    )
    priority_medium_count: int = Field(
        default=0, ge=0, 
        description="Total number of Medium priority issues",
        example=2
    )
    priority_minor_count: int = Field(
        default=0, ge=0, 
        description="Total number of Minor priority issues",
        example=0
    )
    priority_trivial_count: int = Field(
        default=0, ge=0, 
        description="Total number of Trivial priority issues",
        example=0
    )

    # ── Developer-level features (aggregated across all sprint developers) ─────
    no_distinct_actions: float = Field(
        default=15.0, ge=0,
        description="Total number of distinct action types performed by developers before sprint start",
        example=45.0
    )
    developer_activeness: float = Field(
        default=10.0, ge=0,
        description="Average developer activeness score (number of issues involved in past 3 months)",
        example=15.0
    )
    dev_prefer_bug_count: int = Field(default=0, ge=0, description="Number of developers whose preferred issue type is Bug")
    dev_prefer_na_count: int = Field(default=0, ge=0, description="Number of developers with no preferred issue type (Na)")
    dev_prefer_subtask_count: int = Field(default=0, ge=0, description="Number of developers whose preferred issue type is Sub-task")
    dev_prefer_suggestion_count: int = Field(default=0, ge=0, description="Number of developers whose preferred issue type is Suggestion")

    # ── Text feature ───────────────────────────────────────────────────────────
    sprint_text: str = Field(
        ..., min_length=10,
        description="Combined text of all issue summaries and descriptions in this sprint",
        example="Fix login timeout bug affecting mobile users. Improve dashboard load performance. Add export to CSV feature for reports."
    )

    class Config:
        extra = "ignore"
        json_schema_extra = {
            "example": {
                "plan_duration_hours": 336.0,
                "no_issues": 12,
                "no_team_members": 5,
                "no_components": 18.0,
                "no_comments": 36.0,
                "no_description_changes": 4.0,
                "no_priority_changes": 2.0,
                "no_fix_version_changes": 1.0,
                "type_bug_count": 5,
                "type_suggestion_count": 4,
                "type_support_request_count": 3,
                "priority_blocker_count": 0,
                "priority_critical_count": 1,
                "priority_high_count": 2,
                "priority_highest_count": 1,
                "priority_low_count": 2,
                "priority_major_count": 4,
                "priority_medium_count": 2,
                "priority_minor_count": 0,
                "priority_trivial_count": 0,
                "no_distinct_actions": 45.0,
                "developer_activeness": 15.0,
                "dev_prefer_bug_count": 3,
                "dev_prefer_na_count": 1,
                "dev_prefer_subtask_count": 1,
                "dev_prefer_suggestion_count": 0,
                "sprint_text": "Fix login timeout bug affecting mobile users. Improve dashboard load performance. Add export to CSV feature for reports."
            }
        }


# ── Response schemas ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Prediction results returned by the /predict endpoint."""

    productivity: float = Field(
        description="Predicted completion ratio — fraction of committed issues expected to be completed. "
                    "e.g. 0.85 means 85% of committed issues will be completed."
    )
    quality: float = Field(
        description="Predicted reopen ratio — fraction of completed issues expected to be reopened. "
                    "e.g. 0.12 means 12% of completed issues may require rework. Lower is better."
    )
    productivity_label: str = Field(
        description="Human-readable interpretation of the productivity score"
    )
    quality_label: str = Field(
        description="Human-readable interpretation of the quality score"
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    embedding_model: str