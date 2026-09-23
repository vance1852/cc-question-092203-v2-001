"""约束检查模块。"""

from .boundary import SiteBoundary
from .feasibility import (
    AttemptBudget,
    FeasibilityError,
    FeasibilityReport,
    assess_feasibility,
    assert_request_feasible,
    build_feasible_layout,
    capacity_upper_bound,
    validate_layout,
)
from .spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    compute_pairwise_distances,
    enforce_min_spacing,
)

__all__ = [
    "SiteBoundary",
    "AttemptBudget",
    "FeasibilityError",
    "FeasibilityReport",
    "assess_feasibility",
    "assert_request_feasible",
    "build_feasible_layout",
    "capacity_upper_bound",
    "validate_layout",
    "check_min_spacing",
    "compute_min_spacing_from_diameters",
    "compute_pairwise_distances",
    "enforce_min_spacing",
]
