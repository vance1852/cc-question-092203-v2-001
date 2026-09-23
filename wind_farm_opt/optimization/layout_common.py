"""布局生成的统一可行性管线。

网格/交错布局的随机补点与修复、GA/PSO 的随机初始化都经由本模块，
保证三处共享同一套终止语义：

* 开始前调用 :func:`~wind_farm_opt.constraints.feasibility.assess_capacity`
  排除明显无解的组合；
* 随机补点、随机重启、迭代修复都消耗同一个有界的 :class:`AttemptBudget`，
  任何路径都不可能无限循环；
* 成功返回前必须通过
  :func:`~wind_farm_opt.constraints.feasibility.validate_layout`
  的数量/边界/间距终验，失败时抛出携带容量、可用面积与首要违规原因的
  :class:`~wind_farm_opt.constraints.feasibility.LayoutInfeasibleError`，
  绝不把部分布局或非法机位当作有效基线交付。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..constraints.feasibility import (
    FeasibilityReport,
    LayoutInfeasibleError,
    assess_capacity,
    validate_layout,
)
from ..constraints.spacing import (
    check_min_spacing,
    enforce_min_spacing,
)


@dataclass(frozen=True)
class AttemptBudget:
    """统一的尝试预算（所有随机/迭代过程的硬上限）。

    Attributes
    ----------
    fill_draws : int
        随机补点（或单次随机布局采样）允许的拒绝采样总次数
    repair_iterations : int
        间距修复（推挤迭代）的最大次数
    random_restarts : int
        纯随机初始化（GA/PSO）的最大重启轮数
    """

    fill_draws: int = 2000
    repair_iterations: int = 500
    random_restarts: int = 30


def random_fill(
    positions: list[np.ndarray],
    n_target: int,
    boundary,
    min_spacing: float,
    rng: np.random.Generator,
    budget: AttemptBudget,
) -> int:
    """在统一预算内把随机合法机位补入 ``positions``。

    每次拒绝采样只消耗一个预算单位；预算耗尽即返回，调用方必须通过
    终验确认机位数，不得把不足的布局当作成功。

    Returns
    -------
    int
        实际消耗的采样次数
    """
    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    draws = 0
    while len(positions) < n_target and draws < budget.fill_draws:
        cand = np.array([
            rng.uniform(x_min, x_max),
            rng.uniform(y_min, y_max),
        ])
        draws += 1

        if not boundary.contains_point(cand):
            continue

        if positions:
            existing = np.asarray(positions, dtype=np.float64)
            if np.any(np.linalg.norm(existing - cand, axis=1) < min_spacing - 1e-9):
                continue
        positions.append(cand)

    return draws


def repair_or_raise(
    positions: np.ndarray,
    min_spacing: float,
    boundary,
    rng: np.random.Generator,
    budget: AttemptBudget,
) -> np.ndarray:
    """在有界迭代内修复间距/边界，失败即抛出（不吞异常）。"""
    return enforce_min_spacing(
        positions,
        min_spacing,
        boundary,
        rng,
        max_iterations=budget.repair_iterations,
    )


def finalize_layout(
    positions: np.ndarray,
    boundary,
    n_turbines: int,
    min_spacing: float,
    feasibility: FeasibilityReport,
    rng: np.random.Generator,
    budget: AttemptBudget,
    rated_power_kw: Optional[float] = None,
) -> np.ndarray:
    """统一终验闸门：先验一次，非法则给一次有界修复机会，再验。

    任何成功返回都保证机位数正确、全部位于真实多边形内且两两间距达标。
    失败抛出 :class:`LayoutInfeasibleError`（容量、面积、首要违规原因）。
    """
    positions = np.asarray(positions, dtype=np.float64)

    validation = validate_layout(positions, boundary, n_turbines, min_spacing)
    if not validation.valid:
        # 只有机位齐全时修复才有意义；数量不足直接报告
        if positions.shape[0] == n_turbines:
            try:
                positions = repair_or_raise(
                    positions, min_spacing, boundary, rng, budget
                )
            except RuntimeError:
                pass
            validation = validate_layout(
                positions, boundary, n_turbines, min_spacing
            )

    if not validation.valid:
        raise LayoutInfeasibleError.from_reports(
            feasibility=feasibility,
            validation=validation,
            rated_power_kw=rated_power_kw,
        )

    return positions


def assess_or_raise(
    boundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    min_spacing: Optional[float] = None,
    rated_power_kw: Optional[float] = None,
) -> FeasibilityReport:
    """开始搜索前的容量预判，明显无解时直接抛出统一异常。"""
    feasibility = assess_capacity(
        boundary,
        n_turbines,
        rotor_diameters,
        min_multiple=min_multiple,
        min_spacing=min_spacing,
    )
    if not feasibility.feasible:
        raise LayoutInfeasibleError.from_reports(
            feasibility=feasibility,
            rated_power_kw=rated_power_kw,
        )
    return feasibility


def generate_random_feasible_layout(
    boundary,
    n_turbines: int,
    min_spacing: float,
    rng: np.random.Generator,
    budget: Optional[AttemptBudget] = None,
    feasibility: Optional[FeasibilityReport] = None,
    rotor_diameters: Optional[np.ndarray] = None,
) -> np.ndarray:
    """在统一预算内生成一个完全合法的随机布局（GA/PSO 初始化共用）。

    每轮重启在预算内采样候选点；间距不足时给一次有界修复机会；
    终验通过才返回。预算耗尽仍无合法布局时抛出统一异常。
    """
    if budget is None:
        budget = AttemptBudget()

    if feasibility is None:
        if rotor_diameters is None:
            rotor_diameters = np.full(n_turbines, min_spacing / 5.0)
        feasibility = assess_capacity(
            boundary,
            n_turbines,
            np.asarray(rotor_diameters, dtype=np.float64),
            min_spacing=min_spacing,
        )
    if not feasibility.feasible:
        raise LayoutInfeasibleError.from_reports(feasibility=feasibility)

    last_validation = None

    for _ in range(budget.random_restarts):
        candidates = boundary.try_sample_points(
            n_turbines, rng, max_total_attempts=budget.fill_draws
        )
        if candidates.shape[0] < n_turbines:
            continue

        valid, _ = check_min_spacing(candidates, min_spacing)
        if not valid:
            try:
                candidates = repair_or_raise(
                    candidates, min_spacing, boundary, rng, budget
                )
            except RuntimeError:
                continue

        validation = validate_layout(
            candidates, boundary, n_turbines, min_spacing
        )
        if validation.valid:
            return candidates
        last_validation = validation

    raise LayoutInfeasibleError.from_reports(
        feasibility=feasibility,
        validation=last_validation,
    )
