"""基线布局生成（规则网格 / 交错网格）。

两种布局共用 :mod:`wind_farm_opt.constraints.feasibility` 的统一可行性
语义：

* 开始搜索前用真实多边形、转子直径与最小间距判断明显无解的组合；
* 确定性网格点只作为“种子”，越界或间距冲突的点一律丢弃，缺额由带
  统一尝试预算的随机补点补齐；
* 所有成功返回的布局都再次通过边界与间距校验；
* 失败时抛出 :class:`FeasibilityError`，报告容量、可用面积与首要违规
  原因，绝不返回部分布局或带违规的布局。
"""

import math
from typing import Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.feasibility import (
    FeasibilityError,
    assert_request_feasible,
    build_feasible_layout,
)
from ..constraints.spacing import compute_min_spacing_from_diameters

# 构造网格时间距留出的相对余量，避免浮点误差导致相邻点恰好等于最小间距
_GAP_REL_EPS = 1.0e-3


def _plan_grid_dims(
    n_turbines: int,
    aspect_ratio: float,
    width: float,
    height: float,
    spacing_x: float,
    spacing_y: float,
    staggered: bool,
) -> tuple[int, int]:
    """选择 (行数, 列数)，优先保证网格在间距约束下能放进外接矩形。

    遍历所有 rows×cols >= n_turbines 的组合，筛出满足跨度要求的方案，
    取与目标纵横比最接近者；若都不满足，返回纵横比最接近的方案，
    放不下/冲突的点会在种子过滤阶段被丢弃并由随机补点兜底。
    """
    best_fit: Optional[tuple[float, int, int]] = None
    best_any: Optional[tuple[float, int, int]] = None

    for rows in range(1, n_turbines + 1):
        cols = int(math.ceil(n_turbines / rows))
        ratio = cols / rows
        score = abs(math.log(ratio / max(aspect_ratio, 1e-12)))

        span_x = (cols - 1) * spacing_x + (spacing_x / 2.0 if staggered else 0.0)
        span_y = (rows - 1) * spacing_y
        fits = span_x <= width + 1e-9 and span_y <= height + 1e-9

        if best_any is None or score < best_any[0]:
            best_any = (score, rows, cols)
        if fits and (best_fit is None or score < best_fit[0]):
            best_fit = (score, rows, cols)

    chosen = best_fit if best_fit is not None else best_any
    return chosen[1], chosen[2]


def _grid_seed_positions(
    boundary: SiteBoundary,
    n_turbines: int,
    min_spacing: float,
    aspect_ratio: float,
    staggered: bool,
) -> np.ndarray:
    """生成确定性网格候选点（尚未过滤，可能越界或冲突）。

    网格间距不小于 min_spacing（含少量相对余量），因此保留下来的种子
    点彼此间距必然达标；越界点由统一的种子过滤丢弃。
    """
    gap = min_spacing * (1.0 + _GAP_REL_EPS)
    width = boundary.x_max - boundary.x_min
    height = boundary.y_max - boundary.y_min

    n_rows, n_cols = _plan_grid_dims(
        n_turbines, aspect_ratio, width, height, gap, gap, staggered
    )

    span_x = (n_cols - 1) * gap + (gap / 2.0 if staggered else 0.0)
    span_y = (n_rows - 1) * gap
    start_x = boundary.x_min + (width - span_x) / 2.0
    start_y = boundary.y_min + (height - span_y) / 2.0

    seeds = []
    count = 0
    for row in range(n_rows):
        offset = gap / 2.0 if (staggered and row % 2 == 1) else 0.0
        for col in range(n_cols):
            if count >= n_turbines:
                break
            seeds.append(
                [start_x + col * gap + offset, start_y + row * gap]
            )
            count += 1

    return np.asarray(seeds, dtype=np.float64).reshape(-1, 2)


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
    attempt_budget: int | None = None,
) -> np.ndarray:
    """生成规则网格布局作为优化基线。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器
    attempt_budget : Optional[int]
        随机补点与修复共享的尝试预算；None 时使用默认值

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)，保证通过边界与间距校验

    Raises
    ------
    FeasibilityError
        明显无解（容量上界不足等）或预算耗尽仍不可行时抛出，
        报告中包含容量、可用面积与首要违规原因
    """
    if rng is None:
        rng = np.random.default_rng()

    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    # 搜索前明显无解判断（真实多边形 + 最小间距）
    assert_request_feasible(boundary, n_turbines, min_spacing)

    seeds = _grid_seed_positions(
        boundary, n_turbines, min_spacing, aspect_ratio, staggered=False
    )

    return build_feasible_layout(
        boundary,
        n_turbines,
        min_spacing,
        rng=rng,
        seed_positions=seeds,
        attempt_budget=attempt_budget,
    )


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
    attempt_budget: int | None = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数
    dominant_direction : float
        主风向（度），用于确定交错方向
    rng : Optional[np.random.Generator]
        随机数生成器
    attempt_budget : Optional[int]
        随机补点与修复共享的尝试预算；None 时使用默认值

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)，保证通过边界与间距校验

    Raises
    ------
    FeasibilityError
        明显无解（容量上界不足等）或预算耗尽仍不可行时抛出，
        报告中包含容量、可用面积与首要违规原因
    """
    if rng is None:
        rng = np.random.default_rng()

    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    # 搜索前明显无解判断（真实多边形 + 最小间距）
    assert_request_feasible(boundary, n_turbines, min_spacing)

    # 交错方向沿主风向的垂直方向排布行；当前实现按场地坐标轴交错，
    # dominant_direction 保留用于后续按风向旋转网格。
    _ = dominant_direction

    seeds = _grid_seed_positions(
        boundary, n_turbines, min_spacing, aspect_ratio=1.0, staggered=True
    )

    return build_feasible_layout(
        boundary,
        n_turbines,
        min_spacing,
        rng=rng,
        seed_positions=seeds,
        attempt_budget=attempt_budget,
    )
