"""基线布局生成（规则网格与交错网格）。

可行性语义（与 GA/PSO 初始化共用同一套管线，见
:mod:`wind_farm_opt.optimization.layout_common`）：

* 开始搜索前用真实多边形、机组直径与最小间距做容量预判，明显无解直接
  抛出 :class:`~wind_farm_opt.constraints.feasibility.LayoutInfeasibleError`；
* 规则格点不足时的随机补点消耗统一的有界尝试预算，不会无限循环；
* 成功返回前必须再次通过边界与间距终验（数量不足/越界/过密都会被拦截），
  失败时报告容量、可用面积与首要违规原因。
"""

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.feasibility import LayoutInfeasibleError
from ..constraints.spacing import compute_min_spacing_from_diameters
from .layout_common import (
    AttemptBudget,
    assess_or_raise,
    finalize_layout,
    random_fill,
)


def _regular_positions(
    boundary: SiteBoundary,
    n_turbines: int,
    min_spacing: float,
    n_rows: int,
    n_cols: int,
    staggered: bool,
) -> list[np.ndarray]:
    """生成规则（网格或交错）格点，只保留落在真实多边形内的点。

    间距按场地实际可用范围计算，不人为压缩到最小间距以下：
    规则格点放不下的部分交给有界随机补点，而不是制造系统性违规。
    """
    x_min, x_max = boundary.x_min, boundary.x_max
    y_min, y_max = boundary.y_min, boundary.y_max

    margin = min_spacing * 0.5
    x_range = x_max - x_min - 2 * margin
    y_range = y_max - y_min - 2 * margin

    # 格点间距永不小于最小间距：规则格点放不下的点位会落到多边形外被
    # 过滤，再由有界随机补点补齐，避免制造系统性的间距违规。
    if staggered:
        min_dx = min_spacing
        min_dy = min_spacing * np.sqrt(3.0) / 2.0  # 交错排最近邻 = s
    else:
        min_dx = min_spacing
        min_dy = min_spacing

    if n_cols > 1:
        spacing_x = max(x_range / (n_cols - 1), min_dx)
    else:
        spacing_x = 0.0
    if n_rows > 1:
        spacing_y = max(y_range / (n_rows - 1), min_dy)
    else:
        spacing_y = 0.0

    start_x = x_min + margin + (x_range - spacing_x * (n_cols - 1)) / 2.0
    start_y = y_min + margin + (y_range - spacing_y * (n_rows - 1)) / 2.0

    positions: list[np.ndarray] = []
    count = 0
    for row in range(n_rows):
        offset = spacing_x / 2.0 if (staggered and row % 2 == 1) else 0.0
        for col in range(n_cols):
            if count >= n_turbines:
                break
            x = start_x + col * spacing_x + offset
            y = start_y + row * spacing_y
            pos = np.array([x, y])
            if boundary.contains_point(pos):
                positions.append(pos)
                count += 1

    return positions


def _generate_regular_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float,
    rng: np.random.Generator,
    budget: AttemptBudget,
    staggered: bool,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """网格/交错布局共用的有界生成流程。"""
    min_spacing = compute_min_spacing_from_diameters(rotor_diameters, min_multiple)

    # 1) 开始前容量预判：明显无解立即报告，不进入任何随机/修复循环
    feasibility = assess_or_raise(
        boundary, n_turbines, rotor_diameters, min_multiple=min_multiple
    )

    # 2) 规则格点
    if staggered:
        n_rows = max(1, int(np.sqrt(n_turbines)))
    else:
        n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    positions = _regular_positions(
        boundary, n_turbines, min_spacing, n_rows, n_cols, staggered
    )

    # 3) 格点不足：统一预算内随机补点（预算耗尽即停，绝不死循环）
    if len(positions) < n_turbines:
        random_fill(positions, n_turbines, boundary, min_spacing, rng, budget)

    positions = np.asarray(positions, dtype=np.float64)

    # 4) 统一终验闸门：数量/边界/间距全部通过才交付，否则报告并失败
    return finalize_layout(
        positions,
        boundary,
        n_turbines,
        min_spacing,
        feasibility,
        rng,
        budget,
    )


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
    budget: AttemptBudget | None = None,
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
    budget : Optional[AttemptBudget]
        统一尝试预算；None 时使用默认预算

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)，保证全部位于场地内且间距达标

    Raises
    ------
    LayoutInfeasibleError
        容量预判无解，或在预算内无法生成通过终验的布局
    """
    if rng is None:
        rng = np.random.default_rng()
    if budget is None:
        budget = AttemptBudget()

    return _generate_regular_layout(
        boundary=boundary,
        n_turbines=n_turbines,
        rotor_diameters=rotor_diameters,
        min_multiple=min_multiple,
        rng=rng,
        budget=budget,
        staggered=False,
        aspect_ratio=aspect_ratio,
    )


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
    budget: AttemptBudget | None = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    返回值与异常语义同 :func:`generate_grid_layout`：任何成功返回都已
    通过数量、边界与间距终验；失败抛出 :class:`LayoutInfeasibleError`。

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
    budget : Optional[AttemptBudget]
        统一尝试预算；None 时使用默认预算
    """
    if rng is None:
        rng = np.random.default_rng()
    if budget is None:
        budget = AttemptBudget()

    return _generate_regular_layout(
        boundary=boundary,
        n_turbines=n_turbines,
        rotor_diameters=rotor_diameters,
        min_multiple=min_multiple,
        rng=rng,
        budget=budget,
        staggered=True,
    )
