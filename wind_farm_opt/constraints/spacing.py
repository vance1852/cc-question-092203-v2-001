"""风机间距约束。"""

import numpy as np


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足最小距离要求。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    min_distance : float
        最小允许间距 (m)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)，M 为违规对数
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    if n < 2:
        return True, np.zeros((0, 2), dtype=int)

    violations = []
    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = positions[start:end, None, :] - positions[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        for ii in range(start, end):
            bad = np.flatnonzero(dist[ii - start, :ii] < min_distance)
            for j in bad:
                violations.append([int(j), ii])

    if violations:
        return False, np.array(violations, dtype=int)
    return True, np.zeros((0, 2), dtype=int)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算最小间距（取最大直径的倍数）。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数（相对于转子直径）

    Returns
    -------
    float
        最小间距 (m)
    """
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N, 2)

    Returns
    -------
    np.ndarray
        距离矩阵，形状为 (N, N)，对角线为 0
    """
    positions = np.asarray(positions, dtype=np.float64)
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=2)


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    boundary,
    rng: np.random.Generator | None = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动风机来满足最小间距约束。

    当有风机对间距不足时，将它们沿连线方向推开。与随机补点共享同一套
    尝试预算语义：迭代次数即预算，预算耗尽且校验仍不通过时抛出
    :class:`~wind_farm_opt.constraints.feasibility.FeasibilityError`
    （``RuntimeError`` 的子类），绝不返回带违规的布局。

    Parameters
    ----------
    positions : np.ndarray
        初始风机位置，形状为 (N, 2)
    min_distance : float
        最小间距 (m)
    boundary : SiteBoundary
        场地边界
    rng : Optional[np.random.Generator]
        随机数生成器
    max_iterations : int
        最大迭代次数（修复尝试预算）

    Returns
    -------
    np.ndarray
        调整后的风机位置，保证通过边界与间距校验

    Raises
    ------
    FeasibilityError
        预算耗尽仍无法满足约束时抛出，报告中包含首要违规原因
    """
    from .feasibility import AttemptBudget, repair_layout

    if rng is None:
        rng = np.random.default_rng()

    budget = AttemptBudget(max_iterations)
    return repair_layout(boundary, positions, min_distance, rng, budget)
