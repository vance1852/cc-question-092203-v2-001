"""布局可行性处理。

为规则布局与优化算法初始化提供统一的可行性语义：

1. ``assess_feasibility`` —— 开始搜索前利用真实多边形外接矩形与最小间距，
   通过网格单元装填上界判断“明显无解”的组合；
2. ``AttemptBudget`` —— 随机补点与间距修复共享同一套尝试预算，任何路径
   都不会无限循环；
3. ``validate_layout`` —— 所有成功返回前的统一终态校验（边界 + 间距），
   失败时通过 ``FeasibilityError`` 报告容量、可用面积与首要违规原因；
4. ``build_feasible_layout`` / ``repair_layout`` —— 带预算的随机补点与
   推开修复，只返回完整且可行的布局，绝不返回部分布局。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .boundary import SiteBoundary

# 原因代码
REASON_OK = "ok"
REASON_INVALID_REQUEST = "invalid_request"
REASON_SITE_DEGENERATE = "site_degenerate"
REASON_CAPACITY_EXCEEDED = "capacity_exceeded"
REASON_SAMPLING_BUDGET = "sampling_budget_exhausted"
REASON_REPAIR_SPACING = "repair_failed_spacing"
REASON_REPAIR_BOUNDARY = "repair_failed_boundary"

# 面积退化容差（m²）
_AREA_EPS = 1.0
# 构造确定性网格时留出的相对余量，避免浮点误差使相邻点恰好等于最小间距
_GAP_REL_EPS = 1.0e-3


@dataclass
class FeasibilityReport:
    """布局可行性评估/校验结果。

    Attributes
    ----------
    feasible : 是否可行
    n_turbines : 请求的风机台数（容量）
    n_placed : 实际已放置的风机台数
    min_spacing : 最小间距 (m)
    site_area : 真实多边形可用面积 (m²)
    capacity_bound : 由外接矩形单元装填给出的容量上界（台）
    n_out_of_bounds : 越界风机数
    n_spacing_violations : 间距违规风机对数
    worst_pair : 最严重违规的风机对索引
    worst_distance : 最严重违规对的实际距离 (m)
    reason : 首要原因代码
    message : 面向用户的中文说明
    """

    feasible: bool
    n_turbines: int
    n_placed: int = 0
    min_spacing: float = 0.0
    site_area: float = 0.0
    capacity_bound: int = 0
    n_out_of_bounds: int = 0
    n_spacing_violations: int = 0
    worst_pair: Optional[tuple[int, int]] = None
    worst_distance: Optional[float] = None
    reason: str = REASON_OK
    message: str = ""

    def format_message(self) -> str:
        """格式化完整报告。"""
        lines = [self.message]
        lines.append(
            f"  请求容量: {self.n_turbines} 台, 已放置: {self.n_placed} 台, "
            f"容量上界: {self.capacity_bound} 台"
        )
        lines.append(
            f"  可用面积: {self.site_area / 1e6:.4f} km², "
            f"最小间距: {self.min_spacing:.1f} m"
        )
        if self.n_out_of_bounds > 0:
            lines.append(f"  越界风机: {self.n_out_of_bounds} 台")
        if self.n_spacing_violations > 0:
            pair = self.worst_pair
            dist = self.worst_distance
            lines.append(
                f"  间距违规: {self.n_spacing_violations} 对"
                + (
                    f"，最严重: 机组 {pair[0]} 与 {pair[1]} 间距 "
                    f"{dist:.1f} m < {self.min_spacing:.1f} m"
                    if pair is not None and dist is not None
                    else ""
                )
            )
        return "\n".join(lines)


class FeasibilityError(RuntimeError):
    """布局不可行时抛出，携带结构化报告。

    Attributes
    ----------
    report : FeasibilityReport
        失败报告（容量、可用面积、首要违规原因）
    partial_positions : Optional[list[np.ndarray]]
        随机补点失败时已放置的部分机位（仅供调用方补救，绝不是有效结果）
    """

    def __init__(
        self,
        report: FeasibilityReport,
        partial_positions: Optional[list] = None,
    ) -> None:
        self.report = report
        self.partial_positions = partial_positions
        super().__init__(report.format_message())


class AttemptBudget:
    """随机补点与修复共享的统一尝试预算。

    所有随机/迭代路径每消耗一次尝试调用 ``consume``，预算耗尽后立即
    停止，保证任何调用方都不可能无限循环。
    """

    def __init__(self, total: int) -> None:
        if total < 0:
            raise ValueError("尝试预算不能为负")
        self.total = int(total)
        self.remaining = int(total)

    def consume(self, n: int = 1) -> bool:
        """尝试消耗 n 次预算；预算不足时返回 False（预算为 0 时首次即失败）。"""
        if self.remaining < n:
            self.remaining -= n
            return False
        self.remaining -= n
        return True

    @property
    def exhausted(self) -> bool:
        return self.remaining <= 0


def default_fill_budget(n_turbines: int) -> int:
    """随机补点的默认总尝试次数（候选点抽样数）。"""
    return max(2000, 100 * max(1, n_turbines))


def default_repair_budget(n_turbines: int) -> int:
    """间距修复的默认迭代次数。"""
    return min(1000, max(200, 30 * max(1, n_turbines)))


# 随机补点的默认轮数：首轮使用确定性种子，后续轮放弃种子纯随机重试，
# 避免个别坏种子把本来可行的场地堵死；所有轮共享同一总预算。
DEFAULT_FILL_ROUNDS = 4


def capacity_upper_bound(boundary: SiteBoundary, min_spacing: float) -> int:
    """基于真实多边形外接矩形的容量上界。

    以外接矩形划分边长 ``a = s/√2`` 的正方形单元：同一单元内任意两点
    距离严格小于 ``s``，因此每个单元至多容纳一台风机。覆盖外接矩形的
    单元总数是多边形内可容纳机组数的严格上界，与多边形形状无关。

    Returns
    -------
    int
        可布置机组数的上界
    """
    width = boundary.x_max - boundary.x_min
    height = boundary.y_max - boundary.y_min

    # 乘 (1-ε) 保证同单元两点距离严格小于 min_spacing
    a = min_spacing / math.sqrt(2.0) * (1.0 - 1.0e-9)
    n_x = int(math.floor(width / a)) + 1
    n_y = int(math.floor(height / a)) + 1
    return n_x * n_y


def _base_report(
    boundary: SiteBoundary,
    n_turbines: int,
    min_spacing: float,
) -> FeasibilityReport:
    return FeasibilityReport(
        feasible=False,
        n_turbines=int(n_turbines),
        min_spacing=float(min_spacing),
        site_area=float(boundary.area),
        capacity_bound=capacity_upper_bound(boundary, min_spacing),
    )


def assess_feasibility(
    boundary: SiteBoundary,
    n_turbines: int,
    min_spacing: float,
) -> FeasibilityReport:
    """搜索前的明显无解判断。

    检查：请求合法性、多边形非退化、单元装填容量上界。只返回报告，
    不抛异常；调用方可使用 :func:`assert_request_feasible`。
    """
    report = _base_report(boundary, n_turbines, min_spacing)

    if n_turbines < 1:
        report.reason = REASON_INVALID_REQUEST
        report.message = f"风机台数必须为正整数，收到 {n_turbines}"
        return report

    if not np.isfinite(min_spacing) or min_spacing <= 0:
        report.reason = REASON_INVALID_REQUEST
        report.message = f"最小间距必须为正数，收到 {min_spacing}"
        return report

    if boundary.area < _AREA_EPS:
        report.reason = REASON_SITE_DEGENERATE
        report.message = (
            f"场地多边形退化（可用面积 {boundary.area:.2f} m²），"
            "无法布置任何风机"
        )
        return report

    if n_turbines > report.capacity_bound:
        report.reason = REASON_CAPACITY_EXCEEDED
        report.message = (
            f"场地容量明显不足：{n_turbines} 台风机在最小间距 "
            f"{min_spacing:.1f} m 下超过容量上界 {report.capacity_bound} 台，"
            "开始搜索前即可判定无解"
        )
        return report

    report.feasible = True
    report.n_placed = int(n_turbines)
    report.reason = REASON_OK
    report.message = "通过搜索前可行性检查"
    return report


def assert_request_feasible(
    boundary: SiteBoundary,
    n_turbines: int,
    min_spacing: float,
) -> FeasibilityReport:
    """搜索前可行性门禁，不可行时抛出 :class:`FeasibilityError`。"""
    report = assess_feasibility(boundary, n_turbines, min_spacing)
    if not report.feasible:
        raise FeasibilityError(report)
    return report


def validate_layout(
    boundary: SiteBoundary,
    positions: np.ndarray,
    min_spacing: float,
    n_turbines: Optional[int] = None,
) -> FeasibilityReport:
    """终态校验：台数完整、全部在真实多边形内、两两间距满足要求。

    Parameters
    ----------
    positions : (N, 2) 机位
    n_turbines : 请求台数；给定时还校验“不允许部分布局”
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_placed = int(positions.shape[0]) if positions.ndim == 2 else 0
    requested = n_turbines if n_turbines is not None else n_placed

    report = _base_report(boundary, requested, min_spacing)
    report.n_placed = n_placed

    if n_turbines is not None and n_placed != n_turbines:
        report.reason = REASON_SAMPLING_BUDGET
        report.message = (
            f"布局不完整：仅放置 {n_placed}/{n_turbines} 台风机，"
            "部分布局不能作为有效结果"
        )
        return report

    if n_placed == 0:
        if requested == 0:
            report.feasible = True
            report.reason = REASON_OK
            report.message = "空布局"
            return report
        report.reason = REASON_SAMPLING_BUDGET
        report.message = f"未放置任何风机（请求 {requested} 台）"
        return report

    inside = boundary.contains_all(positions)
    out_idx = np.flatnonzero(~inside)
    report.n_out_of_bounds = int(out_idx.size)

    worst_pair, worst_dist, n_violations = _worst_spacing_violation(
        positions, min_spacing
    )
    report.n_spacing_violations = n_violations
    report.worst_pair = worst_pair
    report.worst_distance = worst_dist

    if report.n_out_of_bounds > 0:
        report.reason = REASON_REPAIR_BOUNDARY
        report.message = (
            f"边界校验失败：{report.n_out_of_bounds} 台风机位于场地多边形之外"
            f"（首例: 机组 {int(out_idx[0])}）"
        )
        return report

    if n_violations > 0:
        report.reason = REASON_REPAIR_SPACING
        report.message = (
            f"间距校验失败：{n_violations} 对风机间距小于 "
            f"{min_spacing:.1f} m"
        )
        return report

    report.feasible = True
    report.reason = REASON_OK
    report.message = (
        f"布局通过校验：{n_placed} 台风机均在场地内且间距满足要求"
    )
    return report


def _worst_spacing_violation(
    positions: np.ndarray,
    min_spacing: float,
) -> tuple[Optional[tuple[int, int]], Optional[float], int]:
    """返回（最严重违规对, 最小距离, 违规对数），向量化实现。"""
    n = positions.shape[0]
    if n < 2:
        return None, None, 0

    # n 较大时分块计算，避免 n×n×2 临时数组过大
    worst_pair: Optional[tuple[int, int]] = None
    worst_dist = math.inf
    n_violations = 0
    chunk = 512

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        diff = positions[start:end, None, :] - positions[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # 只看下三角（j < i），每对只计一次
        for ii in range(start, end):
            row = dist[ii - start, :ii]
            if row.size == 0:
                continue
            bad = row < min_spacing
            n_violations += int(np.count_nonzero(bad))
            if bad.any():
                j = int(np.argmin(row))
                if row[j] < worst_dist:
                    worst_dist = float(row[j])
                    worst_pair = (j, ii)

    if worst_pair is None:
        return None, None, 0
    return worst_pair, worst_dist, n_violations


def _hex_lattice_candidates(
    boundary: SiteBoundary,
    min_spacing: float,
    rng: np.random.Generator,
    max_points: int = 100_000,
) -> Optional[np.ndarray]:
    """生成随机旋转/平移的六方点阵候选点（已打乱顺序）。

    六方点阵是最密排布：行间距 ``s·√3/2``、奇数行错位 ``s/2``。每轮随机
    补点使用一份新的随机朝向点阵，密集场地下的命中率远高于均匀随机。
    点阵规模超过 ``max_points`` 时返回 None（退化为纯均匀随机）。
    """
    gap = min_spacing * (1.0 + _GAP_REL_EPS)
    row_h = gap * math.sqrt(3.0) / 2.0

    cx = (boundary.x_min + boundary.x_max) / 2.0
    cy = (boundary.y_min + boundary.y_max) / 2.0
    hw = (boundary.x_max - boundary.x_min) / 2.0
    hh = (boundary.y_max - boundary.y_min) / 2.0

    # 六方点阵具有 60° 旋转对称，随机朝向取 [0, π/3) 即可
    theta = rng.uniform(0.0, math.pi / 3.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # 旋转后仍需覆盖外接矩形的范围
    rw = hw * abs(cos_t) + hh * abs(sin_t)
    rh = hw * abs(sin_t) + hh * abs(cos_t)

    nx = int(math.ceil(2.0 * rw / gap)) + 3
    ny = int(math.ceil(2.0 * rh / row_h)) + 3
    if nx * ny > max_points:
        return None

    ox = rng.uniform(0.0, gap)
    oy = rng.uniform(0.0, row_h)
    xs = ox - rw - gap + np.arange(nx) * gap
    ys = oy - rh - row_h + np.arange(ny) * row_h

    pts = np.empty((nx * ny, 2), dtype=np.float64)
    k = 0
    for j in range(ny):
        off = (gap / 2.0) if (j % 2 == 1) else 0.0
        for i in range(nx):
            pts[k, 0] = xs[i] + off
            pts[k, 1] = ys[j]
            k += 1

    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world = pts @ rot.T + np.array([cx, cy])
    rng.shuffle(world)
    return world


def random_fill(
    boundary: SiteBoundary,
    accepted: Sequence[np.ndarray],
    n_target: int,
    min_spacing: float,
    rng: np.random.Generator,
    budget: AttemptBudget,
    max_attempts: Optional[int] = None,
    lattice_candidates: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """在统一预算内把机位补满到 ``n_target``。

    候选点优先取自随机朝向的六方点阵（密集排布命中率高），点阵用完后
    退化为外接矩形内均匀随机抽样。每评估一个候选点消耗一次预算；只有
    落在真实多边形内且与所有已接受点间距达标的点才会被接受。预算耗尽
    （或达到本轮 ``max_attempts`` 上限）仍未补满时抛出
    :class:`FeasibilityError`，不返回部分布局。
    """
    positions = [np.asarray(p, dtype=np.float64).copy() for p in accepted]
    attempts_this_round = 0
    lattice_idx = 0
    n_lattice = 0 if lattice_candidates is None else len(lattice_candidates)

    def _failure_report(message: str) -> FeasibilityReport:
        report = validate_layout(
            boundary,
            np.asarray(positions, dtype=np.float64).reshape(-1, 2)
            if positions
            else np.zeros((0, 2)),
            min_spacing,
            n_turbines=n_target,
        )
        report.reason = REASON_SAMPLING_BUDGET
        report.message = message
        return report

    while len(positions) < n_target:
        if not budget.consume():
            raise FeasibilityError(
                _failure_report(
                    f"随机补点在 {budget.total} 次尝试预算内只放置了 "
                    f"{len(positions)}/{n_target} 台风机，场地可能已无足够的"
                    "合规机位"
                ),
                partial_positions=positions,
            )

        attempts_this_round += 1
        if max_attempts is not None and attempts_this_round > max_attempts:
            raise FeasibilityError(
                _failure_report(
                    f"本轮随机补点在 {max_attempts} 次尝试内只放置了 "
                    f"{len(positions)}/{n_target} 台风机"
                ),
                partial_positions=positions,
            )

        if lattice_idx < n_lattice:
            cand = lattice_candidates[lattice_idx]
            lattice_idx += 1
        else:
            cand = np.array(
                [
                    rng.uniform(boundary.x_min, boundary.x_max),
                    rng.uniform(boundary.y_min, boundary.y_max),
                ],
                dtype=np.float64,
            )
        if not boundary.contains_point(cand):
            continue
        if positions:
            existing = np.asarray(positions, dtype=np.float64)
            if np.any(np.linalg.norm(existing - cand, axis=1) < min_spacing):
                continue
        positions.append(np.asarray(cand, dtype=np.float64))

    return positions


def repair_layout(
    boundary: SiteBoundary,
    positions: np.ndarray,
    min_spacing: float,
    rng: np.random.Generator,
    budget: AttemptBudget,
) -> np.ndarray:
    """在统一预算内推开过近机组并投影回多边形。

    每轮推开迭代消耗一次预算；预算耗尽或最终校验不通过时抛出
    :class:`FeasibilityError`，绝不返回带违规的布局。
    """
    positions = np.asarray(positions, dtype=np.float64).copy()
    n = positions.shape[0]

    while True:
        report = validate_layout(boundary, positions, min_spacing)
        if report.feasible:
            return positions

        if not budget.consume():
            # 用最后一次校验结果生成失败报告
            final = validate_layout(boundary, positions, min_spacing)
            if final.n_out_of_bounds > 0:
                final.reason = REASON_REPAIR_BOUNDARY
                final.message = (
                    f"边界修复失败：迭代预算耗尽后仍有 "
                    f"{final.n_out_of_bounds} 台机组越界"
                )
            else:
                final.reason = REASON_REPAIR_SPACING
                final.message = (
                    f"间距修复失败：迭代预算耗尽后仍有 "
                    f"{final.n_spacing_violations} 对机组间距不足"
                )
            raise FeasibilityError(final)

        # 沿违规对连线互相推开
        for i in range(n):
            for j in range(i + 1, n):
                vec = positions[j] - positions[i]
                dist = float(np.linalg.norm(vec))
                if dist >= min_spacing:
                    continue
                if dist < 1e-12:
                    vec = rng.standard_normal(2)
                    dist = float(np.linalg.norm(vec))
                vec_norm = vec / dist
                push = (min_spacing - dist) / 2.0 + 1.0e-6
                positions[i] -= vec_norm * push
                positions[j] += vec_norm * push

        # 越界机组投影到多边形边界，小扰动后再次确认
        for k in range(n):
            if boundary.contains_point(positions[k]):
                continue
            positions[k] = boundary.project_to_boundary(positions[k])
            positions[k] += rng.uniform(-5.0, 5.0, 2)
            if not boundary.contains_point(positions[k]):
                positions[k] = boundary.project_to_boundary(positions[k])


def force_complete_and_repair(
    boundary: SiteBoundary,
    partial: Sequence[np.ndarray],
    n_target: int,
    min_spacing: float,
    rng: np.random.Generator,
    budget: AttemptBudget,
) -> np.ndarray:
    """把部分布局补齐到完整台数后用推开修复，全程共享同一预算。

    补点阶段不再要求间距达标（允许临时过密），只要求落在真实多边形内；
    随后由 :func:`repair_layout` 在剩余预算内把过近机组推开。这是对
    “合规抽样命中率极低”场地的兜底：修复失败同样抛出
    :class:`FeasibilityError`，绝不会把部分/违规布局当作结果。
    """
    positions = [np.asarray(p, dtype=np.float64).copy() for p in partial]

    while len(positions) < n_target:
        if not budget.consume():
            report = validate_layout(
                boundary,
                np.asarray(positions, dtype=np.float64).reshape(-1, 2),
                min_spacing,
                n_turbines=n_target,
            )
            report.reason = REASON_SAMPLING_BUDGET
            report.message = (
                f"补齐机位的尝试预算耗尽，仅放置 {len(positions)}/{n_target} 台"
            )
            raise FeasibilityError(report, partial_positions=positions)

        cand = np.array(
            [
                rng.uniform(boundary.x_min, boundary.x_max),
                rng.uniform(boundary.y_min, boundary.y_max),
            ],
            dtype=np.float64,
        )
        if boundary.contains_point(cand):
            positions.append(cand)

    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    return repair_layout(boundary, positions, min_spacing, rng, budget)


def build_feasible_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    min_spacing: float,
    rng: Optional[np.random.Generator] = None,
    seed_positions: Optional[np.ndarray] = None,
    attempt_budget: Optional[int] = None,
    n_rounds: int = DEFAULT_FILL_ROUNDS,
) -> np.ndarray:
    """统一的可行布局构造流水线（搜索前门禁 → 种子 → 随机补点 → 修复 → 终态校验）。

    只返回形状 (n_turbines, 2) 且通过边界/间距校验的布局；任何阶段失败
    都抛出带 :class:`FeasibilityReport` 的 :class:`FeasibilityError`。

    随机补点分为若干轮，所有轮共享同一 ``attempt_budget``：首轮保留确定
    性种子，后续轮放弃种子纯随机重试——个别坏种子可能把本来可行的场地
    堵死，重试轮避免误报无解，同时总预算不变、绝不无限循环。

    Parameters
    ----------
    seed_positions
        确定性候选点（如规则网格点）。越界或彼此冲突的种子会被丢弃，
        缺额由随机补点补齐——不会因为种子冲突而返回违规布局。
    attempt_budget
        随机补点与修复共享的总尝试预算；None 时使用
        :func:`default_fill_budget`。
    n_rounds
        随机补点轮数（>=1），首轮使用种子，其余轮纯随机。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1) 搜索前明显无解判断
    assert_request_feasible(boundary, n_turbines, min_spacing)

    # 2) 过滤种子点：必须在真实多边形内，且彼此间距达标
    accepted: list[np.ndarray] = []
    if seed_positions is not None:
        seeds = np.asarray(seed_positions, dtype=np.float64).reshape(-1, 2)
        for cand in seeds:
            if len(accepted) >= n_turbines:
                break
            if not boundary.contains_point(cand):
                continue
            if accepted:
                existing = np.asarray(accepted, dtype=np.float64)
                if np.any(
                    np.linalg.norm(existing - cand, axis=1)
                    < min_spacing * (1.0 + _GAP_REL_EPS)
                ):
                    continue
            accepted.append(cand.copy())

    # 3) 多轮随机补点（所有轮与修复共享同一总预算）
    budget_total = (
        int(attempt_budget)
        if attempt_budget is not None
        else default_fill_budget(n_turbines)
    )
    budget = AttemptBudget(budget_total)
    n_rounds = max(1, int(n_rounds))

    best_report: Optional[FeasibilityReport] = None

    for round_idx in range(n_rounds):
        if budget.remaining <= 0:
            break
        # 首轮保留种子；后续轮放弃种子纯随机重试
        round_seeds = accepted if round_idx == 0 else []
        share = max(1, budget.remaining // (n_rounds - round_idx))

        try:
            lattice = _hex_lattice_candidates(boundary, min_spacing, rng)
            positions_list = random_fill(
                boundary,
                round_seeds,
                n_turbines,
                min_spacing,
                rng,
                budget,
                max_attempts=share,
                lattice_candidates=lattice,
            )
        except FeasibilityError as e:
            if best_report is None or e.report.n_placed > best_report.n_placed:
                best_report = e.report

            # 兜底：本轮已放置部分合规机位时，补齐到完整台数再推开修复
            if e.partial_positions:
                try:
                    completed = force_complete_and_repair(
                        boundary,
                        e.partial_positions,
                        n_turbines,
                        min_spacing,
                        rng,
                        budget,
                    )
                    completed_report = validate_layout(
                        boundary, completed, min_spacing, n_turbines
                    )
                    if completed_report.feasible:
                        return completed
                    if (
                        best_report is None
                        or completed_report.n_placed > best_report.n_placed
                    ):
                        best_report = completed_report
                except FeasibilityError as e2:
                    if (
                        best_report is None
                        or e2.report.n_placed > best_report.n_placed
                    ):
                        best_report = e2.report
            continue

        positions = np.asarray(positions_list, dtype=np.float64).reshape(-1, 2)

        # 4) 终态校验；若存在意外违规则用剩余预算修复后再校验
        report = validate_layout(boundary, positions, min_spacing, n_turbines)
        if report.feasible:
            return positions
        try:
            positions = repair_layout(
                boundary, positions, min_spacing, rng, budget
            )
        except FeasibilityError as e:
            if best_report is None or e.report.n_placed > best_report.n_placed:
                best_report = e.report
            continue
        report = validate_layout(boundary, positions, min_spacing, n_turbines)
        if report.feasible:
            return positions
        if best_report is None or report.n_placed > best_report.n_placed:
            best_report = report

    # 所有轮均未成功：报告最佳（放置最多）一轮的失败原因
    if best_report is None:
        # 预算为 0 时没有任何一轮真正执行
        best_report = validate_layout(
            boundary, np.zeros((0, 2)), min_spacing, n_turbines
        )
        best_report.reason = REASON_SAMPLING_BUDGET
        best_report.message = (
            f"随机补点预算为 {budget_total}，未放置任何风机"
            f"（请求 {n_turbines} 台）"
        )
    best_report.message += (
        f"（共 {n_rounds} 轮补点，共享 {budget_total} 次尝试预算）"
    )
    raise FeasibilityError(best_report)
