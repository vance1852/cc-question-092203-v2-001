"""布局可行性预判与统一校验。

本模块为所有布局生成路径（规则网格、交错网格、GA/PSO 初始化与修复）
提供统一的可行性语义：

1. :func:`assess_capacity` —— 搜索开始前，基于**真实多边形**、机组直径
   与最小间距，用严格的几何必要条件判定“明显无解”的组合；
2. :func:`validate_layout` —— 任何布局成功返回前的统一终验闸门
   （机位数量、边界、间距）；
3. :class:`LayoutInfeasibleError` —— 统一的失败异常，携带装机容量、
   可用面积与首要违规原因，绝不允许把部分/非法布局当作有效结果返回。
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .spacing import check_min_spacing

# 表示某条几何判据不给出有限上界（跨度足够大时直径判据即如此）
_NO_BOUND = 10 ** 9


@dataclass
class FeasibilityReport:
    """容量可行性预判报告。

    ``max_turbines_bound`` 是由几何必要条件给出的机位数量**上界**：
    超过它一定无解，但不超过它仅表示“通过预判”，仍需实际搜索与终验确认。

    Attributes
    ----------
    feasible : bool
        是否通过预判（False 表示可严格证明无解）
    n_turbines : int
        请求的风机台数
    min_spacing : float
        最小间距 (m)
    rotor_diameter_max : float
        最大转子直径 (m)
    site_area : float
        场地多边形面积 (m²)
    site_perimeter : float
        场地多边形周长 (m)
    max_turbines_bound : int
        几何必要条件给出的机位上界
    reason : Optional[str]
        无解时的首要原因
    """

    feasible: bool
    n_turbines: int
    min_spacing: float
    rotor_diameter_max: float
    site_area: float
    site_perimeter: float
    max_turbines_bound: int
    reason: Optional[str] = None

    @property
    def site_area_km2(self) -> float:
        return self.site_area / 1e6

    def format_message(self, rated_power_kw: Optional[float] = None) -> str:
        """生成面向规划人员的中文报告。"""
        lines = [
            "布局不可行:",
            f"  请求机位: {self.n_turbines} 台"
            + (
                f" (装机容量约 {self.n_turbines * rated_power_kw / 1e3:.1f} MW)"
                if rated_power_kw is not None
                else ""
            ),
            f"  可用面积: {self.site_area_km2:.3f} km²"
            f" (多边形周长 {self.site_perimeter:.0f} m)",
            f"  最小间距: {self.min_spacing:.1f} m"
            f" (最大转子直径 {self.rotor_diameter_max:.1f} m)",
            f"  几何容量上界: {self.max_turbines_bound} 台",
        ]
        if self.reason:
            lines.append(f"  首要原因: {self.reason}")
        return "\n".join(lines)


@dataclass
class LayoutValidation:
    """布局终验结果（数量 / 边界 / 间距三类检查）。"""

    valid: bool
    n_expected: int
    n_positions: int
    min_spacing: float
    site_area: float
    boundary_violations: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=int)
    )
    spacing_violations: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=int)
    )
    min_pair_distance: float = np.inf
    reason: Optional[str] = None

    @property
    def site_area_km2(self) -> float:
        return self.site_area / 1e6

    def format_message(
        self,
        feasibility: Optional[FeasibilityReport] = None,
        rated_power_kw: Optional[float] = None,
    ) -> str:
        """生成面向规划人员的中文报告（含容量、可用面积、首要违规原因）。"""
        cap = (
            f" (装机容量约 {self.n_expected * rated_power_kw / 1e3:.1f} MW)"
            if rated_power_kw is not None
            else ""
        )
        lines = [
            "布局生成失败（未通过最终校验）:",
            f"  请求机位: {self.n_expected} 台{cap}",
            f"  实际机位: {self.n_positions} 台",
            f"  可用面积: {self.site_area_km2:.3f} km²",
            f"  最小间距要求: {self.min_spacing:.1f} m",
        ]
        if feasibility is not None:
            lines.append(f"  几何容量上界: {feasibility.max_turbines_bound} 台")
        lines.append(f"  首要原因: {self.reason}")
        if self.boundary_violations.size:
            idx_preview = ", ".join(str(int(i)) for i in self.boundary_violations[:5])
            more = " ..." if len(self.boundary_violations) > 5 else ""
            lines.append(f"  越界机位索引: {idx_preview}{more}")
        if self.spacing_violations.shape[0]:
            lines.append(
                f"  最小实际机对间距: {self.min_pair_distance:.1f} m"
                f"（{self.spacing_violations.shape[0]} 对违规）"
            )
        return "\n".join(lines)


class LayoutInfeasibleError(RuntimeError):
    """布局无解或在统一预算内无法生成合法布局。

    携带预判报告（容量/面积）与终验报告（首要违规原因），
    是所有布局生成路径共享的失败语义。
    """

    def __init__(
        self,
        message: str,
        feasibility: Optional[FeasibilityReport] = None,
        validation: Optional[LayoutValidation] = None,
        rated_power_kw: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.feasibility = feasibility
        self.validation = validation
        self.rated_power_kw = rated_power_kw

    @classmethod
    def from_reports(
        cls,
        feasibility: Optional[FeasibilityReport] = None,
        validation: Optional[LayoutValidation] = None,
        rated_power_kw: Optional[float] = None,
    ) -> "LayoutInfeasibleError":
        if validation is not None:
            message = validation.format_message(feasibility, rated_power_kw)
        elif feasibility is not None:
            message = feasibility.format_message(rated_power_kw)
        else:
            message = "布局不可行"
        return cls(message, feasibility, validation, rated_power_kw)


def _polygon_diameter(boundary) -> float:
    """多边形直径（任意两顶点的最大距离；凹多边形直径同样在顶点处取得）。"""
    v = boundary.vertices
    diff = v[:, None, :] - v[None, :, :]
    return float(np.sqrt((diff ** 2).sum(axis=2)).max())


def assess_capacity(
    boundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    min_spacing: Optional[float] = None,
) -> FeasibilityReport:
    """搜索开始前的容量可行性预判（只使用严格必要条件，宁可漏判不可误判）。

    判定依据：

    1. **圆盘面积界**：每台风机占据半径 ``s/2`` 的互斥圆盘（机心距 ≥ s），
       所有圆盘都含于多边形的 Minkowski 膨胀域 ``P ⊕ B(s/2)`` 内。
       对任意简单多边形，该膨胀域面积不超过
       ``A + (s/2)·L + π(s/2)²``（A 为多边形面积，L 为周长），
       故 ``n·π(s/2)² ≤ A + (s/2)·L + π(s/2)²`` 是必要条件。
    2. **多边形直径界**：任意两机心距离不超过多边形直径，n ≥ 2 时
       直径必须 ≥ s。

    Parameters
    ----------
    boundary : SiteBoundary
        真实场地多边形
    n_turbines : int
        请求的风机台数
    rotor_diameters : np.ndarray
        各机组转子直径
    min_multiple : float
        最小间距倍数
    min_spacing : Optional[float]
        直接指定最小间距 (m)，优先于 ``min_multiple``

    Returns
    -------
    FeasibilityReport
        预判报告；``feasible=False`` 表示该组合可严格证明无解
    """
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    if min_spacing is None:
        min_spacing = float(min_multiple * np.max(diameters))
    s = float(min_spacing)

    area = float(boundary.area)
    perimeter = float(boundary.perimeter)
    radius = s / 2.0

    # 圆盘面积必要条件（含边界膨胀项，对凹多边形仍为有效上界）
    offset_area_bound = area + radius * perimeter + np.pi * radius ** 2
    disk_area = np.pi * radius ** 2
    max_by_area = int(np.floor(offset_area_bound / disk_area - 1e-9))
    max_by_area = max(0, max_by_area)

    diameter = _polygon_diameter(boundary)
    # 跨度不足 s 时至多放 1 台；跨度足够时该判据不再构成有限上界
    max_by_diameter = 1 if diameter + 1e-9 < s else _NO_BOUND

    max_bound = min(max_by_area, max_by_diameter)

    reason = None
    feasible = True
    if n_turbines <= 0:
        feasible = False
        reason = "风机台数必须为正整数"
    elif n_turbines > max_bound:
        feasible = False
        if max_by_diameter < n_turbines and max_by_diameter <= max_by_area:
            reason = (
                f"场地最大跨度 {diameter:.0f} m 小于最小间距 {s:.0f} m，"
                "连 2 台满足间距的风机都无法容纳"
            )
        else:
            reason = (
                f"按互斥圆盘（直径 {s:.0f} m）与多边形面积/周长估算，"
                f"场地至多容纳 {max_bound} 台，请求 {n_turbines} 台超出容量上界"
            )

    return FeasibilityReport(
        feasible=feasible,
        n_turbines=int(n_turbines),
        min_spacing=s,
        rotor_diameter_max=float(np.max(diameters)),
        site_area=area,
        site_perimeter=perimeter,
        max_turbines_bound=max_bound,
        reason=reason,
    )


def validate_layout(
    positions: np.ndarray,
    boundary,
    n_expected: int,
    min_spacing: float,
) -> LayoutValidation:
    """所有成功返回必须通过的统一终验：数量、边界、间距。

    Parameters
    ----------
    positions : np.ndarray
        待验机位，形状 (N, 2)
    boundary : SiteBoundary
        场地多边形
    n_expected : int
        期望机位数
    min_spacing : float
        最小间距 (m)

    Returns
    -------
    LayoutValidation
        终验报告；``valid=True`` 时布局方可作为有效结果交付
    """
    positions = np.asarray(positions, dtype=np.float64)
    n_pos = int(positions.shape[0]) if positions.ndim == 2 else 0

    boundary_violations = np.zeros(0, dtype=int)
    spacing_violations = np.zeros((0, 2), dtype=int)
    min_pair = np.inf

    if n_pos >= 2:
        diff = positions[:, None, :] - positions[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = float(dist.min())
        iu = np.triu_indices(n_pos, k=1)
        mask = dist[iu] < min_spacing - 1e-9
        bad_i = iu[0][mask]
        bad_j = iu[1][mask]
        spacing_violations = np.column_stack([bad_i, bad_j]).astype(int)

    if n_pos > 0:
        inside = boundary.contains_all(positions)
        boundary_violations = np.where(~inside)[0].astype(int)

    # 数量不足 / 越界 / 过密，按首要性给出一个原因
    reason = None
    valid = True
    if n_pos != n_expected:
        valid = False
        reason = f"机位数量不足：期望 {n_expected} 台，实际仅 {n_pos} 台（部分布局不得作为有效结果）"
    elif boundary_violations.size:
        valid = False
        reason = (
            f"边界违规：{boundary_violations.size} 台风机位于真实场地多边形之外"
        )
    elif spacing_violations.shape[0]:
        valid = False
        reason = (
            f"间距违规：{spacing_violations.shape[0]} 对风机间距小于 "
            f"{min_spacing:.1f} m，最小仅 {min_pair:.1f} m"
        )

    return LayoutValidation(
        valid=valid,
        n_expected=int(n_expected),
        n_positions=n_pos,
        min_spacing=float(min_spacing),
        site_area=float(boundary.area),
        boundary_violations=boundary_violations,
        spacing_violations=spacing_violations,
        min_pair_distance=min_pair,
        reason=reason,
    )
