"""布局可行性处理的行为测试。

用三类场地证明修复后的行为稳定：

* **可行场地**：网格/交错/GA/PSO 都返回数量正确、全部在界内、间距达标的布局；
* **临界场地**：容量上界附近仍能在统一预算内成功，或干净地报告失败，
  绝不无限循环、绝不交出部分布局；
* **确定无解场地**：开始搜索前即被容量预判拦截，抛出携带容量、可用面积
  与首要原因的 LayoutInfeasibleError。
"""

import signal
import time

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import (
    SiteBoundary,
    create_rectangular_boundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
)
from wind_farm_opt.constraints.feasibility import (
    LayoutInfeasibleError,
    assess_capacity,
    validate_layout,
)
from wind_farm_opt.optimization.baseline import (
    generate_grid_layout,
    generate_staggered_grid_layout,
)
from wind_farm_opt.optimization.layout_common import AttemptBudget
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig

D = 126.0  # V126 转子直径
S = 5.0 * D  # 630 m 最小间距


def diameters(n, d=D):
    return np.full(n, d)


def assert_layout_valid(pos, boundary, n, s=S):
    """布局必须数量正确、全部在界内、两两间距达标。"""
    pos = np.asarray(pos, dtype=np.float64)
    assert pos.shape == (n, 2)
    report = validate_layout(pos, boundary, n, s)
    assert report.valid, report.format_message()


# ---------------------------------------------------------------------------
# 1. 容量预判：确定无解
# ---------------------------------------------------------------------------

class TestCapacityPrecheck:
    def test_too_many_turbines_rejected_before_search(self):
        # 1 km x 1 km 场地，最小间距 630 m，最多放 1~2 台；请求 20 台必无解
        boundary = create_rectangular_boundary(1000.0, 1000.0)
        report = assess_capacity(boundary, 20, diameters(20), min_spacing=S)
        assert report.feasible is False
        assert report.max_turbines_bound < 20
        assert report.reason is not None
        # 报告包含容量与可用面积信息
        msg = report.format_message(rated_power_kw=3450.0)
        assert "可用面积" in msg and "容量上界" in msg and "MW" in msg

    def test_tiny_site_cannot_fit_two(self):
        # 场地跨度 400 m < 630 m：连 2 台都放不下
        boundary = create_rectangular_boundary(400.0, 400.0)
        report = assess_capacity(boundary, 2, diameters(2), min_spacing=S)
        assert report.feasible is False
        assert report.max_turbines_bound <= 1

    def test_single_turbine_feasible_when_site_nonempty(self):
        boundary = create_rectangular_boundary(400.0, 400.0)
        report = assess_capacity(boundary, 1, diameters(1), min_spacing=S)
        assert report.feasible is True

    def test_real_polygon_used_not_bbox(self):
        # 用一个很“瘦”的多边形（凹四边形），其外接矩形能放下但本体不行：
        # 三角形 (0,0),(2000,0),(0,300)，面积仅 3e5 m²
        verts = np.array([[0.0, 0.0], [2000.0, 0.0], [0.0, 300.0]])
        boundary = SiteBoundary(verts)
        # 10 台远超面积/膨胀上界
        report = assess_capacity(boundary, 10, diameters(10), min_spacing=S)
        assert report.feasible is False
        assert report.site_area == pytest.approx(300_000.0, rel=1e-9)

    def test_feasible_site_passes_precheck(self):
        boundary = create_rectangular_boundary(3500.0, 3500.0)
        report = assess_capacity(boundary, 12, diameters(12), min_spacing=S)
        assert report.feasible is True
        assert report.max_turbines_bound >= 12


# ---------------------------------------------------------------------------
# 2. 网格 / 交错布局：可行、临界、无解
# ---------------------------------------------------------------------------

class TestRegularLayouts:
    @pytest.mark.parametrize("n", [1, 2, 5, 12, 20])
    @pytest.mark.parametrize("staggered", [False, True])
    def test_feasible_sites_return_valid_layout(self, n, staggered):
        boundary = create_rectangular_boundary(4000.0, 4000.0)
        rng = np.random.default_rng(123)
        gen = (
            generate_staggered_grid_layout if staggered
            else generate_grid_layout
        )
        pos = gen(boundary, n, diameters(n), min_multiple=5.0, rng=rng)
        assert_layout_valid(pos, boundary, n)

    def test_irregular_polygon_feasible(self):
        boundary = create_irregular_boundary()
        rng = np.random.default_rng(7)
        pos = generate_grid_layout(boundary, 12, diameters(12), rng=rng)
        assert_layout_valid(pos, boundary, 12)

    def test_hexagonal_feasible(self):
        boundary = create_hexagonal_boundary(2500.0)
        rng = np.random.default_rng(7)
        pos = generate_staggered_grid_layout(
            boundary, 9, diameters(9), rng=rng
        )
        assert_layout_valid(pos, boundary, 9)

    def test_definitely_infeasible_raises_with_report(self):
        boundary = create_rectangular_boundary(800.0, 800.0)
        rng = np.random.default_rng(1)
        with pytest.raises(LayoutInfeasibleError) as exc:
            generate_grid_layout(boundary, 10, diameters(10), rng=rng)
        err = exc.value
        # 预判阶段即失败
        assert err.feasibility is not None and not err.feasibility.feasible
        assert "容量上界" in str(err)

    def test_critical_density_either_valid_or_clean_failure(self):
        # 临界：场地 2000x2000、间距 630 m，方形网格最多 4x4=16 台。
        # 请求 9 台必须成功；请求 20 台超过实际容量，必须干净地失败
        # （通过预判或终验），而不是返回越界/过密的机位。
        boundary = create_rectangular_boundary(2000.0, 2000.0)
        rng = np.random.default_rng(99)

        pos = generate_grid_layout(boundary, 9, diameters(9), rng=rng)
        assert_layout_valid(pos, boundary, 9)

        rng = np.random.default_rng(99)
        with pytest.raises(LayoutInfeasibleError):
            generate_grid_layout(boundary, 20, diameters(20), rng=rng)

    def test_never_returns_partial_or_illegal_layout(self):
        # 小预算 + 紧张场地：要么合法返回 n 台，要么抛异常，绝不返回部分
        boundary = create_rectangular_boundary(1900.0, 1900.0)
        budget = AttemptBudget(fill_draws=50, repair_iterations=30)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            try:
                pos = generate_grid_layout(
                    boundary, 8, diameters(8), rng=rng, budget=budget
                )
            except LayoutInfeasibleError:
                continue
            assert_layout_valid(pos, boundary, 8)

    def test_no_infinite_loop_under_tight_budget(self):
        # 确定无解 + 极小预算：必须快速失败（旧代码会整夜死循环）
        boundary = create_rectangular_boundary(1200.0, 1200.0)
        budget = AttemptBudget(fill_draws=5, repair_iterations=5)

        def handler(signum, frame):
            raise AssertionError("布局生成超时——存在无界循环")

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        try:
            with pytest.raises(LayoutInfeasibleError):
                generate_staggered_grid_layout(
                    boundary, 6, diameters(6),
                    rng=np.random.default_rng(0), budget=budget,
                )
        finally:
            signal.alarm(0)

    def test_failure_reports_capacity_area_and_primary_reason(self):
        boundary = create_rectangular_boundary(1500.0, 1500.0)
        budget = AttemptBudget(fill_draws=20, repair_iterations=10)
        with pytest.raises(LayoutInfeasibleError) as exc:
            generate_grid_layout(
                boundary, 16, diameters(16),
                rng=np.random.default_rng(3), budget=budget,
            )
        msg = str(exc.value)
        assert "可用面积" in msg
        assert "首要原因" in msg


# ---------------------------------------------------------------------------
# 3. GA / PSO：初始化与终止语义
# ---------------------------------------------------------------------------

def _fitness(pos):
    # 简单、确定性的适应度：点越分散越高（不依赖尾流/AEP 模块）
    if len(pos) < 2:
        return 0.0
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    return float(d[np.triu_indices(len(pos), k=1)].sum())


class TestOptimizerInitialization:
    def test_ga_constructor_rejects_infeasible(self):
        boundary = create_rectangular_boundary(800.0, 800.0)
        with pytest.raises(LayoutInfeasibleError):
            GeneticAlgorithm(
                n_turbines=10,
                rotor_diameters=diameters(10),
                boundary=boundary,
                fitness_fn=_fitness,
                config=GAConfig(population_size=4, max_generations=2, seed=1),
            )

    def test_pso_constructor_rejects_infeasible(self):
        boundary = create_rectangular_boundary(800.0, 800.0)
        with pytest.raises(LayoutInfeasibleError):
            ParticleSwarmOptimizer(
                n_turbines=10,
                rotor_diameters=diameters(10),
                boundary=boundary,
                fitness_fn=_fitness,
                config=PSOConfig(swarm_size=4, max_iterations=2, seed=1),
            )

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_ga_initial_population_all_valid(self, seed):
        boundary = create_rectangular_boundary(3000.0, 3000.0)
        ga = GeneticAlgorithm(
            n_turbines=8,
            rotor_diameters=diameters(8),
            boundary=boundary,
            fitness_fn=_fitness,
            config=GAConfig(population_size=6, max_generations=3, seed=seed),
        )
        pop = ga._initialize_population(6)
        for ind in pop:
            assert_layout_valid(ind.reshape(8, 2), boundary, 8)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_pso_initial_swarm_all_valid(self, seed):
        boundary = create_rectangular_boundary(3000.0, 3000.0)
        pso = ParticleSwarmOptimizer(
            n_turbines=8,
            rotor_diameters=diameters(8),
            boundary=boundary,
            fitness_fn=_fitness,
            config=PSOConfig(swarm_size=6, max_iterations=3, seed=seed),
        )
        pos, _ = pso._initialize_swarm(6)
        for ind in pos:
            assert_layout_valid(ind.reshape(8, 2), boundary, 8)

    def test_ga_optimize_returns_valid_best_and_terminates(self):
        boundary = create_rectangular_boundary(3000.0, 3000.0)
        ga = GeneticAlgorithm(
            n_turbines=8,
            rotor_diameters=diameters(8),
            boundary=boundary,
            fitness_fn=_fitness,
            config=GAConfig(
                population_size=6, max_generations=4, seed=5,
                mutation_rate=0.6, mutation_strength=0.3,
            ),
        )
        start = time.time()
        result = ga.optimize(verbose=False)
        assert time.time() - start < 30
        assert_layout_valid(result.best_positions, boundary, 8)

    def test_pso_optimize_returns_valid_best_and_terminates(self):
        boundary = create_rectangular_boundary(3000.0, 3000.0)
        pso = ParticleSwarmOptimizer(
            n_turbines=8,
            rotor_diameters=diameters(8),
            boundary=boundary,
            fitness_fn=_fitness,
            config=PSOConfig(swarm_size=6, max_iterations=4, seed=5),
        )
        start = time.time()
        result = pso.optimize(verbose=False)
        assert time.time() - start < 30
        assert_layout_valid(result.best_positions, boundary, 8)

    def test_ga_repair_never_keeps_illegal_individual(self):
        boundary = create_rectangular_boundary(3000.0, 3000.0)
        ga = GeneticAlgorithm(
            n_turbines=8,
            rotor_diameters=diameters(8),
            boundary=boundary,
            fitness_fn=_fitness,
            config=GAConfig(population_size=4, max_generations=2, seed=11),
        )
        # 构造一个越界 + 过密的个体
        bad = np.zeros((8, 2))
        bad[:, 0] = [5000.0] * 8  # 全部越界且重叠
        fixed = ga._repair(bad.flatten()).reshape(8, 2)
        assert_layout_valid(fixed, boundary, 8)

    def test_pso_repair_never_keeps_illegal_particle(self):
        boundary = create_rectangular_boundary(3000.0, 3000.0)
        pso = ParticleSwarmOptimizer(
            n_turbines=8,
            rotor_diameters=diameters(8),
            boundary=boundary,
            fitness_fn=_fitness,
            config=PSOConfig(swarm_size=4, max_iterations=2, seed=11),
        )
        bad = np.zeros((8, 2))
        bad[:, 1] = 5000.0
        fixed = pso._repair(bad.flatten()).reshape(8, 2)
        assert_layout_valid(fixed, boundary, 8)
