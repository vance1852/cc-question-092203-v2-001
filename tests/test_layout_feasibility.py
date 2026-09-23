"""布局可行性处理的回归测试。

覆盖三类场地，证明行为稳定：

* 可行场地 —— 规则/交错网格与 GA/PSO 都必须返回通过边界+间距校验的
  完整布局；
* 临界场地 —— 恰好放得下的组合必须成功；理论上放得下但随机搜索命中率
  极低的组合必须在统一尝试预算内终止并报告，不能整夜不返回，也不能
  交出部分布局；
* 确定无解场地 —— 开始搜索前即判定（容量上界/退化场地/非法请求），
  报告容量、可用面积与首要违规原因。

运行方式: python -m unittest discover -s tests -v
"""

import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wind_farm_opt.constraints.boundary import (
    SiteBoundary,
    create_hexagonal_boundary,
    create_irregular_boundary,
    create_rectangular_boundary,
)
from wind_farm_opt.constraints.feasibility import (
    AttemptBudget,
    FeasibilityError,
    REASON_CAPACITY_EXCEEDED,
    REASON_INVALID_REQUEST,
    REASON_REPAIR_BOUNDARY,
    REASON_REPAIR_SPACING,
    REASON_SAMPLING_BUDGET,
    REASON_SITE_DEGENERATE,
    assess_feasibility,
    build_feasible_layout,
    capacity_upper_bound,
    validate_layout,
)
from wind_farm_opt.constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
    enforce_min_spacing,
)
from wind_farm_opt.optimization.baseline import (
    generate_grid_layout,
    generate_staggered_grid_layout,
)
from wind_farm_opt.optimization.ga import GAConfig, GeneticAlgorithm
from wind_farm_opt.optimization.pso import PSOConfig, ParticleSwarmOptimizer

# V126-3.45MW 转子直径，5 倍直径最小间距
ROTOR_DIAMETER = 126.0
MIN_MULTIPLE = 5.0
MIN_SPACING = ROTOR_DIAMETER * MIN_MULTIPLE  # 630 m


def _diameters(n):
    return np.full(n, ROTOR_DIAMETER, dtype=np.float64)


def _assert_valid_layout(test_case, boundary, positions, n_turbines):
    """统一断言：完整、在场内、间距达标。"""
    test_case.assertEqual(positions.shape, (n_turbines, 2))
    report = validate_layout(boundary, positions, MIN_SPACING, n_turbines)
    test_case.assertTrue(report.feasible, report.format_message())
    inside = boundary.contains_all(positions)
    test_case.assertTrue(bool(inside.all()), "存在越界机位")
    valid, violations = check_min_spacing(positions, MIN_SPACING)
    test_case.assertTrue(valid, f"存在 {len(violations)} 对间距违规")


def _trivial_fitness(positions: np.ndarray) -> float:
    """无需风资源模型的快速适应度函数。"""
    return float(np.sum(positions[:, 0]) - 1e-6 * np.sum(positions[:, 1]))


class TestFeasibleSites(unittest.TestCase):
    """可行场地：所有生成路径都必须返回通过校验的完整布局。"""

    def setUp(self):
        self.boundary = create_rectangular_boundary(3500.0, 3500.0)

    def test_grid_layout_multiple_seeds(self):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            pos = generate_grid_layout(
                self.boundary, 12, _diameters(12), MIN_MULTIPLE, rng=rng
            )
            _assert_valid_layout(self, self.boundary, pos, 12)

    def test_staggered_layout_multiple_seeds(self):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            pos = generate_staggered_grid_layout(
                self.boundary, 12, _diameters(12), MIN_MULTIPLE, rng=rng
            )
            _assert_valid_layout(self, self.boundary, pos, 12)

    def test_grid_layout_irregular_boundary(self):
        boundary = create_irregular_boundary()
        rng = np.random.default_rng(7)
        pos = generate_grid_layout(boundary, 8, _diameters(8), MIN_MULTIPLE, rng=rng)
        _assert_valid_layout(self, boundary, pos, 8)

    def test_grid_layout_hexagonal_boundary(self):
        boundary = create_hexagonal_boundary(radius=2000.0)
        rng = np.random.default_rng(3)
        pos = generate_grid_layout(boundary, 7, _diameters(7), MIN_MULTIPLE, rng=rng)
        _assert_valid_layout(self, boundary, pos, 7)

    def test_single_turbine(self):
        rng = np.random.default_rng(0)
        pos = generate_grid_layout(self.boundary, 1, _diameters(1), MIN_MULTIPLE, rng=rng)
        _assert_valid_layout(self, self.boundary, pos, 1)

    def test_ga_returns_valid_layout(self):
        config = GAConfig(
            population_size=6, max_generations=3, seed=1,
            min_spacing_multiple=MIN_MULTIPLE,
        )
        ga = GeneticAlgorithm(
            n_turbines=6,
            rotor_diameters=_diameters(6),
            boundary=self.boundary,
            fitness_fn=_trivial_fitness,
            config=config,
        )
        result = ga.optimize(verbose=False)
        _assert_valid_layout(self, self.boundary, result.best_positions, 6)
        # 初始种群与最终种群中不得存在无效个体
        for ind in result.final_population:
            _assert_valid_layout(
                self, self.boundary, ind.reshape(6, 2), 6
            )

    def test_pso_returns_valid_layout(self):
        config = PSOConfig(
            swarm_size=6, max_iterations=3, seed=1,
            min_spacing_multiple=MIN_MULTIPLE,
        )
        pso = ParticleSwarmOptimizer(
            n_turbines=6,
            rotor_diameters=_diameters(6),
            boundary=self.boundary,
            fitness_fn=_trivial_fitness,
            config=config,
        )
        result = pso.optimize(verbose=False)
        _assert_valid_layout(self, self.boundary, result.best_positions, 6)
        for ind in result.final_population:
            _assert_valid_layout(
                self, self.boundary, ind.reshape(6, 2), 6
            )


class TestBorderlineSites(unittest.TestCase):
    """临界场地：恰好可行必须成功；搜索命中率极低必须在预算内终止。"""

    def test_exact_fit_2x2_grid(self):
        # 2×2 网格跨度 630.63 m，场地 950×640 恰好容纳
        boundary = create_rectangular_boundary(950.0, 640.0)
        rng = np.random.default_rng(0)
        pos = generate_grid_layout(boundary, 4, _diameters(4), MIN_MULTIPLE, rng=rng)
        _assert_valid_layout(self, boundary, pos, 4)

    def test_exact_fit_2x2_staggered(self):
        # 交错 2×2 横向跨度 630.63+315.3≈945.9 < 950，恰好容纳
        boundary = create_rectangular_boundary(950.0, 640.0)
        rng = np.random.default_rng(0)
        pos = generate_staggered_grid_layout(
            boundary, 4, _diameters(4), MIN_MULTIPLE, rng=rng
        )
        _assert_valid_layout(self, boundary, pos, 4)

    def test_thin_strip_terminates_within_budget(self):
        # 2000×500 狭长场地放 6 台：理论上仅存在近乎完美的锯齿形排布，
        # 随机搜索几乎不可能命中 —— 必须在统一预算内终止并报告，
        # 不能整夜不返回，也不能交出部分布局。
        boundary = create_rectangular_boundary(2000.0, 500.0)
        rng = np.random.default_rng(0)
        start = time.time()
        with self.assertRaises(FeasibilityError) as ctx:
            generate_grid_layout(
                boundary, 6, _diameters(6), MIN_MULTIPLE,
                rng=rng, attempt_budget=300,
            )
        elapsed = time.time() - start
        report = ctx.exception.report
        # 补点预算耗尽或随后的推开修复失败，都是预算内的诚实终止原因
        self.assertIn(
            report.reason,
            (
                REASON_SAMPLING_BUDGET,
                REASON_REPAIR_SPACING,
                REASON_REPAIR_BOUNDARY,
            ),
        )
        self.assertEqual(report.n_turbines, 6)
        self.assertFalse(report.feasible)
        self.assertGreater(report.site_area, 0.0)
        # 绝不允许“悄悄交出越界/过密机位”：即使 6 台都已放置，
        # 也必须显式报告存在违规
        if report.n_placed == 6:
            self.assertTrue(
                report.n_spacing_violations > 0
                or report.n_out_of_bounds > 0
            )
        else:
            self.assertLess(report.n_placed, 6)
        self.assertLess(elapsed, 30.0, "预算内未能及时终止")

    def test_thin_strip_staggered_terminates_within_budget(self):
        boundary = create_rectangular_boundary(2000.0, 500.0)
        rng = np.random.default_rng(1)
        with self.assertRaises(FeasibilityError) as ctx:
            generate_staggered_grid_layout(
                boundary, 6, _diameters(6), MIN_MULTIPLE,
                rng=rng, attempt_budget=300,
            )
        report = ctx.exception.report
        self.assertIn(
            report.reason,
            (
                REASON_SAMPLING_BUDGET,
                REASON_REPAIR_SPACING,
                REASON_REPAIR_BOUNDARY,
            ),
        )
        self.assertFalse(report.feasible)
        if report.n_placed == 6:
            self.assertTrue(
                report.n_spacing_violations > 0
                or report.n_out_of_bounds > 0
            )

    def test_ga_borderline_terminates_with_report(self):
        boundary = create_rectangular_boundary(2000.0, 500.0)
        config = GAConfig(
            population_size=4, max_generations=2, seed=0,
            min_spacing_multiple=MIN_MULTIPLE,
            layout_attempt_budget=200,
        )
        ga = GeneticAlgorithm(
            n_turbines=6,
            rotor_diameters=_diameters(6),
            boundary=boundary,
            fitness_fn=_trivial_fitness,
            config=config,
        )
        start = time.time()
        with self.assertRaises(FeasibilityError):
            ga.optimize(verbose=False)
        self.assertLess(time.time() - start, 30.0)

    def test_pso_borderline_terminates_with_report(self):
        boundary = create_rectangular_boundary(2000.0, 500.0)
        config = PSOConfig(
            swarm_size=4, max_iterations=2, seed=0,
            min_spacing_multiple=MIN_MULTIPLE,
            layout_attempt_budget=200,
        )
        pso = ParticleSwarmOptimizer(
            n_turbines=6,
            rotor_diameters=_diameters(6),
            boundary=boundary,
            fitness_fn=_trivial_fitness,
            config=config,
        )
        with self.assertRaises(FeasibilityError):
            pso.optimize(verbose=False)


class TestProvablyInfeasibleSites(unittest.TestCase):
    """确定无解场地：搜索前即判定，报告容量、可用面积与首要原因。"""

    def setUp(self):
        # 500×500 场地在 630 m 最小间距下容量上界为 4 台
        self.boundary = create_rectangular_boundary(500.0, 500.0)

    def test_capacity_bound_value(self):
        bound = capacity_upper_bound(self.boundary, MIN_SPACING)
        self.assertEqual(bound, 4)

    def test_grid_layout_capacity_exceeded(self):
        rng = np.random.default_rng(0)
        start = time.time()
        with self.assertRaises(FeasibilityError) as ctx:
            generate_grid_layout(self.boundary, 5, _diameters(5), MIN_MULTIPLE, rng=rng)
        # 搜索前判定，必须立即返回
        self.assertLess(time.time() - start, 5.0)
        report = ctx.exception.report
        self.assertEqual(report.reason, REASON_CAPACITY_EXCEEDED)
        self.assertEqual(report.n_turbines, 5)
        self.assertEqual(report.capacity_bound, 4)
        self.assertAlmostEqual(report.site_area, 500.0 * 500.0)
        msg = str(ctx.exception)
        self.assertIn("容量", msg)
        self.assertIn("可用面积", msg)

    def test_staggered_layout_capacity_exceeded(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(FeasibilityError) as ctx:
            generate_staggered_grid_layout(
                self.boundary, 5, _diameters(5), MIN_MULTIPLE, rng=rng
            )
        self.assertEqual(ctx.exception.report.reason, REASON_CAPACITY_EXCEEDED)

    def test_ga_capacity_exceeded_fast(self):
        config = GAConfig(
            population_size=4, max_generations=2, seed=0,
            min_spacing_multiple=MIN_MULTIPLE,
        )
        ga = GeneticAlgorithm(
            n_turbines=5,
            rotor_diameters=_diameters(5),
            boundary=self.boundary,
            fitness_fn=_trivial_fitness,
            config=config,
        )
        start = time.time()
        with self.assertRaises(FeasibilityError) as ctx:
            ga.optimize(verbose=False)
        self.assertLess(time.time() - start, 5.0)
        self.assertEqual(ctx.exception.report.reason, REASON_CAPACITY_EXCEEDED)

    def test_pso_capacity_exceeded_fast(self):
        config = PSOConfig(
            swarm_size=4, max_iterations=2, seed=0,
            min_spacing_multiple=MIN_MULTIPLE,
        )
        pso = ParticleSwarmOptimizer(
            n_turbines=5,
            rotor_diameters=_diameters(5),
            boundary=self.boundary,
            fitness_fn=_trivial_fitness,
            config=config,
        )
        with self.assertRaises(FeasibilityError) as ctx:
            pso.optimize(verbose=False)
        self.assertEqual(ctx.exception.report.reason, REASON_CAPACITY_EXCEEDED)

    def test_degenerate_site(self):
        boundary = SiteBoundary(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
        report = assess_feasibility(boundary, 2, MIN_SPACING)
        self.assertFalse(report.feasible)
        self.assertEqual(report.reason, REASON_SITE_DEGENERATE)
        rng = np.random.default_rng(0)
        with self.assertRaises(FeasibilityError):
            generate_grid_layout(boundary, 2, _diameters(2), MIN_MULTIPLE, rng=rng)

    def test_invalid_request(self):
        report = assess_feasibility(self.boundary, 0, MIN_SPACING)
        self.assertFalse(report.feasible)
        self.assertEqual(report.reason, REASON_INVALID_REQUEST)
        rng = np.random.default_rng(0)
        with self.assertRaises(FeasibilityError):
            generate_grid_layout(self.boundary, 0, _diameters(1), MIN_MULTIPLE, rng=rng)


class TestUnifiedBudgetAndValidation(unittest.TestCase):
    """统一尝试预算与终态校验的单元测试。"""

    def setUp(self):
        self.boundary = create_rectangular_boundary(3500.0, 3500.0)

    def test_attempt_budget_exact_count(self):
        budget = AttemptBudget(2)
        self.assertTrue(budget.consume())
        self.assertTrue(budget.consume())
        self.assertFalse(budget.consume())
        self.assertTrue(budget.exhausted)

    def test_zero_budget_immediate_failure(self):
        budget = AttemptBudget(0)
        self.assertFalse(budget.consume())

    def test_build_layout_zero_budget_no_partial_result(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(FeasibilityError) as ctx:
            build_feasible_layout(
                self.boundary, 4, MIN_SPACING, rng=rng, attempt_budget=0
            )
        report = ctx.exception.report
        self.assertEqual(report.reason, REASON_SAMPLING_BUDGET)
        self.assertEqual(report.n_placed, 0)
        self.assertEqual(report.n_turbines, 4)

    def test_enforce_min_spacing_success(self):
        # 4 个点间距 600 m（不足 630），修复后必须达标
        positions = np.array(
            [[0.0, 0.0], [600.0, 0.0], [0.0, 600.0], [600.0, 600.0]]
        )
        rng = np.random.default_rng(0)
        fixed = enforce_min_spacing(positions, MIN_SPACING, self.boundary, rng)
        _assert_valid_layout(self, self.boundary, fixed, 4)

    def test_enforce_min_spacing_budget_exhausted(self):
        # 8 个点全部重合，仅 1 次迭代预算必然修复失败
        positions = np.zeros((8, 2), dtype=np.float64)
        rng = np.random.default_rng(0)
        with self.assertRaises(FeasibilityError) as ctx:
            enforce_min_spacing(
                positions, MIN_SPACING, self.boundary, rng, max_iterations=1
            )
        # FeasibilityError 是 RuntimeError 子类，兼容旧的捕获方式
        self.assertIsInstance(ctx.exception, RuntimeError)
        self.assertIn(
            ctx.exception.report.reason,
            (REASON_REPAIR_SPACING, REASON_REPAIR_BOUNDARY),
        )

    def test_validate_layout_boundary_violation(self):
        positions = np.array([[0.0, 0.0], [10000.0, 10000.0]])
        report = validate_layout(self.boundary, positions, MIN_SPACING, 2)
        self.assertFalse(report.feasible)
        self.assertEqual(report.reason, REASON_REPAIR_BOUNDARY)
        self.assertEqual(report.n_out_of_bounds, 1)

    def test_validate_layout_spacing_violation(self):
        positions = np.array([[0.0, 0.0], [100.0, 0.0], [1500.0, 1500.0]])
        report = validate_layout(self.boundary, positions, MIN_SPACING, 3)
        self.assertFalse(report.feasible)
        self.assertEqual(report.reason, REASON_REPAIR_SPACING)
        self.assertEqual(report.n_spacing_violations, 1)
        self.assertEqual(report.worst_pair, (0, 1))
        self.assertAlmostEqual(report.worst_distance, 100.0)

    def test_validate_layout_partial_layout_rejected(self):
        positions = np.array([[0.0, 0.0], [1000.0, 0.0]])
        report = validate_layout(self.boundary, positions, MIN_SPACING, n_turbines=4)
        self.assertFalse(report.feasible)
        self.assertEqual(report.n_placed, 2)
        self.assertEqual(report.n_turbines, 4)

    def test_capacity_bound_never_below_feasible_count(self):
        # 容量上界不得小于实际可行台数（对可行场地做健全性检查）
        boundary = create_rectangular_boundary(3500.0, 3500.0)
        rng = np.random.default_rng(5)
        pos = generate_grid_layout(boundary, 12, _diameters(12), MIN_MULTIPLE, rng=rng)
        bound = capacity_upper_bound(boundary, MIN_SPACING)
        self.assertGreaterEqual(bound, pos.shape[0])

    def test_report_message_contains_required_fields(self):
        boundary = create_rectangular_boundary(500.0, 500.0)
        rng = np.random.default_rng(0)
        try:
            generate_grid_layout(boundary, 5, _diameters(5), MIN_MULTIPLE, rng=rng)
            self.fail("应当抛出 FeasibilityError")
        except FeasibilityError as e:
            msg = str(e)
            self.assertIn("请求容量", msg)
            self.assertIn("可用面积", msg)
            self.assertIn("最小间距", msg)


if __name__ == "__main__":
    unittest.main()
