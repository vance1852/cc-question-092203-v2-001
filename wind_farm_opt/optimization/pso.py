"""粒子群优化器。"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.feasibility import (
    AttemptBudget,
    FeasibilityError,
    assert_request_feasible,
    build_feasible_layout,
    default_fill_budget,
    default_repair_budget,
    repair_layout,
    validate_layout,
)
from ..constraints.spacing import (
    check_min_spacing,
    compute_min_spacing_from_diameters,
)


@dataclass
class PSOConfig:
    """粒子群算法配置参数。

    Parameters
    ----------
    swarm_size : int
        粒子群大小
    max_iterations : int
        最大迭代次数
    inertia_weight : float
        惯性权重 w
    cognitive_coeff : float
        认知系数 c1
    social_coeff : float
        社会系数 c2
    max_velocity : float
        最大速度（占场地范围的比例）
    min_spacing_multiple : float
        最小间距倍数（相对于转子直径）
    penalty_factor : float
        约束违反惩罚因子
    layout_attempt_budget : Optional[int]
        生成/修复单个初始布局所共享的尝试预算；None 时使用默认值
    seed : Optional[int]
        随机种子
    """

    swarm_size: int = 40
    max_iterations: int = 150
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.49
    social_coeff: float = 1.49
    max_velocity: float = 0.2
    min_spacing_multiple: float = 5.0
    penalty_factor: float = 1e6
    layout_attempt_budget: Optional[int] = None
    seed: Optional[int] = None


class ParticleSwarmOptimizer:
    """粒子群算法机位优化器。"""

    def __init__(
        self,
        n_turbines: int,
        rotor_diameters: np.ndarray,
        boundary: SiteBoundary,
        fitness_fn: Callable[[np.ndarray], float],
        config: Optional[PSOConfig] = None,
    ) -> None:
        self.n_turbines = n_turbines
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        self.boundary = boundary
        self.fitness_fn = fitness_fn
        self.config = config if config is not None else PSOConfig()

        self.rng = np.random.default_rng(self.config.seed)

        self.min_spacing = compute_min_spacing_from_diameters(
            self.rotor_diameters,
            self.config.min_spacing_multiple,
        )

        self.n_dim = n_turbines * 2
        self.x_range = boundary.x_max - boundary.x_min
        self.y_range = boundary.y_max - boundary.y_min

        self.vel_range = np.zeros(self.n_dim, dtype=np.float64)
        for i in range(self.n_dim):
            self.vel_range[i] = (
                self.x_range if i % 2 == 0 else self.y_range
            ) * self.config.max_velocity

        self.pos_bounds = np.zeros((self.n_dim, 2), dtype=np.float64)
        for i in range(self.n_dim):
            if i % 2 == 0:
                self.pos_bounds[i] = [boundary.x_min, boundary.x_max]
            else:
                self.pos_bounds[i] = [boundary.y_min, boundary.y_max]

        self._best_global_pos = None
        self._best_global_fitness = -np.inf
        self._best_iteration = 0

        self.convergence_history: list[float] = []
        self.mean_history: list[float] = []

    def _initialize_swarm(self, swarm_size: int) -> tuple[np.ndarray, np.ndarray]:
        """初始化粒子群。"""
        positions = np.zeros((swarm_size, self.n_dim), dtype=np.float64)
        velocities = np.zeros((swarm_size, self.n_dim), dtype=np.float64)

        for i in range(swarm_size):
            pos = self._generate_valid_layout()
            positions[i] = pos.flatten()
            velocities[i] = self.rng.uniform(
                -self.vel_range, self.vel_range, self.n_dim
            )

        return positions, velocities

    def _generate_valid_layout(self) -> np.ndarray:
        """生成一个满足约束的初始布局。

        与规则布局共用同一套终止语义：统一尝试预算 + 终态校验，
        失败时抛出 :class:`FeasibilityError`，绝不返回部分布局。
        """
        budget = self.config.layout_attempt_budget
        if budget is None:
            budget = default_fill_budget(self.n_turbines)
        return build_feasible_layout(
            self.boundary,
            self.n_turbines,
            self.min_spacing,
            rng=self.rng,
            attempt_budget=budget,
        )

    def _compute_penalty(self, positions_flat: np.ndarray) -> float:
        """计算约束违反惩罚。"""
        positions = positions_flat.reshape(self.n_turbines, 2)

        penalty = 0.0

        inside = self.boundary.contains_all(positions)
        if not inside.all():
            n_violations = np.sum(~inside)
            penalty += n_violations * self.config.penalty_factor

        valid, violations = check_min_spacing(positions, self.min_spacing)
        if not valid:
            for i, j in violations:
                dist = np.linalg.norm(positions[i] - positions[j])
                penalty += (self.min_spacing - dist) * self.config.penalty_factor

        return penalty

    def _evaluate_particles(self, positions: np.ndarray) -> np.ndarray:
        """评估所有粒子的适应度。"""
        swarm_size = positions.shape[0]
        fitness = np.zeros(swarm_size, dtype=np.float64)

        for i in range(swarm_size):
            penalty = self._compute_penalty(positions[i])

            if penalty > 0:
                fitness[i] = -penalty
            else:
                pos_reshaped = positions[i].reshape(self.n_turbines, 2)
                try:
                    fitness[i] = self.fitness_fn(pos_reshaped)
                except Exception:
                    fitness[i] = -self.config.penalty_factor

        return fitness

    def _repair(self, positions_flat: np.ndarray) -> np.ndarray:
        """修复违反约束的粒子。

        修复与随机补点共享同一套尝试预算语义；修复失败时回退到重新
        生成一个可行布局，保证返回的粒子一定通过边界与间距校验，
        不会把无效布局当作有效个体留在粒子群中。
        """
        positions = positions_flat.reshape(self.n_turbines, 2).copy()

        for i in range(self.n_turbines):
            if not self.boundary.contains_point(positions[i]):
                positions[i] = self.boundary.project_to_boundary(positions[i])

        report = validate_layout(self.boundary, positions, self.min_spacing)
        if report.feasible:
            return positions.flatten()

        budget = AttemptBudget(default_repair_budget(self.n_turbines))
        try:
            positions = repair_layout(
                self.boundary, positions, self.min_spacing, self.rng, budget
            )
            return positions.flatten()
        except FeasibilityError:
            # 修复预算耗尽：回退到重新生成一个可行布局
            return self._generate_valid_layout().flatten()

    def optimize(self, verbose: bool = True) -> "OptimizeResult":
        """执行优化。

        Returns
        -------
        OptimizeResult
            优化结果
        """
        from .ga import OptimizeResult

        swarm_size = self.config.swarm_size
        max_iter = self.config.max_iterations

        w = self.config.inertia_weight
        c1 = self.config.cognitive_coeff
        c2 = self.config.social_coeff

        # 搜索前明显无解判断：与规则布局共用同一套可行性门禁
        assert_request_feasible(self.boundary, self.n_turbines, self.min_spacing)

        if verbose:
            print(f"\n=== 粒子群优化开始 ===")
            print(f"风机台数: {self.n_turbines}")
            print(f"粒子群大小: {swarm_size}")
            print(f"最大迭代: {max_iter}")
            print(f"最小间距: {self.min_spacing:.1f} m "
                  f"({self.config.min_spacing_multiple:.1f}倍转子直径)")
            print(f"w={w}, c1={c1}, c2={c2}")
            print("=" * 35)

        positions, velocities = self._initialize_swarm(swarm_size)
        fitness = self._evaluate_particles(positions)

        best_personal_pos = positions.copy()
        best_personal_fitness = fitness.copy()

        best_global_idx = np.argmax(fitness)
        self._best_global_pos = positions[best_global_idx].reshape(self.n_turbines, 2).copy()
        self._best_global_fitness = float(fitness[best_global_idx])
        self._best_iteration = 0

        for iteration in range(max_iter):
            self.convergence_history.append(float(self._best_global_fitness))
            self.mean_history.append(float(np.mean(fitness)))

            r1 = self.rng.random((swarm_size, self.n_dim))
            r2 = self.rng.random((swarm_size, self.n_dim))

            best_global_flat = self._best_global_pos.flatten()

            velocities = (
                w * velocities
                + c1 * r1 * (best_personal_pos - positions)
                + c2 * r2 * (best_global_flat - positions)
            )

            velocities = np.clip(velocities, -self.vel_range, self.vel_range)

            positions = positions + velocities

            positions = np.clip(
                positions,
                self.pos_bounds[:, 0],
                self.pos_bounds[:, 1],
            )

            for i in range(swarm_size):
                positions[i] = self._repair(positions[i])

            fitness = self._evaluate_particles(positions)

            improved_mask = fitness > best_personal_fitness
            best_personal_pos[improved_mask] = positions[improved_mask].copy()
            best_personal_fitness[improved_mask] = fitness[improved_mask].copy()

            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > self._best_global_fitness:
                self._best_global_fitness = float(fitness[current_best_idx])
                self._best_global_pos = positions[current_best_idx].reshape(
                    self.n_turbines, 2
                ).copy()
                self._best_iteration = iteration + 1

            if verbose and (iteration % 5 == 0 or iteration == max_iter - 1):
                print(
                    f"Iter {iteration+1:3d} | "
                    f"Best: {self._best_global_fitness/1e3:8.2f} GWh | "
                    f"Mean: {np.mean(fitness)/1e3:8.2f} GWh | "
                    f"Found@Iter {self._best_iteration}"
                )

        if verbose:
            print("=" * 35)
            print(f"优化完成!")
            print(f"最优净AEP: {self._best_global_fitness/1e3:.2f} GWh")
            print(f"找到最优解的迭代: {self._best_iteration}")

        # 成功返回前的终态校验：最优布局必须通过边界与间距检查
        final_report = validate_layout(
            self.boundary,
            self._best_global_pos,
            self.min_spacing,
            n_turbines=self.n_turbines,
        )
        if not final_report.feasible:
            raise FeasibilityError(final_report)

        return OptimizeResult(
            best_positions=self._best_global_pos.copy(),
            best_fitness=float(self._best_global_fitness),
            best_generation=self._best_iteration,
            convergence_history=self.convergence_history.copy(),
            mean_history=self.mean_history.copy(),
            final_population=positions.copy(),
            final_fitness=fitness.copy(),
        )
