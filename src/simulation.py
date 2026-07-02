# src/simulation.py
"""Основной модуль моделирования случайного блуждания."""
import numpy as np
from typing import Optional, List, Tuple
class DiffusionSimulation:
    """
    Моделирование диффузии методом случайных блужданий.
    Поддерживает векторизованные вычисления для большого числа частиц.
    """
    def __init__(self, n_particles: int, steps: int, step_size: float = 1.0):
        """
        Инициализация параметров моделирования.
        Args:
            n_particles: Количество частиц.
            steps: Число шагов моделирования.
            step_size: Длина одного шага.
        """
        if n_particles <= 0:
            raise ValueError("Number of particles must be positive")
        if steps < 0:
            raise ValueError("Number of steps must be non-negative")
        if step_size <= 0:
            raise ValueError("Step size must be positive")
        self.n_particles = n_particles
        self.steps = steps
        self.step_size = step_size
        self.positions = np.zeros((n_particles, 2), dtype=np.float64)
        self._trajectories: List[np.ndarray] = []  # храним позиции на каждом шаге
    def run(self) -> np.ndarray:
        """
        Запуск моделирования.
        Returns:
            np.ndarray: Финальные позиции частиц (N x 2).
        """
        if self.steps == 0:
            self._trajectories = [self.positions.copy()]
            return self.positions
        self._trajectories = []
        positions = self.positions.copy()
        for _ in range(self.steps):
            # Случайные углы для всех частиц
            angles = np.random.uniform(0, 2 * np.pi, self.n_particles)
            dx = self.step_size * np.cos(angles)
            dy = self.step_size * np.sin(angles)
            positions[:, 0] += dx
            positions[:, 1] += dy
            self._trajectories.append(positions.copy())
        self.positions = positions
        return positions
    def get_msd(self) -> np.ndarray:
        """
        Вычисление среднего квадратичного смещения (MSD) по времени.
        Returns:
            np.ndarray: Массив MSD для каждого шага (длина steps).
        """
        if not self._trajectories:
            self.run()
        msd = np.zeros(self.steps)
        for t, traj in enumerate(self._trajectories, 1):
            # traj: (N, 2) – позиции всех частиц на шаге t
            squared_dist = np.sum(traj ** 2, axis=1)  # квадрат расстояния от начала
            msd[t - 1] = np.mean(squared_dist)
        return msd
    def get_final_distances(self) -> np.ndarray:
        """Расчёт расстояний от начальной точки для всех частиц."""
        return np.sqrt(np.sum(self.positions ** 2, axis=1))
    def get_trajectories(self) -> List[np.ndarray]:
        """Возвращает список массивов траекторий по шагам."""
        if not self._trajectories:
            self.run()
        return self._trajectories
    def get_positions_at_step(self, step: int) -> np.ndarray:
        """Возвращает позиции всех частиц на указанном шаге."""
        if step < 0 or step > self.steps:
            raise IndexError("Step index out of range")
        if not self._trajectories:
            self.run()
        if step == 0:
            return np.zeros((self.n_particles, 2))
        return self._trajectories[step - 1]
