# src/particle.py
"""Модуль, описывающий отдельную частицу (для иллюстрации ООП)."""

import numpy as np


class Particle:
    """Класс, представляющий частицу с координатами и траекторией."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
        self.trajectory = [(x, y)]

    def step(self, step_size: float, angle: float) -> None:
        """Сдвиг частицы на шаг с заданным углом."""
        self.x += step_size * np.cos(angle)
        self.y += step_size * np.sin(angle)
        self.trajectory.append((self.x, self.y))

    def get_position(self) -> tuple:
        return (self.x, self.y)

    def get_trajectory(self) -> list:
        return self.trajectory