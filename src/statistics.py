# src/statistics.py
"""Статистические расчёты для анализа диффузии."""

import numpy as np
from typing import Tuple


def theoretical_msd(steps: int, step_size: float = 1.0) -> np.ndarray:
    """
    Теоретическое значение MSD для двумерной диффузии: <r^2> = 4Dt.
    В дискретной модели D = step_size^2 / 4 (при tau=1), поэтому <r^2> = step_size^2 * t.
    """
    return step_size ** 2 * np.arange(1, steps + 1)


def histogram_distances(distances: np.ndarray, bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Построение гистограммы распределения расстояний.

    Returns:
        hist: массив частот
        bin_edges: границы интервалов
    """
    hist, bin_edges = np.histogram(distances, bins=bins, density=True)
    return hist, bin_edges


def rayleigh_pdf(r: np.ndarray, sigma: float) -> np.ndarray:
    """
    Плотность распределения Рэлея: P(r) = (r/sigma^2) * exp(-r^2/(2*sigma^2)).
    """
    return (r / sigma ** 2) * np.exp(-r ** 2 / (2 * sigma ** 2))