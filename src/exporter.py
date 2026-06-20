# src/exporter.py
"""Модуль экспорта данных в CSV."""

import csv
import numpy as np
import os
from typing import Optional


def export_results_to_csv(positions: np.ndarray,
                          distances: np.ndarray,
                          msd: np.ndarray,
                          filepath: str,
                          step_size: float = 1.0) -> None:
    """
    Экспорт результатов моделирования в CSV-файл.

    Args:
        positions: массив финальных позиций (N, 2)
        distances: массив расстояний
        msd: массив MSD по шагам
        filepath: путь к выходному файлу
        step_size: длина шага (для теоретического MSD)
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')

        # Заголовок
        writer.writerow(['particle_id', 'x_final', 'y_final', 'distance'])
        for i, (x, y) in enumerate(positions):
            writer.writerow([i, f'{x:.6f}', f'{y:.6f}', f'{distances[i]:.6f}'])

        # Пустая строка и блок статистики
        writer.writerow([])
        writer.writerow(['Статистика'])
        writer.writerow(['step', 'MSD_experimental', 'MSD_theoretical'])

        from .statistics import theoretical_msd  # локальный импорт
        theory = theoretical_msd(len(msd), step_size)
        for t, (exp, th) in enumerate(zip(msd, theory), 1):
            writer.writerow([t, f'{exp:.6f}', f'{th:.6f}'])