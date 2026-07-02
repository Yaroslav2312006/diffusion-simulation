# src/visualization.py
"""Модуль визуализации результатов моделирования."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from .statistics import theoretical_msd, rayleigh_pdf, histogram_distances
def plot_trajectories(trajectories: List[np.ndarray], num_particles: int = 10, title: str = "Траектории частиц", save_path: Optional[str] = None) -> None:
    """
    Визуализация траекторий выбранного количества частиц.
    Args:
        trajectories: список массивов (шаг, N, 2) или (шаг, 2) для одной частицы
        num_particles: количество частиц для отображения
        title: заголовок графика
        save_path: путь для сохранения (если None, то показываем)
    """
    if not trajectories:
        return
    # Если передан список массивов, каждый массив - позиции на шаге
    # Преобразуем к удобному виду: выберем первые num_particles частиц
    # trajectories[step] имеет форму (N, 2)
    steps = len(trajectories)
    N = trajectories[0].shape[0]
    show_n = min(num_particles, N)
    plt.figure(figsize=(8, 8))
    fig = plt.gcf()  # получить текущую фигуру
    fig.canvas.manager.set_window_title('Соснин Я.Д., гр. БИС-24-3')
    for p in range(show_n):
        # Собираем траекторию частицы p по всем шагам
        x_coords = [traj[p, 0] for traj in trajectories]
        y_coords = [traj[p, 1] for traj in trajectories]
        plt.plot(x_coords, y_coords, linewidth=0.8, alpha=0.7)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    # Добавляем текст с фамилией и группой в рамке
    ax = plt.gca()
    ax.text(0.2, 0.99, 'Соснин Я.Д., гр. БИС-24-3', transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='black', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_histogram(distances: np.ndarray, bins: int = 20, sigma: Optional[float] = None, title: str = "Распределение расстояний", save_path: Optional[str] = None) -> None:
    """
    Построение гистограммы финальных расстояний и наложение теоретической кривой Рэлея.
    Args:
        distances: массив расстояний от начала
        bins: число интервалов гистограммы
        sigma: параметр распределения Рэлея (если None, оценивается по данным)
        title: заголовок
        save_path: путь для сохранения
    """
    plt.figure(figsize=(8, 6))
    fig = plt.gcf()  # получить текущую фигуру
    fig.canvas.manager.set_window_title('Соснин Я.Д., гр. БИС-24-3')
    hist, bin_edges = np.histogram(distances, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.6, color='skyblue', edgecolor='navy', label='Эксперимент')
    # Теоретическая кривая Рэлея
    if sigma is None:
        # Оценка sigma^2 = среднее(r^2)/2 = MSD/2
        sigma = np.sqrt(np.mean(distances ** 2) / 2)
    r = np.linspace(0, np.max(distances), 100)
    pdf = rayleigh_pdf(r, sigma)
    plt.plot(r, pdf, 'r-', linewidth=2, label=f'Рэлей (σ={sigma:.2f})')
    plt.title(title)
    plt.xlabel('Расстояние от начала')
    plt.ylabel('Плотность вероятности')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.text(0.2, 0.99, 'Соснин Я.Д., гр. БИС-24-3', transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='black', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_msd(experimental_msd: np.ndarray, step_size: float = 1.0, title: str = "Среднее квадратичное смещение", save_path: Optional[str] = None) -> None:
    """
    Сравнение экспериментального MSD с теоретическим.
    Args:
        experimental_msd: массив MSD по шагам
        step_size: длина шага
        title: заголовок
        save_path: путь для сохранения
    """
    steps = len(experimental_msd)
    time = np.arange(1, steps + 1)
    theoretical = theoretical_msd(steps, step_size)
    plt.figure(figsize=(8, 6))
    fig = plt.gcf()  # получить текущую фигуру
    fig.canvas.manager.set_window_title('Соснин Я.Д., гр. БИС-24-3')
    plt.plot(time, experimental_msd, 'bo-', label='Эксперимент', markersize=4)
    plt.plot(time, theoretical, 'r--', linewidth=2, label='Теория (4Dt)')
    plt.xlabel('Номер шага')
    plt.ylabel('MSD')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    ax = plt.gca()
    ax.text(0.2, 0.99, 'Соснин Я.Д., гр. БИС-24-3', transform=ax.transAxes, fontsize=12, verticalalignment='bottom', horizontalalignment='right', color='black', bbox=dict(boxstyle='square,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5))
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
