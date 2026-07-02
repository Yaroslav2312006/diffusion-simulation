# src/main.py
"""Точка входа в приложение. Обработка аргументов командной строки."""
import argparse
import sys
import numpy as np
from .simulation import DiffusionSimulation
from .visualization import plot_trajectories, plot_histogram, plot_msd
from .exporter import export_results_to_csv
import os
os.makedirs('output', exist_ok=True)
def main():
    parser = argparse.ArgumentParser(description='Моделирование диффузии частиц')
    parser.add_argument('-n', '--particles', type=int, default=1000, help='Количество частиц (по умолчанию 1000)')
    parser.add_argument('-s', '--steps', type=int, default=100, help='Число шагов моделирования (по умолчанию 100)')
    parser.add_argument('-l', '--step-size', type=float, default=1.0, help='Длина шага (по умолчанию 1.0)')
    parser.add_argument('-o', '--output-dir', type=str, default='./output', help='Директория для сохранения результатов (по умолчанию ./output)')
    parser.add_argument('--no-viz', action='store_true', help='Отключить визуализацию (только экспорт)')
    parser.add_argument('--seed', type=int, default=None, help='Зерно для генератора случайных чисел (для воспроизводимости)')
    args = parser.parse_args()
    # Установка зерна
    if args.seed is not None:
        np.random.seed(args.seed)
    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    # Моделирование
    sim = DiffusionSimulation(args.particles, args.steps, args.step_size)
    print(f"Запуск моделирования: {args.particles} частиц, {args.steps} шагов, шаг {args.step_size}")
    positions = sim.run()
    print("Моделирование завершено.")
    # Расчёт статистик
    msd = sim.get_msd()
    distances = sim.get_final_distances()
    trajectories = sim.get_trajectories()
    # Сохранение результатов в CSV
    csv_path = os.path.join(args.output_dir, 'results.csv')
    export_results_to_csv(positions, distances, msd, csv_path, args.step_size)
    print(f"Результаты экспортированы в {csv_path}")
    if not args.no_viz:
        # Визуализация траекторий
        traj_path = os.path.join(args.output_dir, 'trajectories.png')
        plot_trajectories(trajectories, num_particles=10, title=f"Траектории частиц (N={args.particles})", save_path='output/trajectories Соснин Я.Д. БИС-24-3.png')
        print(f"Траектории сохранены в {traj_path}")
        # Гистограмма распределения
        hist_path = os.path.join(args.output_dir, 'histogram.png')
        plot_histogram(distances, bins=20, title="Распределение расстояний", save_path='output/histogram Соснин Я.Д. БИС-24-3.png')
        print(f"Гистограмма сохранена в {hist_path}")
        # График MSD
        msd_path = os.path.join(args.output_dir, 'msd.png')
        plot_msd(msd, args.step_size, title="Среднее квадратичное смещение", save_path='output/msd Соснин Я.Д. БИС-24-3.png')
        print(f"График MSD сохранён в {msd_path}")
    # Дополнительная информация в консоль
    print(f"Среднее расстояние: {np.mean(distances):.3f}")
    print(f"Средний квадрат расстояния: {np.mean(distances ** 2):.3f}")
    print(f"Теоретическое MSD на последнем шаге: {args.step_size ** 2 * args.steps:.3f}")
    print("Готово.")
if __name__ == '__main__':
    main()
