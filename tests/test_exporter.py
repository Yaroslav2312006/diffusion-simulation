# tests/test_exporter.py
import os
import tempfile
import numpy as np
from src.exporter import export_results_to_csv
def test_export_csv():
    positions = np.array([[1.0, 2.0], [3.0, 4.0]])
    distances = np.array([2.236, 5.0])
    msd = np.array([1.0, 2.0, 3.0])
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test.csv')
        export_results_to_csv(positions, distances, msd, filepath, step_size=1.0)
        assert os.path.exists(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Проверяем наличие заголовков и данных
        assert 'particle_id' in lines[0]
        assert '1.000000;2.000000' in lines[1]
        # Проверяем наличие блока статистики (гибко, без жёстких индексов)
        assert any('Статистика' in line for line in lines)
        assert any('step;MSD_experimental;MSD_theoretical' in line for line in lines)
