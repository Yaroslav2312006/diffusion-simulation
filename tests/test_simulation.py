# tests/test_simulation.py
import pytest
import numpy as np
from src.simulation import DiffusionSimulation

def test_initialization():
    sim = DiffusionSimulation(100, 50, 0.5)
    assert sim.n_particles == 100
    assert sim.steps == 50
    assert sim.step_size == 0.5
    assert sim.positions.shape == (100, 2)
    assert np.all(sim.positions == 0)

def test_invalid_params():
    with pytest.raises(ValueError):
        DiffusionSimulation(0, 10)
    with pytest.raises(ValueError):
        DiffusionSimulation(10, -1)
    with pytest.raises(ValueError):
        DiffusionSimulation(10, 10, 0)

def test_run_zero_steps():
    sim = DiffusionSimulation(5, 0)
    pos = sim.run()
    assert np.all(pos == 0)
    assert len(sim.get_trajectories()) == 1  # начальное состояние

def test_step_magnitude():
    sim = DiffusionSimulation(1, 1, 2.0)
    pos = sim.run()
    distance = np.sqrt(pos[0,0]**2 + pos[0,1]**2)
    # Поскольку угол случайный, расстояние должно быть равно step_size с плавающей точностью
    # Но из-за случайности может не быть точно, проверяем близость
    assert np.isclose(distance, 2.0, rtol=1e-3)

def test_msd_shape():
    sim = DiffusionSimulation(10, 20)
    sim.run()
    msd = sim.get_msd()
    assert len(msd) == 20
    assert msd[0] > 0  # первый шаг даёт средний квадрат step_size^2

def test_final_distances():
    sim = DiffusionSimulation(10, 5, 1.0)
    sim.run()
    dist = sim.get_final_distances()
    assert len(dist) == 10
    assert np.all(dist >= 0)