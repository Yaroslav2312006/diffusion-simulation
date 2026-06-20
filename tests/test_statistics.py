# tests/test_statistics.py
import numpy as np
from src.statistics import theoretical_msd, rayleigh_pdf

def test_theoretical_msd():
    msd = theoretical_msd(5, 0.5)
    expected = np.array([0.25, 0.5, 0.75, 1.0, 1.25])
    assert np.allclose(msd, expected)

def test_rayleigh_pdf():
    r = np.linspace(0, 3, 10)
    pdf = rayleigh_pdf(r, 1.0)
    # Проверка, что интеграл ~1 (численно)
    integral = np.trapz(pdf, r)
    assert np.isclose(integral, 1.0, rtol=0.05)