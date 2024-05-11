import pytest
from curve_visualizer.pulse import Pulse
import pandas as pd
import numpy as np

test_pulse = Pulse()

def test_get_times():
    test_array = test_pulse.get_times()
    assert np.isclose(test_array[0], 0.)
    assert np.isclose(test_array[1:-1], test_array[2:])

def test_get_voltages():
    test_array = test_pulse.get_voltage()
    assert np.isclose(test_array[:-1], test_array[1:])