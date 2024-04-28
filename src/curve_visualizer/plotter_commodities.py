import numpy as np
import pandas as pd
import scipy.signal as signal
from matplotlib.axes import Axes


from .local_constants import SCALE, RESISTANCE, TIME, VOLTAGE, CURRENT, FACTOR, OHM


def initialization(data: pd.DataFrame) -> int:
    """
    This function initializes the DataFrame, this is done via sideeffect,
    the return value is the magnitude order of the calculated resistance.
    """
    power = np.floor(data[SCALE].mean())
    scale = power - power % 3
    return scale


def initialization_long(data: pd.DataFrame) -> int:
    """
    This function initializes the DataFrame, this is done via sideeffect,
    the return value is the magnitude order of the calculated resistance.
    """

    data[SCALE] = np.log10(
        np.abs(data[RESISTANCE]) + 1e-18,
    )

    power = np.floor(data[SCALE].mean())
    scale = power - power % 3

    previous = None
    for row in np.nditer(data["rowid"].unique()):
        print(f"{row=}")
        print(f'Misura={data.loc[data["rowid"]==row, "misura"].unique()}')
        if previous == None:
            previous = row
            continue
        # print(f'{data.loc[data["rowid"] == previous, TIME]=}')
        data.loc[data["rowid"] == row, TIME] += data.loc[
            data["rowid"] == previous, TIME
        ].max()
        previous = row

    return scale


def calc_scale(df: pd.DataFrame) -> int:
    power = np.floor(df[SCALE].mean())
    scale = power - power % 3
    return scale


def PSD_inititializer(self, data: pd.DataFrame) -> tuple[np.ndarray]:
    sampling_freq = np.reciprocal(
        np.mean(
            data.iloc[1:, data.columns.get_loc(TIME)].values
            - data.iloc[:-1, data.columns.get_loc(TIME)].values
        )
    )
    print(f"{sampling_freq =}")

    freq, PSD_data = signal.periodogram(data[RESISTANCE], sampling_freq)
    return freq, PSD_data


def calc_gamma(data: pd.DataFrame) -> list[np.ndarray]:
    get_changing_voltage: np.ndarray = (
        np.nonzero(np.diff(data[VOLTAGE].values))[0] + 1
    )  # Find all indeces where voltage changes

    actual_changing_voltage: np.ndarray = np.zeros(get_changing_voltage.size + 1)
    # get changing voltages has one elemet less
    # because it doesn't contain the first voltage assumed.
    actual_changing_voltage[1:] = get_changing_voltage  # Transferring Info

    voltage_mean_value: np.ndarray = data.loc[
        actual_changing_voltage, VOLTAGE
    ].values  # get_changing_voltages has one elements less,
    current_mean: np.ndarray = np.zeros(
        voltage_mean_value.size
    )  # Allocating vector for current means
    equals_elements: np.ndarray = np.diff(actual_changing_voltage)

    for index, element in enumerate(actual_changing_voltage):
        if index != actual_changing_voltage.size - 1:
            end_pos = equals_elements[index] + element
            current_mean[index] = np.mean(data.loc[element:end_pos, CURRENT].values)

    log_v = np.log(np.abs(voltage_mean_value))
    log_i = np.log(np.abs(current_mean))
    gamma = np.gradient(log_i, log_v)
    gradient = np.gradient(
        voltage_mean_value
    )  # To be used as filter, reduces amount of computing needed
    sqrt_v = np.sqrt(np.abs(voltage_mean_value))
    return sqrt_v, gamma, gradient, voltage_mean_value


def apply_common_labels(axs: Axes, color: str, scale: int):
    """Function to reduce code duplication. It applies to impulse, measure and long_plot"""
    axs.set_xlabel("Time [s]")
    axs.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
    axs.tick_params(axis="y", labelcolor=color)
    axs.yaxis.label.set_color(color)

def exp_to_fit(x, tau, A_0, q):
            return A_0 * np.exp(-x / tau) + q
