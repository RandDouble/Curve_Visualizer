import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from .pulse import Pulse

TIME = "times"
VOLTAGE = "voltage"
CURRENT = "current"
RESISTANCE = "Resistance"
OHM = r"$\Omega$"
SCALE = "log"
FACTOR = {0: "", 3: "k", 6: "M", -3: "m", -6: r"$\mu$", -9: "n"}

# Iv Curve Offsets, do not touch
X_OFFSET = 1
Y_OFFSET = 15


class Plotter:
    def __init__(self):
        self.cbar = None

    def clear_all(self, fig: FigureCanvasQTAgg, axes: Axes):
        if self.cbar:
            self.cbar.remove()
            self.cbar = None
        for ax in fig.figure.axes:
            ax.clear()
            if ax != axes:
                ax.remove()

    def initialization(self, data: pd.DataFrame) -> int:
        """
        This function initializes the DataFrame, this is done via sideeffect,
        the return value is the magnitude order of the calculated resistance.
        """

        data[RESISTANCE] = data[VOLTAGE] / data[CURRENT]
        data[SCALE] = np.log10(
            np.abs(data[RESISTANCE]) + 1e-18,
        )
        power = np.floor(data[SCALE].mean())
        scale = power - power % 3
        return scale

    def IV_Curve(
        self,
        fig: FigureCanvasQTAgg,
        axs: Axes,
        df: pd.DataFrame,
    ) -> None:
        df[SCALE] = np.log10(np.abs(df[CURRENT]))
        power = np.floor(df[SCALE].mean())
        scale = power - power % 3

        # each point of the Curve
        points = np.array([df[VOLTAGE], df[CURRENT] / 10**scale]).T.reshape(-1, 1, 2)
        # segments connecting each point
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Normalization for a color scale dependent on time
        norm = Normalize(df[TIME].min(), df[TIME].max())

        # Creation of the smooth line
        lc = LineCollection(segments=segments, cmap="rainbow", norm=norm)

        # setting normalization, creation of color line
        lc.set_array(df[TIME])
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
        # Colorbar
        self.cbar = fig.figure.colorbar(line, ax=axs)
        self.cbar.ax.set_ylabel("Time [s]")

        axs.set_xlabel("Voltage [V]")
        axs.set_ylabel(f"Current [{FACTOR[scale]}A]")
        axs.set_title("IV Curve")

        # The problem with using LineCollection is that you need to manually select limits
        # for x and y coordinates

        axs.set_xlim(df[VOLTAGE].min() - X_OFFSET, df[VOLTAGE].max() + X_OFFSET)
        axs.set_ylim(
            df[CURRENT].min() / 10**scale - Y_OFFSET,
            df[CURRENT].max() / 10**scale + Y_OFFSET,
        )
        fig.draw()

    def impulse(
        self,
        fig: FigureCanvasQTAgg,
        ax1: Axes,
        df_read: pd.DataFrame,
        df_impulso: pd.DataFrame,
        n_rep: int,
    ) -> None:
        scale: int = self.initialization(df_read)

        ####################
        # COMPUTING PULSES #
        ####################

        pulses = Pulse(n_rep=n_rep, df_pulses=df_impulso)
        pulses.from_dataframe_to_array()
        time = pulses.get_times()
        voltage = pulses.get_voltage()

        # Do computation, it is needed for some mod later
        time_offset: float = pulses.get_time_offset()

        ############
        # PLOTTING #
        ############

        color = "C0"

        ax1.plot(
            time_offset + df_read[TIME],
            df_read[RESISTANCE] / 10**scale,
            color=color,
            marker=".",
        )
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
        ax1.set_title(RESISTANCE + " vs " + TIME)
        ax1.tick_params(axis="y", labelcolor=color)

        color = "C1"
        ax2: Axes = ax1.twinx()
        ax2.set_ylabel("Voltage [V]")

        ax2.tick_params(axis="y", labelcolor=color)

        ax2.plot(
            time,
            voltage,
            color=color,
            alpha=0.3,
        )

        fig.draw()

    def measure(self, fig: FigureCanvasQTAgg, ax1: Axes, df: pd.DataFrame) -> None:
        # Calculating scale for measure unit
        scale: int = self.initialization(df)

        ############
        # PLOTTING #
        ############

        color = "C0"

        # plot of Resistance
        ax1.plot(
            df[TIME],
            df[RESISTANCE] / 10**scale,
            color=color,
        )

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
        ax1.set_title(RESISTANCE + " vs " + TIME)
        ax1.tick_params(axis="y", labelcolor=color)

        # Plot of Voltage
        color = "C1"
        ax2: Axes = ax1.twinx()
        ax2.set_ylabel("Voltage [V]")

        ax2.tick_params(axis="y", labelcolor=color)

        ax2.plot(
            df[TIME],
            df[VOLTAGE],
            color=color,
        )

        fig.draw()
