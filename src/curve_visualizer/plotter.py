import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.optimize as optimize
from sklearn.metrics import r2_score

from seaborn import set_theme, color_palette
from matplotlib import rc as mpl_rc
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

# from matplotlib.pyplot import set_loglevel

from .pulse import Pulse


set_theme(style="whitegrid")
mpl_rc("svg", fonttype="none")
mpl_rc("font", family="serif", size=20)
mpl_rc("axes", titlesize=32, labelsize=28)
mpl_rc("ytick", labelsize=20)
mpl_rc("xtick", labelsize=20)

TIME = "times"
VOLTAGE = "voltage"
CURRENT = "current"
RESISTANCE = "resistance"
FREQUENCY = "Frequency"
PSD = "PSD"
OHM = r"$\Omega$"
SCALE = "log"
FACTOR = {
    12: "T",
    9: "G",
    6: "M",
    3: "k",
    0: "",
    -3: "m",
    -6: r"$\mu$",
    -9: "n",
    -12: "p",
    -15: "f",
    -18: "a",
}


class Plotter:
    def __init__(self):
        self.cbar = None

    def clear_all(self, fig: FigureCanvasQTAgg, axes: Axes) -> None:
        # Eliminate Colorbar
        if self.cbar:
            self.cbar.remove()
            self.cbar = None
        # Restore Initial Layout
        axes.set_subplotspec(GridSpec(1, 1, fig.figure)[0, 0])
        # Removing Legends
        for leg in fig.figure.legends:
            leg.remove()
        # remove axes in excess
        for ax in fig.figure.axes:
            ax.clear()
            if ax != axes:
                fig.figure.delaxes(ax)

    def initialization(self, data: pd.DataFrame) -> int:
        """
        This function initializes the DataFrame, this is done via sideeffect,
        the return value is the magnitude order of the calculated resistance.
        """
        power = np.floor(data[SCALE].mean())
        scale = power - power % 3
        return scale

    def initialization_long(self, data: pd.DataFrame) -> int:
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

    def scale(self, df: pd.DataFrame) -> int:
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
        lc = LineCollection(
            segments=segments, cmap=color_palette("Spectral", as_cmap=True), norm=norm
        )

        # setting normalization, creation of color line
        lc.set_array(df[TIME])
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
        # Colorbar
        self.cbar = fig.figure.colorbar(line, ax=axs)
        self.cbar.ax.set_ylabel("Time [s]")

        axs.axhline(0, color="k")
        axs.axvline(0, color="k")
        axs.set_xlabel("Voltage [V]", labelpad=5)
        axs.set_ylabel(f"Current [{FACTOR[scale]}A]", labelpad=5)
        axs.tick_params(axis="y", color="k")  # Resetting color to black
        # axs.set_title("IV Curve", fontdict={"size": 22}, pad=10)

        # The problem with using LineCollection is that you need to manually select limits
        # for x and y coordinates

        axs.autoscale(True, "both")
        # axs.set_xlim(df[VOLTAGE].min() - X_OFFSET, df[VOLTAGE].max() + X_OFFSET)
        # axs.set_ylim(
        #     df[CURRENT].min() / 10**scale - Y_OFFSET,
        #     df[CURRENT].max() / 10**scale + Y_OFFSET,
        # )

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
        time_offset: np.float64 = pulses.get_time_offset()

        ############
        # PLOTTING #
        ############

        color = "C0"

        ax1.plot(
            time_offset + df_read[TIME],
            df_read[RESISTANCE] / 10**scale,
            color=color,
            marker=".",
            linewidth=1,
            alpha=0.7,
            linestyle="dashed",
        )
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
        # ax1.set_title(RESISTANCE + " vs " + TIME)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.yaxis.label.set_color(color)
        # ax1.grid(visible=True)

        color = "C1"
        ax2: Axes = ax1.twinx()
        ax2.set_ylabel("Voltage [V]")

        ax2.tick_params(axis="y", labelcolor=color)
        ax2.yaxis.label.set_color(color)

        ax2.plot(
            time,
            voltage,
            color=color,
            alpha=0.3,
            linewidth=1,
        )
        ax2.grid(visible=False)

    def measure(self, fig: FigureCanvasQTAgg, ax1: Axes, df: pd.DataFrame) -> None:
        """
        Methods to plot Reading measures, defined as current read whilst a constant voltage is applied.
        In this case resistance is defined as Ohm's Law.a
        """
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
            label="Data",
            linewidth=1,
            marker=".",
            linestyle="dotted",
            alpha=0.9,
        )

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
        # ax1.set_title(RESISTANCE + " vs " + TIME)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.yaxis.label.set_color(color)

        # Plot of Voltage
        color = "C1"
        ax2: Axes = ax1.twinx()

        ax2.set_ylabel("Voltage [V]")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.yaxis.label.set_color(color)

        ax2.plot(
            df[TIME],
            df[VOLTAGE],
            color=color,
            # alpha=0.8,
        )
        # ax1.grid(visible=True)
        ax2.grid(visible=False)

    def measure_fit(self, fig: FigureCanvasQTAgg, ax: Axes, df: pd.DataFrame) -> None:
        scale = self.initialization(df)

        def exp_to_fit(x, tau, A_0, q):
            return A_0 * np.exp(-x / tau) + q

        # Starting value for [tau, A_0, q]
        p0 = (1.0, 230, 20)
        # Setting bounds for optimal value
        # Remember the format for bounds is [[inferior bound], [superior bound]]
        # And the bounds order must match the optimized value order
        # So first [tau, A_0, q]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        popt, _ = optimize.curve_fit(
            exp_to_fit, df[TIME], df[RESISTANCE] / (10**scale), p0=p0, bounds=bounds
        )

        r2 = r2_score(df[RESISTANCE] / 10**scale, exp_to_fit(df[TIME], *popt))

        color = "C2"
        ax.plot(
            df[TIME],
            exp_to_fit(df[TIME], *popt),
            color=color,
            label=(
                "Fit\n"
                r"$R^2$" + f" :  {r2:.2f}\n"
                r"$\tau$" + f" : {popt[0]:.2f}\n"
                r"$A_0$" + f" : {popt[1]:.2f}\n"
                r"$q$" + f" : {popt[2]:.2f}"
            ),
        )

        fig.figure.legend(ncols=2)

    def PSD(self, fig: FigureCanvasQTAgg, ax: Axes, df: pd.DataFrame) -> None:
        freq, PSD_point = self.PSD_inititializer(df)

        new_ax = fig.figure.add_subplot()

        color = "C0"

        new_ax.plot(freq, PSD_point, color=color, label="PSD")
        new_ax.set_xlabel("Frequency [Hz]")
        new_ax.set_ylabel("PSD")
        new_ax.set_yscale("log")
        # new_ax.set_title("PSD of Resistance")

        gs = GridSpec(2, 1)

        ax.set_subplotspec(gs[0, 0])
        new_ax.set_subplotspec(gs[1, 0])

        fig.figure.add_subplot(ax)
        fig.figure.add_subplot(new_ax)

    def long_plot(self, fig: FigureCanvasQTAgg, ax1: Axes, df: pd.DataFrame):
        scale = self.initialization_long(df)

        color = "C0"

        ax1.plot(
            df[TIME],
            df[RESISTANCE] / 10**scale,
            label="resistance",
            color=color,
            linewidth=1,
            alpha=0.9,
        )

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
        # ax1.set_title(RESISTANCE + " vs " + TIME)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.yaxis.label.set_color(color)

        color = "C1"
        ax2: Axes = ax1.twinx()

        ax2.plot(
            df[TIME],
            df[VOLTAGE],
            label="voltage",
            color=color,
            linewidth=1,
        )

        ax2.set_ylabel("Voltage [V]")
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.yaxis.label.set_color(color)
        # ax1.grid(visible=True)
        ax2.grid(visible=False)

    def gamma_curve(self, fig: FigureCanvasQTAgg, ax1: Axes, df: pd.DataFrame) -> None:
        """Reimplemented from F. Profumo matlab code and from A. Acha paper. Calculates Gamma Curve from IV Graph"""

        gs = GridSpec(3, 2, figure=fig.figure, width_ratios=(90, 8), height_ratios=(40, 30, 30))

        ax1.set_subplotspec(gs[0, 0])

        first_quadrant = fig.figure.add_subplot(gs[1, 0])
        third_quadrant = fig.figure.add_subplot(gs[2, 0], sharex=first_quadrant)
        fig.figure.add_subplot(ax1)

        get_changing_voltage: np.ndarray = (
            np.nonzero(np.diff(df[VOLTAGE].values))[0] + 1
        )  # Find all indeces where voltage changes

        actual_changing_voltage: np.ndarray = np.zeros(get_changing_voltage.size + 1)
        # get changing voltages has one elemet less
        # because it doesn't contain the first voltage assumed.
        actual_changing_voltage[1:] = get_changing_voltage  # Transferring Info

        voltage_mean_value: np.ndarray = df.loc[
            actual_changing_voltage, VOLTAGE
        ].values  # get_changing_voltages has one elements less,
        current_mean: np.ndarray = np.zeros(
            voltage_mean_value.size
        )  # Allocating vector for current means
        equals_elements: np.ndarray = np.diff(actual_changing_voltage)

        for index, element in enumerate(actual_changing_voltage):
            if index != actual_changing_voltage.size - 1:
                end_pos = equals_elements[index] + element
                current_mean[index] = np.mean(df.loc[element:end_pos, CURRENT].values)

        log_v = np.log(np.abs(voltage_mean_value))
        log_i = np.log(np.abs(current_mean))
        gamma = np.gradient(log_i, log_v)
        gradient = np.gradient(voltage_mean_value) # To be used as filter, reduces amount of computing needed
        sqrt_v = np.sqrt(np.abs(voltage_mean_value))

        color = "C0"  # First quadrant, voltage increasing
        first_quadrant.plot(
            sqrt_v[(voltage_mean_value >= 0) & (gradient > 0)],
            gamma[(voltage_mean_value >= 0) & (gradient > 0)],
            color=color,
            label="I quadrant\nV decreasing",
            linestyle="",
            marker=".",
        )

        color = "C1"  # First quadrant, voltage decreasing
        first_quadrant.plot(
            sqrt_v[(voltage_mean_value >= 0) & (gradient < 0)],
            gamma[(voltage_mean_value >= 0) & (gradient < 0)],
            color=color,
            label="I quadrant\nV decreasing",
            linestyle="",
            marker=".",
        )

        color = "C2"  # Third quadrant, voltage increasing
        third_quadrant.plot(
            sqrt_v[(voltage_mean_value < 0) & (gradient > 0)],
            gamma[(voltage_mean_value < 0) & (gradient > 0)],
            color=color,
            label="III quadrant\nV increasing",
            linestyle="",
            marker=".",
        )

        color = "C3"  # Third quadrant, voltage decreasing
        third_quadrant.plot(
            sqrt_v[(voltage_mean_value < 0) & (gradient < 0)],
            gamma[(voltage_mean_value < 0) & (gradient < 0)],
            color=color,
            label="III quadrant\nV decreasing",
            linestyle="",
            marker=".",
        )


        third_quadrant.set_xlabel(r"$\sqrt{V}$ [$V^{1/2}$]")
        third_quadrant.set_ylabel(r"$\gamma$")
        first_quadrant.set_ylabel(r"$\gamma$")
        first_quadrant.tick_params('x', labelbottom = 0)
        fig.figure.legend(
            loc="center right",
            # bbox_to_anchor=(1.01, 0.3, 0.25, 0.45),
            # bbox_transform=first_quadrant.transAxes,
            # borderaxespad=0.0,
        )
