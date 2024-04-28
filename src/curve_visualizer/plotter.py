import numpy as np
import pandas as pd
import scipy.optimize as optimize
from sklearn.metrics import r2_score

from seaborn import set_theme, color_palette
from matplotlib import rc as mpl_rc
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from .local_constants import FACTOR, TIME, RESISTANCE, SCALE, OHM, CURRENT, VOLTAGE
# from matplotlib.pyplot import set_loglevel

from .pulse import Pulse
from .plotter_commodities import (
    initialization,
    initialization_long,
    calc_scale,
    PSD_inititializer,
    calc_gamma,
    apply_common_labels,
    exp_to_fit
)

set_theme(style="whitegrid")
mpl_rc("svg", fonttype="none")
mpl_rc("font", family="serif", size=20)
mpl_rc("axes", titlesize=32, labelsize=28)
mpl_rc("ytick", labelsize=20)
mpl_rc("xtick", labelsize=20)


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
        for legend in fig.figure.legends:
            legend.remove()
        # remove axes in excess
        for ax in fig.figure.axes:
            ax.clear()
            if ax != axes:
                fig.figure.delaxes(ax)

    def IV_Curve(
        self,
        fig: FigureCanvasQTAgg,
        axs: Axes,
        df: pd.DataFrame,
    ) -> None:
        df[SCALE] = np.log10(np.abs(df[CURRENT]))

        scale: int = calc_scale(df)

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
        scale: int = initialization(df_read)

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
        apply_common_labels(axs=ax1, color=color, scale=scale)

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
        scale: int = initialization(df)

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

        apply_common_labels(ax1, color, scale)

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
        scale = initialization(df)
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
        freq, PSD_point = PSD_inititializer(df)

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
        scale = initialization_long(df)

        color = "C0"

        ax1.plot(
            df[TIME],
            df[RESISTANCE] / 10**scale,
            label="resistance",
            color=color,
            linewidth=1,
            alpha=0.9,
        )

        apply_common_labels(ax1, color=color, scale=scale)

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

        gs = GridSpec(
            3, 2, figure=fig.figure, width_ratios=(90, 8), height_ratios=(40, 30, 30)
        )

        ax1.set_subplotspec(gs[0, 0])

        first_quadrant = fig.figure.add_subplot(gs[1, 0])
        third_quadrant = fig.figure.add_subplot(gs[2, 0], sharex=first_quadrant)
        fig.figure.add_subplot(ax1)

        sqrt_v, gamma, gradient, voltage_mean_value = calc_gamma(df)
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
        first_quadrant.tick_params("x", labelbottom=0)
        fig.figure.legend(
            loc="center right",
            # bbox_to_anchor=(1.01, 0.3, 0.25, 0.45),
            # bbox_transform=first_quadrant.transAxes,
            # borderaxespad=0.0,
        )
