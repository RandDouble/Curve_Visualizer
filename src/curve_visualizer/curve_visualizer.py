import sys
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import QAbstractTableModel, QFile,QDir, QModelIndex, Qt, Slot
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QFileDialog, QHeaderView, QLabel,
                               QLineEdit, QMainWindow, QPushButton, QStyle,
                               QTableView, QWidget)
from PySide6.QtGui import QAction

from .data_keeper import Data

# TIME = "times"
# VOLTAGE = "voltage"
# CURRENT = "current"
# RESISTANCE = "Resistance"
# OHM = r"$\Omega$"
# SCALE = "log"
# FACTOR = {0: "", 3: "k", 6: "M", -3: "m", -6: r"$\mu$", -9: "n"}

# # Iv Curve Offsets, do not touch
# X_OFFSET = 1
# Y_OFFSET = 15


class AppDatabase(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Curve Visualizer")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        # Creating Figure to Plot
        self.view = FigureCanvasQTAgg(Figure(figsize=(7, 3), layout="constrained"))
        self.toolbar = NavigationToolbar(self.view, self)
        self.axes = self.view.figure.add_subplot()

        # Inserting Database Name
        self.input_db = QLineEdit()
        self.input_db.setPlaceholderText("Enter Database Name")
        # Search in "File Explorer"
        self.open_folder_action : QAction = self.input_db.addAction(
            QApplication.instance().style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon), QLineEdit.ActionPosition.TrailingPosition
        )
        self.open_folder_action.triggered.connect(self.on_open_folder)

        self.button_con = QPushButton("Connect")

        self.input_db.returnPressed.connect(self.input_submitted)
        self.button_con.clicked.connect(self.input_submitted)

        # Inserting Database Filter
        self.selector_campione = QComboBox()
        self.selector_date = QComboBox()
        self.selector_type = QComboBox()

        self.selector_campione.setEditable(False)
        self.selector_date.setEditable(False)
        self.selector_type.setEditable(False)
        self.selector_campione.setPlaceholderText("Campione")
        self.selector_date.setPlaceholderText("Data")
        self.selector_type.setPlaceholderText("Tipologia")

        self.selector_campione.activated.connect(self.update_select_datetime)
        self.selector_date.activated.connect(self.update_select_tipologia)
        self.selector_type.activated.connect(self.update_table)

        # Creating checkbox row
        self.label_fit_checkbox = QLabel("Fit with Exponential")
        self.fit_checkbox = QCheckBox()
        self.label_PSD_checkbox = QLabel("PDS")
        self.PSD_checkbox = QCheckBox()

        self.fit_checkbox.setChecked(False)
        self.fit_checkbox.setChecked(False)
        self.fit_checkbox.setEnabled(False)
        self.PSD_checkbox.setEnabled(False)

        # Creating Table
        self.table = QTableView()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().hide()
        self.table.activated.connect(self.plotting_selection)

        # Creating Left Input Database Section
        dblayout = QtWidgets.QHBoxLayout()
        dblayout.addWidget(self.input_db)
        dblayout.addWidget(self.button_con)

        # Creating Left Selector Section
        sel_layout = QtWidgets.QHBoxLayout()
        sel_layout.addWidget(self.selector_campione)
        sel_layout.addWidget(self.selector_date)
        sel_layout.addWidget(self.selector_type)

        # Creating Left Checkbox Section
        check_layout = QtWidgets.QHBoxLayout()
        check_layout.addWidget(self.label_fit_checkbox)
        check_layout.addWidget(self.fit_checkbox)
        check_layout.addWidget(self.label_PSD_checkbox)
        check_layout.addWidget(self.PSD_checkbox)

        # Combining Left Section
        llayout = QtWidgets.QVBoxLayout()
        llayout.addLayout(dblayout)
        llayout.addLayout(sel_layout)
        llayout.addLayout(check_layout)
        llayout.addWidget(self.table)

        # Creating Right Visualization Section
        rlayout = QtWidgets.QVBoxLayout()
        rlayout.addWidget(self.toolbar)
        rlayout.addWidget(self.view, stretch=1)

        # Combining All widget
        main_widget = QWidget(self)
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        main_layout.addLayout(llayout)
        main_layout.addLayout(rlayout)

        # Setting Central Widget
        self.setCentralWidget(main_widget)

    @Slot()
    def input_submitted(self) -> None:
        if Path(self.input_db.text()).exists():
            self.data = Data(self.input_db.text())
            self.data.get_camp_df()
            self.update_select_campione()
            color = "#82E0AA"
        else:
            print("File Not Existant")
            color = "#EC7063"

        self.input_db.setStyleSheet(f"QLineEdit {{background-color : {color}}}")

    @Slot()
    def plotting_selection(self) -> None:
        index = self.table.selectionModel().selectedRows()[0]
        print(f"{index.row()=}")
        rowid = self.plottable_elementes_df.iloc[index.row()].loc["rowid"]
        print(f"{self.plottable_elementes_df.iloc[index.row()].loc["rowid"]=}")
        self.data.plot.clear_all(self.view, self.axes)

        match self.data.tipologia:
            case "IV_CURVE":
                df_measure = self.data.get_measures(rowid)
                self.data.plot.IV_Curve(
                    self.view,
                    self.axes,
                    df_measure,
                )

            case "LETTURA":
                df_measure = self.data.get_measures(rowid)
                self.data.plot.measure(
                    self.view,
                    self.axes,
                    df_measure,
                )

                if self.fit_checkbox.isChecked():
                    self.data.plot.measure_fit(self.view, self.axes, df_measure)

                if self.PSD_checkbox.isChecked():
                    self.data.plot.PSD(self.view, self.axes, df_measure)

            case "IMPULSO_UNIPOLARE" | "IMPULSO_ALTERNATO":
                df_measure = self.data.get_measures(rowid)
                df_impulses, n_rep = self.data.get_impulses(rowid)
                self.data.plot.impulse(
                    self.view, self.axes, df_measure, df_impulses, int(n_rep)
                )

        self.view.draw()

    @Slot()
    def update_select_campione(self):
        self.selector_campione.setEditable(True)
        self.selector_campione.insertItems(
            0, [name for name in self.data.get_campioni()]
        )

    @Slot()
    def update_select_datetime(self):
        # Prima inserire parametro per filtro
        self.data.campione = self.selector_campione.currentText()
        self.possible_date = self.data.get_datetimes()
        # Modifico selettore successivo
        self.selector_date.setEditable(True)
        self.selector_date.clear()
        self.selector_date.insertItems(
            0,
            [
                pd.to_datetime(name).strftime("%Y-%b-%d")
                for name in self.data.get_datetimes()
            ],
        )

    @Slot()
    def update_select_tipologia(self):
        # print(self.selector_date.currentIndex())
        self.selector_type.setEditable(True)
        self.selector_type.clear()
        self.data.datetime = self.possible_date[self.selector_date.currentIndex()]
        self.selector_type.insertItems(0, [name for name in self.data.get_tipologie()])

    @Slot()
    def update_table(self) -> None:
        self.data.tipologia = self.selector_type.currentText()

        # enable or disable checkbox
        if self.data.tipologia == "LETTURA":
            self.fit_checkbox.setEnabled(True)
            self.PSD_checkbox.setEnabled(True)
        else:
            self.fit_checkbox.setEnabled(False)
            self.PSD_checkbox.setEnabled(False)

        self.plottable_elementes_df = self.data.list_plottable_elements_df()
        self.table.horizontalHeader().show()
        model = PandasModel(self.plottable_elementes_df)
        self.table.setModel(model)
        # self.table.resizeColumnsToContents()
        self.table.show()

    @Slot()
    def on_open_folder(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", QDir.homePath(),
            # QFileDialog.ShowDirsOnly
        )

        if Path(file_path).is_file():
            self.input_db.setText(file_path)



class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe"""

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None


###############################################################
# OLD Script                                                  #
# All the following lines were transferred in anothers files. #
# For historical reasons a copy is also kept here.            #
###############################################################


# class Pulse:
#     def __init__(
#         self,
#         n_rep: int,
#         df_pulses: pd.DataFrame | None = None,
#         str_header: str | None = None,
#     ) -> None:
#         self.n_rep = n_rep
#         self.df_pulses = df_pulses
#         self.str_header = str_header

#     def from_dataframe_to_array(self):
#         self.v_params = self.df_pulses.iloc[:, 1:].values
#         self.v_period = self.df_pulses["periodoTot"].values
#         self.v_pulse = self.df_pulses["activeTime"].values
#         self.v_amp = self.df_pulses["voltage"].values
#         self.v_pulse_rep = self.df_pulses["rip"].values

#     def from_header_to_array(self, n_pulses: int) -> np.ndarray:
#         params = np.array(
#             [p.split() for p in self.str_header[:n_pulses]], dtype=np.float64
#         )
#         self.v_params = params
#         self.v_period = self.v_params[:, 0]
#         self.v_pulse = self.v_params[:, 2]
#         self.v_amp = self.v_params[:, 4]
#         self.v_pulse_rep = self.v_params[:, 7].astype(np.int64)

#     def get_times(self):
#         v_dead_time = self.v_period - self.v_pulse

#         # ^
#         # |
#         # |          #2--------#3
#         # |           |         |
#         # |           |         |
#         # | #0-------#1        #0--
#         # ------------------------->
#         #    0       Dead       Period = Dead + Pulse

#         # Ogni impulso richiede 4 punti per essere disegnato
#         v_times = np.zeros(self.v_pulse_rep.sum() * 4 * self.n_rep)
#         # Matrice di singole celle temporali
#         m_single_cell = np.zeros((self.v_period.size, 4))

#         # per ragioni di calcolo questa è la cella degli incrementi temporali
#         # [0 Dead 0 Pulse]
#         m_single_cell[:, 1] = v_dead_time

#         m_single_cell[:, 3] = self.v_pulse

#         # Ripeto il per le ripetizioni interne
#         v_times_not_repeated = np.repeat(m_single_cell, self.v_pulse_rep, axis=0)

#         # Ripeto con le ripetizioni generali
#         v_times = np.tile(v_times_not_repeated, (self.n_rep, 1)).ravel()

#         # Aggiungiamo a ogni riga la somma dei tempi precedenti
#         v_times = np.tri(v_times.size) @ v_times
#         return v_times

#     def get_voltage(self) -> np.ndarray:
#         # ^
#         # |
#         # |          #2--------#3
#         # |           |         |
#         # |           |         |
#         # | #0-------#1        #0--
#         # ------------------------->
#         #    0       Dead       Period = Dead + Pulse

#         # Ogni impulso richiede 4 punti per essere disegnato
#         v_repeted_amps = np.zeros_like(self.v_pulse_rep.sum() * 4 * self.n_rep)

#         m_single_cell = np.zeros((self.v_period.size, 4))
#         m_single_amp = np.zeros_like(m_single_cell)

#         m_single_amp[:, 2:] = np.tile(self.v_amp, (2, 1)).T

#         v_amps_not_repeated = np.repeat(m_single_amp, self.v_pulse_rep, axis=0)

#         # Ripeto con le ripetizioni generali
#         v_repeted_amps = np.tile(v_amps_not_repeated, (self.n_rep, 1)).ravel()

#         return v_repeted_amps

#     def get_time_offset(self) -> float:
#         time_offset = np.inner(self.v_period, self.v_pulse_rep) - self.v_period[-1]
#         return time_offset

#     def from_array_to_func(self, v_params: np.ndarray, n_rep: int) -> Tuple[np.ndarray]:
#         v_period = v_params[:, 0]
#         v_pulse = v_params[:, 2]
#         v_amp = v_params[:, 4]
#         v_pulse_rep = v_params[:, 7].astype(np.int64)

#         v_dead_time = v_period - v_pulse

#         # ^
#         # |
#         # |          #2--------#3
#         # |           |         |
#         # |           |         |
#         # | #0-------#1        #0--
#         # ------------------------->
#         #    0       Dead       Period = Dead + Pulse

#         # Ogni impulso richiede 4 punti per essere disegnato
#         v_times = np.zeros(v_pulse_rep.sum() * 4 * n_rep)
#         v_repeted_amps = np.zeros_like(v_times)

#         m_single_cell = np.zeros((v_period.size, 4))
#         m_single_amp = np.zeros_like(m_single_cell)

#         m_single_amp[:, 2:] = np.tile(v_amp, (2, 1)).T

#         # per ragioni di calcolo questa è la cella degli incrementi temporali
#         # [0 Dead 0 Pulse]
#         m_single_cell[:, 1] = v_dead_time

#         m_single_cell[:, 3] = v_pulse

#         # Ripeto il per le ripetizioni interne
#         v_times_not_repeated = np.repeat(m_single_cell, v_pulse_rep, axis=0)
#         v_amps_not_repeated = np.repeat(m_single_amp, v_pulse_rep, axis=0)

#         # Ripeto con le ripetizioni generali
#         v_times = np.tile(v_times_not_repeated, (n_rep, 1)).ravel()
#         v_repeted_amps = np.tile(v_amps_not_repeated, (n_rep, 1)).ravel()

#         # Aggiungiamo a ogni riga la somma dei tempi precedenti
#         v_times = np.tri(v_times.size) @ v_times

#         return v_times, v_repeted_amps


# class Plotter:
#     def __init__(self):
#         self.cbar = None

#     def clear_all(self, fig: FigureCanvasQTAgg, axes: Axes):
#         if self.cbar:
#             self.cbar.remove()
#             self.cbar = None
#         for ax in fig.figure.axes:
#             ax.clear()
#             if ax != axes:
#                 ax.remove()

#     def initialization(self, data: pd.DataFrame) -> int:
#         """
#         This function initializes the DataFrame, this is done via sideeffect,
#         the return value is the magnitude order of the calculated resistance.
#         """

#         data[RESISTANCE] = data[VOLTAGE] / data[CURRENT]
#         data[SCALE] = np.log10(
#             np.abs(data[RESISTANCE]) + 1e-18,
#         )
#         power = np.floor(data[SCALE].mean())
#         scale = power - power % 3
#         return scale

#     def IV_Curve(
#         self,
#         fig: FigureCanvasQTAgg,
#         axs: Axes,
#         df: pd.DataFrame,
#     ) -> None:
#         df[SCALE] = np.log10(np.abs(df[CURRENT]))
#         power = np.floor(df[SCALE].mean())
#         scale = power - power % 3

#         # each point of the Curve
#         points = np.array([df[VOLTAGE], df[CURRENT] / 10**scale]).T.reshape(-1, 1, 2)
#         # segments connecting each point
#         segments = np.concatenate([points[:-1], points[1:]], axis=1)
#         # Normalization for a color scale dependent on time
#         norm = Normalize(df[TIME].min(), df[TIME].max())

#         # Creation of the smooth line
#         lc = LineCollection(segments=segments, cmap="rainbow", norm=norm)

#         # setting normalization, creation of color line
#         lc.set_array(df[TIME])
#         lc.set_linewidth(2)
#         line = axs.add_collection(lc)
#         # Colorbar
#         self.cbar = fig.figure.colorbar(line, ax=axs)
#         self.cbar.ax.set_ylabel("Time [s]")

#         axs.set_xlabel("Voltage [V]")
#         axs.set_ylabel(f"Current [{FACTOR[scale]}A]")
#         axs.set_title("IV Curve")

#         # The problem with using LineCollection is that you need to manually select limits
#         # for x and y coordinates

#         axs.set_xlim(df[VOLTAGE].min() - X_OFFSET, df[VOLTAGE].max() + X_OFFSET)
#         axs.set_ylim(
#             df[CURRENT].min() / 10**scale - Y_OFFSET,
#             df[CURRENT].max() / 10**scale + Y_OFFSET,
#         )
#         fig.draw()

#     def impulse(
#         self,
#         fig: FigureCanvasQTAgg,
#         ax1: Axes,
#         df_read: pd.DataFrame,
#         df_impulso: pd.DataFrame,
#         n_rep: int,
#     ) -> None:
#         scale: int = self.initialization(df_read)

#         ####################
#         # COMPUTING PULSES #
#         ####################

#         pulses = Pulse(n_rep=n_rep, df_pulses=df_impulso)
#         pulses.from_dataframe_to_array()
#         time = pulses.get_times()
#         voltage = pulses.get_voltage()

#         # Do computation, it is needed for some mod later
#         time_offset: float = pulses.get_time_offset()

#         ############
#         # PLOTTING #
#         ############

#         color = "C0"

#         ax1.plot(
#             time_offset + df_read[TIME],
#             df_read[RESISTANCE] / 10**scale,
#             color=color,
#             marker=".",
#         )
#         ax1.set_xlabel("Time [s]")
#         ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
#         ax1.set_title(RESISTANCE + " vs " + TIME)
#         ax1.tick_params(axis="y", labelcolor=color)

#         color = "C1"
#         ax2: Axes = ax1.twinx()
#         ax2.set_ylabel("Voltage [V]")

#         ax2.tick_params(axis="y", labelcolor=color)

#         ax2.plot(
#             time,
#             voltage,
#             color=color,
#             alpha=0.3,
#         )

#         fig.draw()

#     def measure(self, fig: FigureCanvasQTAgg, ax1: Axes, df: pd.DataFrame) -> None:
#         # Calculating scale for measure unit
#         scale: int = self.initialization(df)

#         ############
#         # PLOTTING #
#         ############

#         color = "C0"

#         # plot of Resistance
#         ax1.plot(
#             df[TIME],
#             df[RESISTANCE] / 10**scale,
#             color=color,
#         )

#         ax1.set_xlabel("Time [s]")
#         ax1.set_ylabel(f"Resistance [{FACTOR[scale]}{OHM}]")
#         ax1.set_title(RESISTANCE + " vs " + TIME)
#         ax1.tick_params(axis="y", labelcolor=color)

#         # Plot of Voltage
#         color = "C1"
#         ax2: Axes = ax1.twinx()
#         ax2.set_ylabel("Voltage [V]")

#         ax2.tick_params(axis="y", labelcolor=color)

#         ax2.plot(
#             df[TIME],
#             df[VOLTAGE],
#             color=color,
#         )

#         fig.draw()


# class Data:
#     """A class where to store datas"""

#     def __init__(self, db_name) -> None:
#         self.con = Connection(db_name)
#         self.cur = self.con.cursor()
#         self.campione = ""
#         self.datetime = ""
#         self.tipologia = ""
#         self.plot = Plotter()

#     def __del__(self) -> None:
#         self.cur.close()
#         self.con.close()

#     @property
#     def campione(self):
#         return self._campione

#     @campione.setter
#     def campione(self, value):
#         self._campione = value

#     @property
#     def datetime(self):
#         return self._datetime

#     @datetime.setter
#     def datetime(self, value) -> None:
#         self._datetime = value

#     @property
#     def tipologia(self):
#         return self._tipologia

#     @tipologia.setter
#     def tipologia(self, value):
#         self._tipologia = value

#     def get_camp_df(self) -> None:
#         self.camp_df = pd.read_sql(
#             "SELECT * FROM campioni",
#             con=self.con,
#             parse_dates={"date": {"format": "%Y-%m-%d"}},
#             dtype={
#                 "campione": "string",
#                 "tipologia": "category",
#             },
#         )

#     def get_campioni(self) -> np.ndarray:
#         return self.camp_df["campione"].unique()

#     def get_datetimes(self) -> np.ndarray:
#         return self.camp_df.loc[
#             self.camp_df["campione"] == self.campione, "date"
#         ].unique()

#     def get_tipologie(self) -> np.ndarray:
#         return self.camp_df.loc[
#             (self.camp_df["campione"] == self.campione)
#             & (self.camp_df["date"] == self.datetime),
#             "tipologia",
#         ].unique()

#     def list_plottable_elements(self) -> np.ndarray:
#         plottable = self.camp_df.loc[
#             (self.camp_df["campione"] == self.campione)
#             & (self.camp_df["date"] == self.datetime)
#             & (self.camp_df["tipologia"] == self.tipologia),
#             ["rowid", "date", "misura", "tipologia"],
#         ].values

#         return plottable

#     def list_plottable_elements_df(self) -> pd.DataFrame:
#         plottable = self.camp_df.loc[
#             (self.camp_df["campione"] == self.campione)
#             & (self.camp_df["date"] == self.datetime)
#             & (self.camp_df["tipologia"] == self.tipologia),
#             ["rowid", "date", "misura", "tipologia"],
#         ]

#         return plottable

#     def get_measures(self, rowid: int) -> pd.DataFrame:
#         df = pd.read_sql(
#             f"SELECT * FROM misure WHERE rowid is {rowid}",
#             con=self.con,
#         )
#         return df

#     def get_impulses(self, rowid: int) -> list[pd.DataFrame, np.ndarray]:
#         df = pd.read_sql(f"SELECT * FROM impulsi WHERE rowid IS {rowid}", con=self.con)

#         return df, self.camp_df.loc[self.camp_df["rowid"] == rowid, "gen_rep"].values


def main():
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    qapp.setStyle("Fusion")
    app = AppDatabase()
    # qapp.setStyleSheet(load_stylesheet(qt_api ="pyside6" ,palette = LightPalette))
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()


if __name__ == "__main__":
    # Check whether there is already a running QApplication (e.g., if running
    # from an IDE).
    qapp = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

    qapp.setStyle("Fusion")
    app = AppDatabase()
    # qapp.setStyleSheet(load_stylesheet(qt_api ="pyside6" ,palette = LightPalette))
    app.show()
    app.activateWindow()
    app.raise_()
    qapp.exec()
