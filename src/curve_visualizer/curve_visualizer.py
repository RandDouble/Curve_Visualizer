import sys
from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import QAbstractTableModel, QDir, QFile, QModelIndex, Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QStyle,
    QTableView,
    QWidget,
    QTabWidget,
    QDoubleSpinBox,
)

from .data_keeper import Data, ViewType


class AppDatabase(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Curve Visualizer")
        self.setMinimumSize(1080, 720)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        # Creating Figure to Plot
        self.view = FigureCanvasQTAgg(Figure(figsize=(3, 2), layout="constrained"))
        self.toolbar = NavigationToolbar(self.view, self)
        self.axes = self.view.figure.add_subplot()

        # Inserting Database Name
        self.input_db = QLineEdit()
        self.input_db.setPlaceholderText("Enter Database Name")

        # Creating tabs for different Visualization

        self.tab_widget = QTabWidget()

        # Search in "File Explorer"
        self.open_folder_action: QAction = self.input_db.addAction(
            QApplication.instance()
            .style()
            .standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon),
            QLineEdit.ActionPosition.TrailingPosition,
        )
        self.open_folder_action.triggered.connect(self.on_open_folder)

        self.button_con = QPushButton("Connect")

        self.input_db.returnPressed.connect(self.input_submitted)
        self.button_con.clicked.connect(self.input_submitted)

        # Creating Left Input Database Section

        dblayout = QtWidgets.QHBoxLayout()
        dblayout.addWidget(self.input_db)
        dblayout.addWidget(self.button_con)

        # Creating Right Visualization Section
        graph_widget = QWidget(self)
        rlayout = QtWidgets.QVBoxLayout(graph_widget)
        rlayout.addWidget(self.toolbar)
        rlayout.addWidget(self.view, stretch=1)
        # graph_widget.setSizePolicy(
        #     QtWidgets.QSizePolicy.Policy.Expanding,
        #     QtWidgets.QSizePolicy.Policy.Expanding,
        # )

        # Adding Widget Necessary for Single Measure Visualization
        SV_widget = self.single_visualization()
        LV_widget = self.long_visualization()

        self.tab_widget.addTab(SV_widget, "Single Visualization")
        self.tab_widget.addTab(LV_widget, "Long Visualization")

        # Combining All widget
        self.main_widget = QWidget()
        main_layout = QtWidgets.QGridLayout(self.main_widget)
        main_layout.addWidget(graph_widget, 0, 1, -1, -1)
        main_layout.addLayout(dblayout, 0, 0)
        main_layout.addWidget(self.tab_widget, 1, 0, 1, 1)
        main_layout.setColumnStretch(0, 4)
        main_layout.setColumnStretch(1, 5)
        main_layout.setColumnMinimumWidth(1, 400)

        # Setting Central Widget
        self.setCentralWidget(self.main_widget)

    @Slot()
    def single_visualization(self) -> QWidget:
        self.create_database_filter_SV()
        self.create_check_box()
        self.create_filters()
        self.table_SV: QTableView = self.create_table()
        self.table_SV.activated.connect(self.plotting_selection)
        return self.create_SV_layout()

    @Slot()
    def long_visualization(self) -> QWidget:
        self.create_database_filter_LV()
        self.plot_long = QPushButton("Long Plot")
        self.plot_long.clicked.connect(self.long_plot)
        self.table_LV: QTableView = self.create_table()
        return self.create_LV_layout()

    @Slot()
    def create_check_box(self) -> None:
        # Creating checkbox row
        self.label_fit_checkbox = QLabel("Fit with Exponential")
        self.fit_checkbox = QCheckBox()
        self.label_PSD_checkbox = QLabel("PDS")
        self.PSD_checkbox = QCheckBox()

        self.fit_checkbox.setChecked(False)
        self.fit_checkbox.setChecked(False)

        self.fit_checkbox.setEnabled(False)
        self.PSD_checkbox.setEnabled(False)

    @Slot()
    def create_filters(self) -> None:
        self.label_inferior_limit = QLabel("Enable Min :")
        self.label_superior_limit = QLabel("Enable Max :")

        self.inferior_limit = QDoubleSpinBox()
        self.superior_limit = QDoubleSpinBox()
        self.inferior_limit.setRange(-999.0, 999.0)
        self.superior_limit.setRange(-999.0, 999.0)
        self.inferior_limit.setDisabled(True)
        self.superior_limit.setDisabled(True)
        self.inferior_limit.setPrefix("min:\t")
        self.superior_limit.setPrefix("max:\t")

        self.check_sup_limit = QCheckBox()
        self.check_inf_limit = QCheckBox()
        self.check_sup_limit.setChecked(False)
        self.check_inf_limit.setChecked(False)
        self.check_sup_limit.stateChanged.connect(self.enable_max_select)
        self.check_inf_limit.stateChanged.connect(self.enable_min_select)

        self.button_replot = QPushButton("Plot")
        self.button_replot.pressed.connect(self.plotting_selection)
        self.button_replot.setProperty("plotting", False)
        self.button_replot.setStyleSheet(
            """
                                        QPushButton::pressed{
                                        background-color: #EC7063;
                                        }
                                        QPushButton::disabled{
                                        background-color: #EC7063;
                                        }"""
        )

    @Slot()
    def create_table(self) -> QTableView:
        # Creating Table
        table = QTableView()

        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)

        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().hide()

        return table

    @Slot()
    def create_database_filter_SV(self) -> None:
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

    @Slot()
    def create_database_filter_LV(self) -> None:
        self.selector_campione_LV = QComboBox()
        self.selector_date_LV = QComboBox()

        self.selector_campione_LV.setEditable(False)
        self.selector_date_LV.setEditable(False)

        self.selector_campione_LV.setPlaceholderText("Campione")
        self.selector_date_LV.setPlaceholderText("Data")

        self.selector_campione_LV.activated.connect(self.update_select_datetime_LV)
        self.selector_date_LV.activated.connect(self.update_table_LV)

    @Slot()
    def create_SV_layout(self) -> QWidget:
        # Creating Left Selector Section
        sel_layout = QtWidgets.QHBoxLayout()
        sel_layout.addWidget(self.selector_campione, stretch=2)
        sel_layout.addWidget(self.selector_date, stretch=1)
        sel_layout.addWidget(self.selector_type, stretch=1)

        # Creating Left Checkbox Section
        options = QWidget()
        check_layout = QtWidgets.QGridLayout(options)
        check_layout.addWidget(self.label_fit_checkbox, 0, 1)
        check_layout.addWidget(self.fit_checkbox, 0, 0)
        check_layout.addWidget(self.label_PSD_checkbox, 1, 1)
        check_layout.addWidget(self.PSD_checkbox, 1, 0)

        check_layout.addWidget(self.label_inferior_limit, 0, 3)
        check_layout.addWidget(self.check_inf_limit, 0, 4)
        check_layout.addWidget(self.inferior_limit, 0, 5)
        check_layout.addWidget(self.label_superior_limit, 1, 3)
        check_layout.addWidget(self.check_sup_limit, 1, 4)
        check_layout.addWidget(self.superior_limit, 1, 5)
        check_layout.setColumnMinimumWidth(5, 200)

        check_layout.addWidget(self.button_replot, 0, 6, 2, 1)

        self.button_replot.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        check_layout.setColumnStretch(2, 1)

        # check_layout.setColumnStretch(4, 1)

        SV_widget = QWidget(self)
        # Combining Left Section
        llayout = QtWidgets.QVBoxLayout(SV_widget)
        # llayout.addLayout(dblayout)
        llayout.addLayout(sel_layout)
        llayout.addWidget(options)
        # llayout.addLayout(check_layout)
        llayout.addWidget(self.table_SV)

        return SV_widget

    @Slot()
    def create_LV_layout(self) -> QWidget:
        # Creating Left Selector Section
        sel_layout = QtWidgets.QHBoxLayout()
        sel_layout.addWidget(self.selector_campione_LV)
        sel_layout.addWidget(self.selector_date_LV)
        sel_layout.addWidget(self.plot_long)
        LV_widget = QWidget()
        # Combining Left Section
        llayout = QtWidgets.QVBoxLayout(LV_widget)
        # llayout.addLayout(dblayout)
        llayout.addLayout(sel_layout)
        llayout.addWidget(self.table_LV)

        return LV_widget

    @Slot()
    def enable_min_select(self) -> None:
        self.inferior_limit.setEnabled(not self.inferior_limit.isEnabled())

    @Slot()
    def enable_max_select(self) -> None:
        self.superior_limit.setEnabled(not self.superior_limit.isEnabled())

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
        self.button_replot.setDisabled(True)
        index = self.table_SV.selectionModel().selectedRows()[0]
        print(f"{index.row()=}")
        rowid = self.plottable_elementes_df.iloc[index.row()].loc["rowid"]
        print(f"{rowid=}")
        self.data.plot.clear_all(self.view, self.axes)

        df_measure, scale = self.data.get_measures(rowid)
        df_filter_min = df_measure["resistance"] < np.inf
        df_filter_max = df_measure["resistance"] > -np.inf

        if self.inferior_limit.isEnabled():
            df_filter_min = (
                df_measure["resistance"] > self.inferior_limit.value() * 10**scale
            )
        else:
            self.inferior_limit.setValue(self.data.min)

        if self.superior_limit.isEnabled():
            df_filter_max = (
                df_measure["resistance"] < self.superior_limit.value() * 10**scale
            )
        else:
            self.superior_limit.setValue(self.data.max)

        df_filter = df_filter_max & df_filter_min

        match self.data.tipologia:
            case "IV_CURVE":
                self.data.plot.IV_Curve(
                    self.view,
                    self.axes,
                    df_measure,
                )

            case "LETTURA":
                self.data.plot.measure(
                    self.view,
                    self.axes,
                    df_measure.loc[df_filter, :],
                )

                if self.fit_checkbox.isChecked():
                    self.data.plot.measure_fit(self.view, self.axes, df_measure)

                if self.PSD_checkbox.isChecked():
                    self.data.plot.PSD(self.view, self.axes, df_measure)

            case "IMPULSO_UNIPOLARE" | "IMPULSO_ALTERNATO":
                df_impulses, n_rep = self.data.get_impulses(rowid)
                self.data.plot.impulse(
                    self.view,
                    self.axes,
                    df_measure[df_filter],
                    df_impulses,
                    n_rep.item(),
                )

        self.toolbar.update()
        self.view.draw()
        self.button_replot.setEnabled(True)
        # self.button_replot.setProperty(
        #     "plotting", not self.button_replot.property("plotting")
        # )
        # self.button_replot.style().unpolish(self.button_replot)
        # self.button_replot.style().polish(self.button_replot)

    @Slot()
    def long_plot(self) -> None:
        rowids = self.data.list_plottable_elements_df_LV()["rowid"].values
        df = self.data.get_measures_LV(rowids)
        self.data.plot.clear_all(self.view, self.axes)
        self.data.plot.long_plot(self.view, self.axes, df)
        self.view.draw()

    @Slot()
    def update_select_campione(self):
        self.selector_campione.setEditable(True)
        self.selector_campione.clear()
        self.selector_campione.insertItems(
            0, [name for name in self.data.get_campioni()]
        )
        self.selector_campione_LV.setEditable(True)
        self.selector_campione_LV.clear()
        self.selector_campione_LV.insertItems(
            0, [name for name in self.data.get_campioni()]
        )

    @Slot()
    def update_select_datetime(self):
        # Prima inserire parametro per filtro
        self.data.campione = self.selector_campione.currentText()
        self.possible_date = self.data.get_datetimes(ViewType.SV)
        # Modifico selettore successivo
        self.selector_date.setEditable(True)
        self.selector_date.clear()
        self.selector_date.insertItems(
            0,
            [
                pd.to_datetime(name).strftime("%Y-%b-%d")
                for name in self.data.get_datetimes(ViewType.SV)
            ],
        )

    @Slot()
    def update_select_datetime_LV(self) -> None:
        # Prima inserire parametro per filtro
        self.data.campione_LV = self.selector_campione_LV.currentText()
        self.possible_date_LV = self.data.get_datetimes(ViewType.LV)
        # Modifico selettore successivo
        self.selector_date_LV.setEditable(True)
        self.selector_date_LV.clear()
        self.selector_date_LV.insertItems(
            0,
            [
                pd.to_datetime(name).strftime("%Y-%b-%d")
                for name in self.data.get_datetimes(ViewType.LV)
            ],
        )

    @Slot()
    def update_select_tipologia(self):
        # print(self.selector_date.currentIndex())
        self.selector_type.setEditable(True)
        self.selector_type.clear()
        self.data.datetime = self.possible_date[self.selector_date.currentIndex()]
        self.selector_type.insertItems(
            0, [name for name in self.data.get_tipologie(ViewType.SV)]
        )

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
        self.table_SV.horizontalHeader().show()
        model = PandasModel(self.plottable_elementes_df)
        self.table_SV.setModel(model)
        # self.table_SV.resizeColumnsToContents()
        self.table_SV.show()

    @Slot()
    def update_table_LV(self) -> None:
        self.data.datetime_LV = self.possible_date_LV[
            self.selector_date_LV.currentIndex()
        ]
        self.plottable_elementes_df = self.data.list_plottable_elements_df_LV()
        self.table_LV.horizontalHeader().show()
        model = PandasModel(self.plottable_elementes_df)
        self.table_LV.setModel(model)
        # self.table_SV.resizeColumnsToContents()
        self.table_LV.show()

    @Slot()
    def on_open_folder(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            QDir.homePath(),
            # QFileDialog.ShowDirsOnly
        )

        if Path(file_path).is_file():
            self.input_db.setText(file_path)


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe"""

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=None) -> int:
        """Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent is None:
            parent = QModelIndex()

        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=None) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """

        if parent is None:
            parent = QModelIndex()
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._dataframe.index[section])

        return None


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
    main()
