import sys
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import (QAbstractTableModel, QDir, QFile, QModelIndex, Qt,
                            Slot)
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QFileDialog, QHeaderView, QLabel,
                               QLineEdit, QMainWindow, QPushButton, QStyle,
                               QTableView, QWidget)

from .data_keeper import Data


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
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
                    self.view, self.axes, df_measure, df_impulses, n_rep.item()
                )

        self.view.draw()

    @Slot()
    def update_select_campione(self):
        self.selector_campione.setEditable(True)
        self.selector_campione.insertItems(0, [name for name in self.data.get_campioni()])

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
