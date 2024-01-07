from sqlite3 import Connection

import numpy as np
import numpy.typing as npt
import pandas as pd

from enum import Enum

from .plotter import Plotter


class ViewType(Enum):
    SV = 0
    LV = 1


class Data:
    """A class where to store datas"""

    def __init__(self, db_name) -> None:
        self.con = Connection(db_name)
        self.cur = self.con.cursor()
        self.campione = ""
        self.campione_LV = ""
        self.datetime = ""
        self.datetime_LV = ""
        self.tipologia = ""
        self.plot = Plotter()

    def __del__(self) -> None:
        self.cur.close()
        self.con.close()

    @property
    def campione(self):
        return self._campione

    @campione.setter
    def campione(self, value):
        self._campione = value

    @property
    def campione_LV(self):
        return self._campione_LV

    @campione_LV.setter
    def campione_LV(self, value):
        self._campione_LV = value

    @property
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, value) -> None:
        self._datetime = value

    @property
    def datetime_LV(self):
        return self._datetime_LV

    @datetime_LV.setter
    def datetime_LV(self, value) -> None:
        self._datetime_LV = value

    @property
    def tipologia(self):
        return self._tipologia

    @tipologia.setter
    def tipologia(self, value):
        self._tipologia = value

    def get_camp_df(self) -> None:
        self.camp_df = pd.read_sql(
            "SELECT * FROM campioni",
            con=self.con,
            parse_dates={"date": {"format": "%Y-%m-%d"}},
            dtype={
                "campione": "string",
                "tipologia": "category",
            },
        )

    def get_campioni(self) -> np.ndarray:
        return self.camp_df["campione"].unique()

    def get_datetimes(self, view_type: ViewType) -> np.ndarray:
        match view_type:
            case ViewType.SV:
                camp = self.campione
            case ViewType.LV:
                camp = self.campione_LV

        return self.camp_df.loc[self.camp_df["campione"] == camp, "date"].unique()

    def get_tipologie(self, view_type : ViewType) -> np.ndarray:
        match view_type:
            case ViewType.SV:
                camp = self.campione
                time = self.datetime
            case ViewType.LV:
                camp = self.campione_LV
                time = self.datetime_LV

        return self.camp_df.loc[
            (self.camp_df["campione"] == camp)
            & (self.camp_df["date"] == time),
            "tipologia",
        ].unique()

    def list_plottable_elements(self) -> np.ndarray:
        plottable = self.camp_df.loc[
            (self.camp_df["campione"] == self.campione)
            & (self.camp_df["date"] == self.datetime)
            & (self.camp_df["tipologia"] == self.tipologia),
            ["rowid", "date", "misura", "tipologia"],
        ].values

        return plottable

    def list_plottable_elements_LV(self) -> np.ndarray:
        plottable = self.camp_df.loc[
            (self.camp_df["campione"] == self.campione_LV)
            & (self.camp_df["date"] == self.datetime_LV)
            & (self.camp_df["tipologia"] != "IV_CURVE"),
            ["rowid", "date", "misura", "tipologia"],
        ].values

        return plottable

    def list_plottable_elements_df(self) -> pd.DataFrame:
        plottable = self.camp_df.loc[
            (self.camp_df["campione"] == self.campione)
            & (self.camp_df["date"] == self.datetime)
            & (self.camp_df["tipologia"] == self.tipologia),
            ["rowid", "date", "misura", "tipologia"],
        ]

        return plottable

    def list_plottable_elements_df_LV(self) -> pd.DataFrame:
        # print(f'{self.campione_LV=}')
        # print(f'{self.datetime_LV=}')
        plottable = self.camp_df.loc[
            (self.camp_df["campione"] == self.campione_LV)
            & (self.camp_df["date"] == self.datetime_LV)
            & (self.camp_df["tipologia"] != "IV_CURVE"),
            ["rowid", "date", "misura", "tipologia"],
        ]
        # print(plottable.head())
        return plottable

    def get_measures(self, rowid: int) -> pd.DataFrame:
        df = pd.read_sql(
            f"SELECT * FROM misure WHERE rowid is {rowid}",
            con=self.con,
        )
        return df

    def get_impulses(self, rowid: int) -> tuple[pd.DataFrame, np.ndarray]:
        df = pd.read_sql(f"SELECT * FROM impulsi WHERE rowid IS {rowid}", con=self.con)

        return df, self.camp_df.loc[self.camp_df["rowid"] == rowid, "gen_rep"].values

    def get_measures_LV(self, rowids: list[int] | np.ndarray) -> pd.DataFrame:
        df = pd.read_sql(
            f"SELECT * FROM campioni INNER JOIN misure using(rowid) WHERE rowid IN  ({', '.join(str(i) for i in rowids)})",
            con = self.con,
            parse_dates={"date": {"format": "%Y-%m-%d"}},
            dtype={
                "campione": "string",
                "tipologia": "category",
            },
        )
        df.sort_values(by=["date", "misura"], inplace=True)
        return df
