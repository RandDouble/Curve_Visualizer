from sqlite3 import Connection, Cursor

import numpy as np
import pandas as pd

from enum import Enum

from .plotter import Plotter
from .local_constants import SCALE, RESISTANCE

class ViewType(Enum):
    SV = 0
    LV = 1


class Data:
    """A class where to store datas"""

    def __init__(self, db_name) -> None:
        self.con = Connection(db_name)
        self.cur : Cursor = self.con.cursor()
        self.campione = ""
        self.campione_LV = ""
        self.datetime = ""
        self.datetime_LV = ""
        self.tipologia = ""
        self.plot = Plotter()
        self._plottable_max: float | np.float64 = 0.0
        self._plottable_min: float | np.float64 = 0.0
        self.max: float | np.float64 = 0.0
        self.min: float | np.float64 = 0.0
        self.scale: int = 0

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

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, value):
        if value <= self._plottable_max + 1e-12 and value > self._plottable_min - 1e-12:
            self._max = value
        else:
            print("value error, setting on plottable max")
            self._max = self._plottable_max

    @property
    def plottable_max(self):
        return self._plottable_max

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, value):
        if value < self._plottable_max + 1e-12 and value >= self._plottable_min - 1e-12:
            self._min = value
        else:
            print("value error, setting on plottable min")
            self._min = self._plottable_min

    @property
    def plottable_min(self):
        return self._plottable_min

    def get_camp_df(self) -> None:
        self.camp_df = pd.read_sql(
            "SELECT * FROM campioni",
            con=self.con,
            parse_dates={"date": {"format": "%Y-%m-%d"}},
            dtype={"campione": "string", "tipologia": "category", "gen_rep": "int64"},
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

    def get_tipologie(self, view_type: ViewType) -> np.ndarray:
        match view_type:
            case ViewType.SV:
                camp = self.campione
                time = self.datetime
            case ViewType.LV:
                camp = self.campione_LV
                time = self.datetime_LV

        return self.camp_df.loc[
            (self.camp_df["campione"] == camp) & (self.camp_df["date"] == time),
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
        plottable.sort_values(by="misura", inplace=True)
        return plottable

    def list_plottable_elements_df_LV(self) -> pd.DataFrame:
        plottable = self.camp_df.loc[
            (self.camp_df["campione"] == self.campione_LV)
            & (self.camp_df["date"] == self.datetime_LV)
            & (self.camp_df["tipologia"] != "IV_CURVE"),
            ["rowid", "date", "misura", "tipologia"],
        ]
        return plottable

    def get_measures(self, rowid: int) -> pd.DataFrame:
        df = pd.read_sql(
            f"SELECT *, voltage / current as {RESISTANCE} FROM misure WHERE rowid is {rowid}",
            con=self.con,
        )

        df[SCALE] = np.log10(
            np.abs(df[RESISTANCE]) + 1e-18,
        )

        power = np.floor(df[SCALE].mean())
        scale = power - power % 3

        self._plottable_max = df[RESISTANCE].max() / 10**scale
        self._plottable_min = df[RESISTANCE].min() / 10**scale

        self.min = self._plottable_min
        self.max = self._plottable_max

        return df, scale

    def get_impulses(self, rowid: int) -> tuple[pd.DataFrame, np.ndarray]:
        df = pd.read_sql(f"SELECT * FROM impulsi WHERE rowid IS {rowid}", con=self.con)

        return df, self.camp_df.loc[self.camp_df["rowid"] == rowid, "gen_rep"].values

    def get_measures_LV(self, rowids: list[int] | np.ndarray) -> pd.DataFrame:
        df = pd.read_sql(
            f"SELECT *, misure.voltage / misure.current as resistance FROM campioni INNER JOIN misure using(rowid) WHERE rowid IN  ({', '.join(str(i) for i in rowids)})",
            con=self.con,
            parse_dates={"date": {"format": "%Y-%m-%d"}},
            dtype={
                "campione": "string",
                "tipologia": "category",
            },
        )
        df.sort_values(by=["date", "misura"], inplace=True)
        return df
