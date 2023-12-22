from sqlite3 import Connection

import numpy as np
import pandas as pd

from .plotter import Plotter


class Data:
    """A class where to store datas"""

    def __init__(self, db_name) -> None:
        self.con = Connection(db_name)
        self.cur = self.con.cursor()
        self.campione = ""
        self.datetime = ""
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
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, value) -> None:
        self._datetime = value

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

    def get_datetimes(self) -> np.ndarray:
        return self.camp_df.loc[
            self.camp_df["campione"] == self.campione, "date"
        ].unique()

    def get_tipologie(self) -> np.ndarray:
        return self.camp_df.loc[
            (self.camp_df["campione"] == self.campione)
            & (self.camp_df["date"] == self.datetime),
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

    def list_plottable_elements_df(self) -> pd.DataFrame:
        plottable = self.camp_df.loc[
            (self.camp_df["campione"] == self.campione)
            & (self.camp_df["date"] == self.datetime)
            & (self.camp_df["tipologia"] == self.tipologia),
            ["rowid", "date", "misura", "tipologia"],
        ]

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
