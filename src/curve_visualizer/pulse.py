from typing import Tuple

import numpy as np
import pandas as pd


class Pulse:
    def __init__(
        self,
        n_rep: int,
        df_pulses: pd.DataFrame | None = None,
        str_header: str | None = None,
    ) -> None:
        self.n_rep = n_rep
        self.df_pulses = df_pulses
        self.str_header = str_header

    def from_dataframe_to_array(self) -> None:
        self.v_params = self.df_pulses.iloc[:, 1:].values
        self.v_period = self.df_pulses["periodoTot"].values
        self.v_pulse = self.df_pulses["activeTime"].values
        self.v_amp = self.df_pulses["voltage"].values
        self.v_pulse_rep = self.df_pulses["rip"].astype('int64').values

    def from_header_to_array(self, n_pulses: int) -> np.ndarray:
        params = np.array(
            [p.split() for p in self.str_header[:n_pulses]], dtype=np.float64
        )
        self.v_params = params
        self.v_period = self.v_params[:, 0]
        self.v_pulse = self.v_params[:, 2]
        self.v_amp = self.v_params[:, 4]
        self.v_pulse_rep = self.v_params[:, 7].astype(np.int64)

    def get_times(self) -> np.ndarray:
        v_dead_time = self.v_period - self.v_pulse

        # ^
        # |
        # |          #2--------#3
        # |           |         |
        # |           |         |
        # | #0-------#1        #0--
        # ------------------------->
        #    0       Dead       Period = Dead + Pulse

        # Ogni impulso richiede 4 punti per essere disegnato
        v_times = np.zeros(self.v_pulse_rep.sum() * 4 * self.n_rep)
        # Matrice di singole celle temporali
        m_single_cell = np.zeros((self.v_period.size, 4))

        # per ragioni di calcolo questa è la cella degli incrementi temporali
        # [0 Dead 0 Pulse]
        m_single_cell[:, 1] = v_dead_time

        m_single_cell[:, 3] = self.v_pulse

        # Ripeto il per le ripetizioni interne
        v_times_not_repeated = np.repeat(m_single_cell, self.v_pulse_rep, axis=0)

        # Ripeto con le ripetizioni generali
        v_times = np.tile(v_times_not_repeated, (self.n_rep, 1)).ravel()

        # Aggiungiamo a ogni riga la somma dei tempi precedenti
        v_times = np.tri(v_times.size) @ v_times
        return v_times

    def get_voltage(self) -> np.ndarray:
        # ^
        # |
        # |          #2--------#3
        # |           |         |
        # |           |         |
        # | #0-------#1        #0--
        # ------------------------->
        #    0       Dead       Period = Dead + Pulse

        # Ogni impulso richiede 4 punti per essere disegnato
        v_repeted_amps = np.zeros_like(self.v_pulse_rep.sum() * 4 * self.n_rep)

        m_single_cell = np.zeros((self.v_period.size, 4))
        m_single_amp = np.zeros_like(m_single_cell)

        m_single_amp[:, 2:] = np.tile(self.v_amp, (2, 1)).T

        v_amps_not_repeated = np.repeat(m_single_amp, self.v_pulse_rep, axis=0)

        # Ripeto con le ripetizioni generali
        v_repeted_amps = np.tile(v_amps_not_repeated, (self.n_rep, 1)).ravel()

        return v_repeted_amps

    def get_time_offset(self) -> np.float64:
        pulse_mean = np.mean(np.abs(self.v_amp))
        read_pos = np.argmax(np.abs(self.v_amp) < pulse_mean)
        v_dead_time = self.v_period - self.v_pulse
        time_offset = np.inner(self.v_period[:read_pos], self.v_pulse_rep[:read_pos ]) + v_dead_time[read_pos]

        # time_offset = np.inner(self.v_period, self.v_pulse_rep) - self.v_period[-1]
        return time_offset

    def from_array_to_func(self, v_params: np.ndarray, n_rep: int) -> Tuple[np.ndarray]:
        v_period = v_params[:, 0]
        v_pulse = v_params[:, 2]
        v_amp = v_params[:, 4]
        v_pulse_rep = v_params[:, 7].astype(np.int64)

        v_dead_time = v_period - v_pulse

        # ^
        # |
        # |          #2--------#3
        # |           |         |
        # |           |         |
        # | #0-------#1        #0--
        # ------------------------->
        #    0       Dead       Period = Dead + Pulse

        # Ogni impulso richiede 4 punti per essere disegnato
        v_times = np.zeros(v_pulse_rep.sum() * 4 * n_rep)
        v_repeted_amps = np.zeros_like(v_times)

        m_single_cell = np.zeros((v_period.size, 4))
        m_single_amp = np.zeros_like(m_single_cell)

        m_single_amp[:, 2:] = np.tile(v_amp, (2, 1)).T

        # per ragioni di calcolo questa è la cella degli incrementi temporali
        # [0 Dead 0 Pulse]
        m_single_cell[:, 1] = v_dead_time

        m_single_cell[:, 3] = v_pulse

        # Ripeto il per le ripetizioni interne
        v_times_not_repeated = np.repeat(m_single_cell, v_pulse_rep, axis=0)
        v_amps_not_repeated = np.repeat(m_single_amp, v_pulse_rep, axis=0)

        # Ripeto con le ripetizioni generali
        v_times = np.tile(v_times_not_repeated, (n_rep, 1)).ravel()
        v_repeted_amps = np.tile(v_amps_not_repeated, (n_rep, 1)).ravel()

        # Aggiungiamo a ogni riga la somma dei tempi precedenti
        v_times = np.tri(v_times.size) @ v_times

        return v_times, v_repeted_amps
