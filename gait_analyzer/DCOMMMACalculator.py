import os
import numpy as np
import pickle

from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.subject import Subject
from gait_analyzer.biomechanics_quantities.angular_momentum_calculator import AngularMomentumCalculator


class DCOMMMACalculator:
    """
    Computes the dCoM-MMA as defined by:
    dCoM-MMA = (R × dH_COM/dt) / ||R||²
    where R is the total ground reaction force obtained by summing all force plates.
    """

    def __init__(
        self,
        angular_momentum_calculator: AngularMomentumCalculator,
        experimental_data: ExperimentalData,
        subject: Subject,
        q,
        skip_if_existing: bool = False,
    ):
        self.angular_momentum_calculator = angular_momentum_calculator
        self.experimental_data = experimental_data
        self.subject_mass = subject.subject_mass
        self.f_ext = self.experimental_data.f_ext_sorted_filtered
        self.q = q
        self.nb_frames = angular_momentum_calculator.H_total.shape[1]

        # Outputs
        self.Hdot = None
        self.F_resultant = None
        self.r_MMA = None
        self.dCoM_MMA_norm = None

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_dcom_mma = True
        else:
            self.compute_dcom_mma()
            self.save_dcom_mma()

    def compute_Hdot(self):
        dt = self.experimental_data.markers_dt
        self.Hdot = np.gradient(
            self.angular_momentum_calculator.H_total, dt, axis=1
        )

    def compute_resultant_force(self):
        """
        Compute total ground reaction force R as sum over all force plates.
        self.f_ext shape = [n_plates, fx, fy, fz, n_frames] ?
        We assume forces are in indices 6:9 (fx, fy, fz) as in previous discussion.
        """
        self.F_resultant = np.sum(self.f_ext[:, 6:9, :], axis=0)

    def compute_dcom_mma(self):
        self.compute_Hdot()
        self.compute_resultant_force()

        self.r_MMA = np.zeros((3, self.nb_frames))
        self.dCoM_MMA_norm = np.zeros(self.nb_frames)

        for i in range(self.nb_frames):
            F = self.F_resultant[:, i]
            Hdot = self.Hdot[:, i]

            norm_F_sq = np.linalg.norm(F) ** 2
            if norm_F_sq < 1e-6:
                self.r_MMA[:, i] = np.nan
                self.dCoM_MMA_norm[i] = np.nan
                continue

            r_mma = np.cross(F, Hdot) / norm_F_sq
            self.r_MMA[:, i] = r_mma
            self.dCoM_MMA_norm[i] = np.linalg.norm(r_mma)

        return self.r_MMA, self.dCoM_MMA_norm

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        return f"{result_folder}/dcom_mma_{trial_name}.pkl"

    def save_dcom_mma(self):
        with open(self.get_result_file_full_path(), "wb") as file:
            pickle.dump(self.outputs(), file)

    def check_if_existing(self) -> bool:
        path = self.get_result_file_full_path()
        if os.path.exists(path):
            with open(path, "rb") as file:
                data = pickle.load(file)
                self.Hdot = data["Hdot"]
                self.F_resultant = data["F_resultant"]
                self.r_MMA = data["r_MMA"]
                self.dCoM_MMA_norm = data["dCoM_MMA_norm"]
            return True
        return False

    def outputs(self):
        return {
            "Hdot": self.Hdot,
            "F_resultant": self.F_resultant,
            "r_MMA": self.r_MMA,
            "dCoM_MMA_norm": self.dCoM_MMA_norm,
        }
