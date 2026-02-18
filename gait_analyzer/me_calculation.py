import os
import numpy as np
import pickle
import biorbd
import matplotlib.pyplot as plt

from gait_analyzer.subject import Subject
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.experimental_data import ExperimentalData

class MechanicalEnergyCalculator:
    """
    Compute the total mechanical energy:
    Em = mgh + 1/2 m V^2 + sum_s( 1/2 m_s V*^2 + 1/2 ω_s^T I_s ω_s )
    """

    def __init__(
            self,
            biorbd_model: biorbd.Model,
            experimental_data: ExperimentalData,
            kinematics_reconstructor: KinematicsReconstructor,
            subject: Subject,
            segments_data: dict,
    ):
        self.model = biorbd_model
        self.experimental_data = experimental_data
        self.q = kinematics_reconstructor.q_filtered
        self.qdot = kinematics_reconstructor.qdot
        self.subject_mass = subject.subject_mass
        self.subject_height = subject.subject_height
        self.gravity = biorbd_model.getGravity().to_array()
        self.nb_frames = self.q.shape[1]
        self.segments_data = segments_data

        self.Em = None
        self.Em_norm = None
        self.E_pot_vec = None
        self.E_kin_vec = None
        self.E_kin_global_vec = None
        self.E_pot_norm = None
        self.E_kin_norm = None

    # ------------------------------------------------------
    # Compute energy
    # ------------------------------------------------------
    def compute_mechanical_energy(self, skip_if_existing: bool = False):
        if skip_if_existing and self.check_if_existing():
            return self.Em

        nb_frames = self.nb_frames
        nb_segments = self.model.nbSegment()
        g = np.linalg.norm(self.gravity)

        Em = np.zeros(nb_frames)
        Em_norm = np.zeros(nb_frames)
        E_pot_vec = np.zeros(nb_frames)
        E_kin_vec = np.zeros(nb_frames)
        E_kin_global_vec = np.zeros(nb_frames)
        E_pot_norm = np.zeros(nb_frames)
        E_kin_norm = np.zeros(nb_frames)

        for frame in range(nb_frames):
            q_i = self.q[:, frame]
            qdot_i = self.qdot[:, frame]

            # CoM global
            com_global = self.model.CoM(q_i).to_array()
            comdot_global = self.model.CoMdot(q_i, qdot_i, True).to_array()

            h = com_global[2]
            V = np.linalg.norm(comdot_global)

            # Énergie globale du corps entier
            E_pot = self.subject_mass * g * h
            E_kin_global = 0.5 * self.subject_mass * V ** 2

            E_segments = 0

            # Contributions segmentaires
            for seg_i in range(nb_segments):
                seg_name = self.model.segment(seg_i).name().to_string()
                seg_data = self.segments_data[seg_name]

                m_s = seg_data["Masse"]

                # --- Ignorer les segments de masse quasi nulle ---
                if m_s < 1e-6:
                    continue

                I_local = seg_data["Inertie"]
                com_s = seg_data["COM"][:, frame]
                comdot_s = seg_data["COMdot"][:, frame]

                # --- Vérifier les unités ---
                if np.max(np.abs(comdot_s)) > 10:  # mm/s → m/s
                    comdot_s = comdot_s / 1000
                if np.max(np.abs(comdot_global)) > 10:
                    comdot_global = comdot_global / 1000

                # Translational segment energy
                V_rel = comdot_s - comdot_global
                E_trans = 0.5 * m_s * np.dot(V_rel, V_rel)

                # Rotational segment energy
                R_seg_global = np.array(self.model.globalJCS(q_i, seg_i).to_array())[:3, :3]
                omega_global = self.model.segmentAngularVelocity(q_i, qdot_i, seg_i).to_array()
                if np.max(np.abs(omega_global)) > 10:  # deg/s → rad/s
                    omega_global = omega_global * np.pi / 180
                omega_local = R_seg_global.T @ omega_global
                E_rot = 0.5 * omega_local.T @ I_local @ omega_local

                E_segments += E_trans + E_rot

            # Énergie cinétique totale
            E_kin_total = E_kin_global + E_segments

            # Energie mécanique totale
            Em[frame] = E_pot + E_kin_total
            Em_norm[frame] = Em[frame] / (self.subject_mass * g * self.subject_height)
            E_pot_vec[frame] = E_pot
            E_kin_vec[frame] = E_kin_total
            E_kin_global_vec[frame] = E_kin_global

            # Normalisation
            E_pot_norm[frame] = E_pot / (self.subject_mass * g * self.subject_height)
            E_kin_norm[frame] = E_kin_total #/ (self.subject_mass) #* g * self.subject_height)

        # Stockage
        self.Em = Em
        self.Em_norm = Em_norm
        self.E_pot_vec = E_pot_vec
        self.E_kin_vec = E_kin_vec
        self.E_kin_global_vec = E_kin_global_vec
        self.E_pot_norm = E_pot_norm
        #self.E_kin_norm = E_kin_norm

        return Em, Em_norm, E_pot_vec, E_kin_vec

    # ------------------------------------------------------
    # Save & load
    # ------------------------------------------------------
    def result_file_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        return f"{result_folder}/mechanical_energy_{trial_name}.pkl"

    def save(self):
        with open(self.result_file_path(), "wb") as f:
            pickle.dump({"Mechanical_energy": self.Em}, f)

    def check_if_existing(self):
        path = self.result_file_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.Em = data["Mechanical_energy"]
            return True
        return False

    def outputs(self) -> dict:
        if self.Em is None:
            return {}

        g = np.linalg.norm(self.gravity)

        return {
            "mechanical_energy": self.Em,
            "mechanical_energy_norm": self.Em_norm,
            "mechanical_energy_potential": self.E_pot_vec,
            "mechanical_energy_potential_norm": self.E_pot_norm,
            "mechanical_energy_kinetic": self.E_kin_vec,
            #"mechanical_energy_kinetic_norm": self.E_kin_norm,
            "mechanical_energy_kinetic_com": self.E_kin_global_vec
        }
