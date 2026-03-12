import os
import numpy as np
import pickle
import biorbd

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

        self.mechanical_energy = None
        self.mechanical_energy_normalized = None
        self.E_pot_vec = None
        self.E_kin_vec = None
        self.E_kin_global_vec = None
        self.E_pot_normalized = None
        self.E_kin_normalized = None

    def compute_mechanical_energy(self, skip_if_existing: bool = False):
        if skip_if_existing and self.check_if_existing():
            return self.mechanical_energy

        nb_frames = self.nb_frames
        nb_segments = self.model.nbSegment()
        g = np.linalg.norm(self.gravity)

        mechanical_energy = np.zeros(nb_frames)
        mechanical_energy_normalized = np.zeros(nb_frames)
        E_pot_vec = np.zeros(nb_frames)
        E_kin_vec = np.zeros(nb_frames)
        E_kin_global_vec = np.zeros(nb_frames)
        E_pot_normalized = np.zeros(nb_frames)
        E_kin_normalized = np.zeros(nb_frames)

        for frame in range(nb_frames):
            q_i = self.q[:, frame]
            qdot_i = self.qdot[:, frame]

            com_global = self.model.CoM(q_i).to_array()
            comdot_global = self.model.CoMdot(q_i, qdot_i, True).to_array()

            h = com_global[2]
            V = np.linalg.norm(comdot_global)

            E_pot = self.subject_mass * g * h
            E_kin_global = 0.5 * self.subject_mass * V ** 2

            E_segments = 0

            for seg_i in range(nb_segments):
                seg_name = self.model.segment(seg_i).name().to_string()
                seg_data = self.segments_data[seg_name]

                m_s = seg_data["Masse"]

                if m_s < 1e-6:
                    continue

                I_local = seg_data["Inertie"]
                com_s = seg_data["COM"][:, frame]
                comdot_s = seg_data["COMdot"][:, frame]

                if np.max(np.abs(comdot_s)) > 10:  # mm/s → m/s
                    comdot_s = comdot_s / 1000
                if np.max(np.abs(comdot_global)) > 10:
                    comdot_global = comdot_global / 1000

                V_rel = comdot_s - comdot_global
                E_trans = 0.5 * m_s * np.dot(V_rel, V_rel)

                R_seg_global = np.array(self.model.globalJCS(q_i, seg_i).to_array())[:3, :3]
                omega_global = self.model.segmentAngularVelocity(q_i, qdot_i, seg_i).to_array()
                if np.max(np.abs(omega_global)) > 10:  # deg/s → rad/s
                    omega_global = omega_global * np.pi / 180
                omega_local = R_seg_global.T @ omega_global
                E_rot = 0.5 * omega_local.T @ I_local @ omega_local

                E_segments += E_trans + E_rot

            E_kin_total = E_kin_global + E_segments

            mechanical_energy[frame] = E_pot + E_kin_total
            mechanical_energy_normalized[frame] = mechanical_energy[frame] / (self.subject_mass * g * self.subject_height)
            E_pot_vec[frame] = E_pot
            E_kin_vec[frame] = E_kin_total
            E_kin_global_vec[frame] = E_kin_global

            E_pot_normalized[frame] = E_pot / (self.subject_mass * g * self.subject_height)
            E_kin_normalized[frame] = E_kin_total / (self.subject_mass * g * self.subject_height)

        self.mechanical_energy = mechanical_energy
        self.mechanical_energy_normalized = mechanical_energy_normalized
        self.E_pot_vec = E_pot_vec
        self.E_kin_vec = E_kin_vec
        self.E_kin_global_vec = E_kin_global_vec
        self.E_pot_normalized = E_pot_normalized
        self.E_kin_normalized = E_kin_normalized

        return mechanical_energy, mechanical_energy_normalized, E_pot_vec, E_kin_vec

    def result_file_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        return f"{result_folder}/mechanical_energy_{trial_name}.pkl"

    def save(self):
        with open(self.result_file_path(), "wb") as f:
            pickle.dump({"Mechanical_energy": self.mechanical_energy}, f)

    def check_if_existing(self):
        path = self.result_file_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.mechanical_energy = data["Mechanical_energy"]
            return True
        return False

    def outputs(self) -> dict:
        if self.mechanical_energy is None:
            return {}

        g = np.linalg.norm(self.gravity)

        return {
            "mechanical_energy": self.mechanical_energy,
            "mechanical_energy_normalized": self.mechanical_energy_normalized,
            "mechanical_energy_potential": self.E_pot_vec,
            "mechanical_energy_potential_normalized": self.E_pot_normalized,
            "mechanical_energy_kinetic": self.E_kin_vec,
            "mechanical_energy_kinetic_normalized": self.E_kin_normalized,
            "mechanical_energy_kinetic_com": self.E_kin_global_vec
        }
