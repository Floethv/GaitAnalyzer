import os
import numpy as np
import pickle
import biorbd

from gait_analyzer.subject import Subject
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.experimental_data import ExperimentalData


class AngularMomentumCalculator:
    """
    Computes the angular momentum (H) of the whole body and of each segment around the global center of mass (CoM).
    """

    def __init__(
        self,
        biorbd_model: biorbd.Model,
        experimental_data: ExperimentalData,
        kinematics_reconstructor: KinematicsReconstructor,
        subject: Subject,
        skip_if_existing: bool,
    ):
        self.model = biorbd_model
        self.experimental_data = experimental_data
        self.q = kinematics_reconstructor.q_filtered
        self.qdot = kinematics_reconstructor.qdot
        self.subject_mass = subject.subject_mass
        self.subject_height = subject.subject_height
        self.gravity = biorbd_model.getGravity().to_array()
        self.nb_frames = self.q.shape[1]
        self.dof_names = [m.to_string() for m in self.model.nameDof()]

        # Outputs
        self.H_segments = None
        self.H_total = None
        self.segments_data = None
        self.total_angular_momentum_normalized = None

        # Chargement si déjà existant
        if skip_if_existing and self.check_if_existing():
            self.is_loaded_angular_momentum = True
        else:
            self.calculate_H_and_angular_momentum()
            self.normalize_total_angular_momentum()
            self.save_angular_momentum()

    def calculate_H_and_angular_momentum(self):
        """
        Calcule le moment cinétique segmentaire et total autour du CoM global.
        """
        segments_data = {}
        H_segments = {}
        nb_segments = self.model.nbSegment()
        nb_frames = self.nb_frames
        H_total = np.zeros((3, nb_frames))

        # Initialisation par segment
        for segment_i in range(nb_segments):
            seg = self.model.segment(segment_i)
            char = seg.characteristics()
            seg_name = seg.name().to_string()

            segments_data[seg_name] = {
                "Masse": char.mass(),
                "Inertie": np.array(char.inertia().to_array())[:3, :3],
                "COM": np.zeros((3, nb_frames)),
                "COMdot": np.zeros((3, nb_frames)),
            }
            H_segments[seg_name] = np.zeros((3, nb_frames))

        # Boucle temporelle
        for frame_i in range(nb_frames):
            q_i = self.q[:, frame_i]
            qdot_i = self.qdot[:, frame_i]

            # CoM global et vitesse du CoM global
            com_global = self.model.CoM(q_i).to_array()
            comdot_global = self.model.CoMdot(q_i, qdot_i, True).to_array()

            H_frame_total = np.zeros(3)

            # Boucle segments
            for segment_i in range(nb_segments):
                seg = self.model.segment(segment_i)
                seg_name = seg.name().to_string()
                mass = segments_data[seg_name]["Masse"]
                I_local = segments_data[seg_name]["Inertie"]

                # COM segmentaire et vitesse
                com_seg = self.model.CoMbySegment(q_i, segment_i, True).to_array()
                comdot_seg = self.model.CoMdotBySegment(q_i, qdot_i, segment_i, True).to_array()

                segments_data[seg_name]["COM"][:, frame_i] = com_seg
                segments_data[seg_name]["COMdot"][:, frame_i] = comdot_seg

                # Matrice de rotation globale du segment
                R_seg_global = np.array(self.model.globalJCS(q_i, segment_i).to_array())[:3, :3]

                # Vitesse angulaire globale puis locale
                omega_global = self.model.segmentAngularVelocity(q_i, qdot_i, segment_i).to_array()
                omega_local = R_seg_global.T @ omega_global

                # Moment cinétique de rotation (local → global)
                H_rot_local = I_local @ omega_local
                H_rot_global = R_seg_global @ H_rot_local

                # Relation de transport
                H_trans = np.cross(com_seg - com_global, mass * (comdot_seg - comdot_global))

                # Moment cinétique total du segment
                H_seg = H_rot_global + H_trans
                H_segments[seg_name][:, frame_i] = H_seg


                # Ajout au moment total du corps
                H_frame_total += H_seg

            H_total[:, frame_i] = H_frame_total

        self.H_segments = H_segments
        self.H_total = H_total
        self.segments_data = segments_data

        return segments_data, H_segments, H_total

    def normalize_total_angular_momentum(self):
        """
        Normalise le moment cinétique total par m * h * sqrt(g*h).
        """
        g_norm = np.linalg.norm(self.gravity)
        normalization_factor = self.subject_mass * self.subject_height * np.sqrt(g_norm * self.subject_height)
        self.total_angular_momentum_normalized = self.H_total / normalization_factor

    def check_if_existing(self) -> bool:
        result_file_full_path = self.get_result_file_full_path()
        if os.path.exists(result_file_full_path):
            with open(result_file_full_path, "rb") as file:
                data = pickle.load(file)
                self.H_segments = data["H_segments"]
                self.H_total = data["H_total"]
                self.segments_data = data["segments_data"]
                self.total_angular_momentum_normalized = data["total_angular_momentum_normalized"]
            return True
        return False

    def get_result_file_full_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = self.experimental_data.c3d_full_file_path.split("/")[-1][:-4]
        return f"{result_folder}/ang_mom_{trial_name}.pkl"

    def save_angular_momentum(self):
        result_file_full_path = self.get_result_file_full_path()
        with open(result_file_full_path, "wb") as file:
            pickle.dump(self.outputs(), file)

    def outputs(self):
        return {
            "H_segments": self.H_segments,
            "H_total": self.H_total,
            "segments_data": self.segments_data,
            "total_angular_momentum_normalized": self.total_angular_momentum_normalized,
            "DoF_names" : self.dof_names
        }
