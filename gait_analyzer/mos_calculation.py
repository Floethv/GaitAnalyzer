import numpy as np
import pickle
import os
import ezc3d

class MosCalculation:
    def __init__(self, model, markers_sorted, model_marker_names, q, qdot,
                 experimental_data):
        self.model = model
        self.markers_sorted = markers_sorted
        self.model_marker_names = model_marker_names
        self.q = q
        self.qdot = qdot
        self.experimental_data = experimental_data
        self.f_ext = self.experimental_data.f_ext_sorted_filtered
        #m/s (0 si inconnue)

        self.AP_MoS = None
        self.ML_MoS = None

    # --------------------------------------------------
    def _idx(self, name):
        if name not in self.model_marker_names.index:
            raise RuntimeError(f"Marker {name} not found")
        return self.model_marker_names.index.index(name)

    # ------------------------------------------
    def compute_mos(self, threshold=15):
        n_frames = self.q.shape[1]
        g = 9.81

        # --- indices des marqueurs ---
        idx_LCAL = self.model_marker_names.index("LCAL")
        idx_RCAL = self.model_marker_names.index("RCAL")
        idx_LTT2 = self.model_marker_names.index("LTT2")
        idx_RTT2 = self.model_marker_names.index("RTT2")
        idx_LMFH5 = self.model_marker_names.index("LMFH5")
        idx_RMFH5 = self.model_marker_names.index("RMFH5")

        AP_MoS = np.full(n_frames, np.nan)
        ML_MoS = np.full(n_frames, np.nan)
        xcom = np.full(n_frames, np.nan)
        ycom = np.full(n_frames, np.nan)
        bos_AP = np.full(n_frames, np.nan)
        bos_ML = np.full(n_frames, np.nan)

        c3d_static = ezc3d.c3d(self.experimental_data.model_creator.static_trial)

        markers = c3d_static["data"]["points"][:3, :, :]
        exp_marker_names = c3d_static["parameters"]["POINT"]["LABELS"]["value"]

        marker_units = 1.0
        if c3d_static["parameters"]["POINT"]["UNITS"]["value"][0] == "mm":
            marker_units = 0.001
        markers *= marker_units

        n_frames_static = markers.shape[2]
        markers_sorted_static = np.full(
            (3, len(self.model_marker_names), n_frames_static),
            np.nan
        )

        for i_model, name in enumerate(self.model_marker_names):
            if name not in exp_marker_names:
                raise ValueError(f"Marker {name} in model but not in static C3D")
            i_exp = exp_marker_names.index(name)
            markers_sorted_static[:, i_model, :] = markers[:, i_exp, :]

        # --- Longueur du pendule inversé ---
        LCAL_pos = markers_sorted_static[:, idx_LCAL, :]
        RCAL_pos = markers_sorted_static[:, idx_RCAL, :]
        com0 = self.model.CoM(self.q[:, 0]).to_array()
        trochanteric_height = np.mean([np.linalg.norm(com0[2] - LCAL_pos[2, 0]),
                                       np.linalg.norm(com0[2] - RCAL_pos[2, 0])])
        l_AP = 1.24 * trochanteric_height
        l_ML = 1.34 * trochanteric_height
        omega_AP = np.sqrt(g / l_AP)
        omega_ML = np.sqrt(g / l_ML)

        # --- Détection de l'appui via les forces ---
        fz_left = self.f_ext[0, 8, :]
        fz_right = self.f_ext[1, 8, :]
        left_stance = fz_left > threshold
        right_stance = fz_right > threshold

        # --- Boucle sur toutes les frames ---
        for frame in range(n_frames):
            com = self.model.CoM(self.q[:, frame]).to_array()
            comdot = self.model.CoMdot(self.q[:, frame], self.qdot[:, frame], True).to_array()

            # XCoM
            xcom[frame] = com[0] + comdot[0] / omega_AP
            ycom[frame] = com[1] + comdot[1] / omega_ML

            # Marqueurs BOS
            RCAL = self.markers_sorted[:, idx_RCAL, frame]
            LCAL = self.markers_sorted[:, idx_LCAL, frame]
            RTT2 = self.markers_sorted[:, idx_RTT2, frame]
            LTT2 = self.markers_sorted[:, idx_LTT2, frame]

            # --- Détermination du BOS ---
            if left_stance[frame] and not right_stance[frame]:
                bos_AP[frame] = LTT2[0]
                bos_ML[frame] = LCAL[1]
            elif right_stance[frame] and not left_stance[frame]:
                bos_AP[frame] = RTT2[0]
                bos_ML[frame] = RCAL[1]
            elif left_stance[frame] and right_stance[frame]:
                if RCAL[0] >= LCAL[0]:
                    bos_AP[frame] = RTT2[0]
                    bos_ML[frame] = RCAL[1]
                else:
                    bos_AP[frame] = LTT2[0]
                    bos_ML[frame] = LCAL[1]

        # --- Calcul du MoS par frame ---
        AP_MoS = bos_AP - xcom
        ML_MoS = bos_ML - ycom

        self.AP_MoS = AP_MoS
        self.ML_MoS = ML_MoS
        # --- Plot ---
        time = np.arange(n_frames)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(14, 6))

        # AP
        plt.subplot(2, 1, 1)
        plt.plot(time, xcom, label='XCoM', color='blue')
        plt.plot(time, bos_AP, label='BOS AP', color='red', linestyle='--')
        plt.ylabel('AP (m)')
        plt.title('Centre de masse et BOS en AP')
        plt.legend()
        plt.grid(True)

        # ML
        plt.subplot(2, 1, 2)
        plt.plot(time, ycom, label='YCoM', color='blue')
        plt.plot(time, bos_ML, label='BOS ML', color='red', linestyle='--')
        plt.xlabel('Frame')
        plt.ylabel('ML (m)')
        plt.title('Centre de masse et BOS en ML')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        return AP_MoS, ML_MoS

    # ------------------------------------------
    # Sauvegarde et chargement
    # ------------------------------------------
    def result_file_path(self, result_folder=None):
        if result_folder is None:
            result_folder = self.experimental_data.result_folder
        trial_name = os.path.basename(self.experimental_data.c3d_full_file_path)[:-4]
        return os.path.join(result_folder, f"mos_{trial_name}.pkl")

    def save(self):
        with open(self.result_file_path(), "wb") as f:
            pickle.dump({"AP_MoS": self.AP_MoS, "ML_MoS": self.ML_MoS}, f)

    def check_if_existing(self):
        path = self.result_file_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.AP_MoS = data.get("AP_MoS", None)
            self.ML_MoS = data.get("ML_MoS", None)
            return True
        return False