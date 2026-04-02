import numpy as np
from scipy.stats import norm
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.subject import Subject
from gait_analyzer.biomechanics_quantities.marginofstability_calculator import MarginofStabilityCalculation

class ProbabilityOfInstability:
    """
    Computes the Probability of Instability (PoI) based on ML and AP Margin of Stability (MoS)

    Steps:
    1. Detect heel strikes from GRF
    2. Compute one MoS value per step
    3. Estimate distribution (mu, sigma)
    4. Compute PoI = P(MoS < 0)
    """

    def __init__(
        self,
        marginofstability_calculator: MarginofStabilityCalculation,
        experimental_data: ExperimentalData,
        subject: Subject,
        skip_if_existing: bool,
    ):
        self.marginofstability_calculator = marginofstability_calculator
        self.experimental_data = experimental_data
        self.subject_mass = subject.subject_mass
        self.subject_height = subject.subject_height
        self.f_foot1 = self.experimental_data.f_ext_sorted_filtered[0, 6:9, :]
        self.f_foot2 = self.experimental_data.f_ext_sorted_filtered[1, 6:9, :]
        self.ML_MoS = self.marginofstability_calculator.ML_MoS
        self.AP_MoS = self.marginofstability_calculator.AP_MoS

        self.heel_strikes = None
        self.mos_steps_AP = None
        self.mos_steps_ML = None
        self.mu_ML = None
        self.mu_AP = None
        self.sigma_ML = None
        self.sigma_AP = None
        self.PoI_ML = None
        self.PoI_AP = None
        self.PoI_empirical_ML = None
        self.PoI_empirical_AP = None

        if skip_if_existing and self.check_if_existing():
            self.is_loaded_mos = True
        else:
            self.detect_heel_strikes()
            self.compute_mos_per_step()
            self.compute_poi()


    def detect_heel_strikes(self):
        """
        Detect heel strikes for both feet using vertical GRF threshold crossing
        """
        fv_foot1 = self.f_foot1[2, :]
        fv_foot2 = self.f_foot2[2, :]

        threshold = 20

        contact1 = fv_foot1 > threshold
        contact2 = fv_foot2 > threshold

        hs_foot1 = []
        hs_foot2 = []

        for i in range(1, len(contact1) - 1):
            if contact1[i] and not contact1[i - 1]:
                if fv_foot1[i + 1] > threshold:
                    hs_foot1.append(i)

        for i in range(1, len(contact2) - 1):
            if contact2[i] and not contact2[i - 1]:
                if fv_foot2[i + 1] > threshold:
                    hs_foot2.append(i)

        hs_foot1 = np.array(hs_foot1)
        hs_foot2 = np.array(hs_foot2)

        hs_all = np.concatenate([
            np.vstack((hs_foot1, np.ones(len(hs_foot1)))).T,
            np.vstack((hs_foot2, np.zeros(len(hs_foot2)))).T
        ])

        hs_all = hs_all[np.argsort(hs_all[:, 0])]

        self.heel_strikes_all = hs_all  # [frame, foot_id]

    def compute_mos_per_step(self):
        """
        Average ML and AP MoS per step, handling different sampling rates
        between GRF (heel strikes) and MoS.
        """
        mos_steps_ML = []
        mos_steps_AP = []

        fs_grf = 2000
        fs_mos = 100
        factor = fs_grf / fs_mos

        heel_strikes_mos = (self.heel_strikes_all[:, 0] / factor).astype(int)

        for i in range(len(heel_strikes_mos) - 1):
            start = heel_strikes_mos[i]
            end = heel_strikes_mos[i + 1]

            if end <= start:
                continue

            mos_segment_ML = self.ML_MoS[start:end]
            mos_segment_AP = self.AP_MoS[start:end]

            # Calculer moyenne en ignorant les NaN
            if not np.all(np.isnan(mos_segment_ML)):
                mos_steps_ML.append(np.nanmean(mos_segment_ML))
            if not np.all(np.isnan(mos_segment_AP)):
                mos_steps_AP.append(np.nanmean(mos_segment_AP))

        self.mos_steps_ML = np.array(mos_steps_ML)
        self.mos_steps_AP = np.array(mos_steps_AP)

        print(f"Nb segments ML: {len(self.mos_steps_ML)}, Nb segments AP: {len(self.mos_steps_AP)}")

    def compute_poi(self):
        """
        Compute PoI from MoS distribution
        """
        # ML
        if len(self.mos_steps_ML) < 2:
            self.mu_ML = np.nan
            self.sigma_ML = np.nan
            self.PoI_ML = np.nan
            return

        self.mu_ML = np.mean(self.mos_steps_ML)
        self.sigma_ML = np.std(self.mos_steps_ML, ddof=1)

        if self.sigma_ML > 0:
            z = (0 - self.mu_ML) / self.sigma_ML
            self.PoI_ML = norm.cdf(z)
        else:
            self.PoI_ML = np.nan

        self.PoI_ML = (np.sum(self.mos_steps_ML < 0) / len(self.mos_steps_ML))*100

        # AP
        if len(self.mos_steps_AP) < 2:
            self.mu_AP = np.nan
            self.sigma_AP = np.nan
            self.PoI_AP = np.nan
            return

        self.mu_AP = np.mean(self.mos_steps_AP)
        self.sigma_AP = np.std(self.mos_steps_AP, ddof=1)

        if self.sigma_AP > 0:
            z = (0 - self.mu_AP) / self.sigma_AP
            self.PoI_AP = norm.cdf(z)
        else:
            self.PoI_AP = np.nan

        self.PoI_AP = (np.sum(self.mos_steps_AP < 0) / len(self.mos_steps_AP))*100

    def outputs(self):
        return {
            "PoI_ML": self.PoI_ML,
            "PoI_AP": self.PoI_AP,
        }
