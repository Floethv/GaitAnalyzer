import logging

from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    Subject,
    Side,
    ReconstructionType,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # --- Defining full paths for C3D files ---
    base_data_path = f"/Users/floethv/Desktop/Doctorat/Fork/GaitAnalyzer/data/{subject.subject_name}"
    c3d_dynamic_path = f"{base_data_path}/{c3d_file_name}"
    c3d_static_path = f"{static_trial}"

    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=c3d_static_path,
        result_folder=result_folder,
        c3d_full_file_path=c3d_dynamic_path,
        static_trial_full_file_path=c3d_static_path
    )

    # Creation of model
    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        mvc_trials_path=f"{base_data_path}/mvc_trials/",
        functional_trials_path=f"{base_data_path}/functional_trials/",
        q_regularization_weight=1,
        skip_if_existing=True,
        animate_model_flag=False,
    )

    # --- Ignore certain markers and channels ---
    markers_to_ignore = ['U1', 'U2', 'U3', 'U4']
    analogs_to_ignore = [
        "Channel_01", "Channel_02", "Channel_03", "Channel_04",
        "Channel_05", "Channel_06", "Channel_07", "Channel_08",
        "Channel_09", "Channel_10", "Channel_11", "Channel_12",
        "Bertec_treadmill_speed",
    ]
    results.add_experimental_data(
        c3d_file_name=c3d_file_name,
        markers_to_ignore=markers_to_ignore,
        analogs_to_ignore=analogs_to_ignore
    )

    # --- Detection of cyclic events ---
    results.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=False, plot_phases_flag=False)

    # --- Reconstruction of the kinematics ---
    results.reconstruct_kinematics(
        reconstruction_type=[
            ReconstructionType.LSQ,
            ReconstructionType.ONLY_LM,
            ReconstructionType.LM,
            ReconstructionType.TRF,
        ],
        animate_kinematics_flag=True,
        plot_kinematics_flag=True,
        skip_if_existing=False,
    )

    # --- Biomechanical/Stability calculations ---
    results.compute_angular_momentum()
    results.compute_mechanical_energy(skip_if_existing=False)
    results.compute_mos(skip_if_existing=False)
    results.compute_dcom_mma()

    # Inverse kinetic
    results.perform_inverse_dynamics(
        skip_if_existing=False,
        reintegrate_flag=True,
        animate_dynamics_flag=True,
    )

    return results

def main():
    subjects_to_analyze = [
        Subject(
            subject_name="LED09",
            subject_height=1.71,
        )
    ]

    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=None,
        result_folder="results",
        trails_to_analyze=["Cond0001", "Cond0004"], # "Cond0002","Cond0003"
        skip_if_existing=False,
    )


if __name__ == "__main__":
    main()


