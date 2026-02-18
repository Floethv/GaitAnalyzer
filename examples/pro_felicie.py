import logging

from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    Subject,
    ReconstructionType,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
):

    # --- Example of analysis that must be performed in order --- #
    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=static_trial,
        result_folder=result_folder,
    )
    # # This step is to show the markers and eventually change their labeling manually
    # marker_handler = MarkerLabelingHandler("path_to_the_c3d_you_want_to_check.c3d")
    # marker_handler.show_marker_labeling_plot()
    # marker_handler.invert_marker_labeling([name_of_the_marker, name_of_another_marker], frame_start=0, frame_end=100)
    # marker_handler.save_c3d(output_c3d_path)
    # # The hip SCoREs did not seem to help improve the models, so they were not use din this example.
    # marker_handler = MarkerLabelingHandler("/Users/floethv/Desktop/Doctorat/Fork/GaitAnalyzer/data/LAO01/LAO01_static.c3d")
    # marker_handler.show_marker_labeling_plot()

    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        mvc_trials_path=f"../data/{subject.subject_name}/MVC_trials/",
        # functional_trials_path=None,  # If you want to skip the functional trials for this example
        q_regularization_weight=1,
        functional_trials_path=None,
        skip_if_existing=True,
        animate_model_flag=True,
    )
    markers_to_ignore = ['LBARRE', 'RBARRE']

    results.add_experimental_data(c3d_file_name=c3d_file_name, markers_to_ignore=markers_to_ignore)

    results.reconstruct_kinematics(
        reconstruction_type=[
            ReconstructionType.LSQ,
            ReconstructionType.ONLY_LM,
            ReconstructionType.LM,
            ReconstructionType.TRF,
        ],
        animate_kinematics_flag=True,
        plot_kinematics_flag=False,
        skip_if_existing=True,
    )

    results.perform_inverse_dynamics(
        skip_if_existing=True,
        reintegrate_flag=True,
        animate_dynamics_flag=True,
    )

    return results


def main():
    # info subjects
    subjects_to_analyze = []
    subjects_to_analyze.append(
        Subject(
            subject_name="S04",
            subject_height=1.71,
        )
    )
    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=None,
        result_folder="results",
        trails_to_analyze=["_charge"],
        skip_if_existing=False,
    )


if __name__ == "__main__":
    main()