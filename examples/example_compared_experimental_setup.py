from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    PlotLegData,
    LegToPlot,
    PlotType,
    Subject,
    Side,
    EventIndexType,
)
from gait_analyzer.kinematics_reconstructor import ReconstructionType


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range,
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
    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        skip_if_existing=False,
        skip_scaling=True,  # We skip the scaling since it was already done in OpenSim's GUI
        animate_model_flag=False)
    results.add_experimental_data(
        c3d_file_name=c3d_file_name, markers_to_ignore=["U1", "U2", "U3", "U4"], animate_c3d_flag=False
    )
    if "Vicon" in c3d_file_name:
        results.add_unique_events(skip_if_existing=True, plot_phases_flag=False)
    elif "Qualisys" in c3d_file_name:
        results.add_cyclic_events(force_plate_sides=[Side.RIGHT, Side.LEFT], skip_if_existing=False,
                                  plot_phases_flag=False)
    else:
        raise RuntimeError("To compare both experimental setups, the name of the results files must contain 'Vicon' or 'Qualisys'")

    results.reconstruct_kinematics(
        reconstruction_type=[ReconstructionType.ONLY_LM, ReconstructionType.LM, ReconstructionType.TRF],
        animate_kinematics_flag=False,
        plot_kinematics_flag=True,
        skip_if_existing=True,
    )
    results.compute_angular_momentum()
    return results


if __name__ == "__main__":

    # --- Create the list of participants --- #
    subjects_to_analyze = []
    # Inputs to correct for AOT_01: dominant_leg, subject_height
    subjects_to_analyze.append(Subject(subject_name="AOT_01", subject_mass=69.2, subject_height=1.84, dominant_leg=Side.RIGHT, preferential_speed=1.06))
    # Inputs to correct for VIF_04: preferential_speed, dominant_leg
    # subjects_to_analyze.append(
    #     Subject(subject_name="VIF_04", subject_mass=71.0,subject_height=1.84, dominant_leg=Side.RIGHT, preferential_speed=1.06)  # ?  # ?
    # )
    # ... add other participants here

    # --- Example of how to run the analysis --- #
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=None,
        result_folder="results",
        trails_to_analyze=["_L400_F50_I60_Qualisys"],  # If not specified, all trials will be analyzed
        skip_if_existing=False,
    )

    # --- Example of how to plot the joint angles --- #
    plot = PlotLegData(
        result_folder="results",
        leg_to_plot=LegToPlot.RIGHT,
        plot_type=PlotType.Q,
        unique_event_to_split={
            "event_index_type": EventIndexType.MARKERS,
            "start": lambda data: int(data["events"][0]["heel_touch"][0]),
            "stop": lambda data: int(data["events"][2]["heel_touch"][0]),
        },
        conditions_to_compare=["_Cond0006"],
    )
    plot.draw_plot()
    plot.save("results/AOT_01_Q_plot_temporary.png")
    plot.show()

