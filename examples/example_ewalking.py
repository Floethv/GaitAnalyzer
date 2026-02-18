import logging

from gait_analyzer import (
    ResultManager,
    OsimModels,
    AnalysisPerformer,
    inverse_dynamics_performer,
    PlotBiomechanicsQuantity,
    PlotType,
    Subject,
    Side,
    ReconstructionType,
    OrganizedResult,
    StatsPerformer,
    QuantityToExtractType,
    StatsType,
    MarkerLabelingHandler,
)


def analysis_to_perform(
    subject: Subject,
    cycles_to_analyze: range | None,
    static_trial: str,
    c3d_file_name: str,
    result_folder: str,
    treadmill_speed: float = 0.0,  # <-- nouvelle variable pour vT
):

    # --- Définition des chemins complets pour les fichiers C3D ---
    base_data_path = f"/Users/floethv/Desktop/Doctorat/Fork/GaitAnalyzer/data/{subject.subject_name}"
    c3d_dynamic_path = f"{base_data_path}/{c3d_file_name}"
    c3d_static_path = f"{static_trial}"

    # --- Création du ResultManager ---
    results = ResultManager(
        subject=subject,
        cycles_to_analyze=cycles_to_analyze,
        static_trial=c3d_static_path,
        result_folder=result_folder,
        c3d_full_file_path=c3d_dynamic_path,
        static_trial_full_file_path=c3d_static_path
    )

    # --- Création du modèle biomécanique ---
    results.create_model(
        osim_model_type=OsimModels.WholeBody(),
        mvc_trials_path=f"{base_data_path}/mvc_trials/",
        functional_trials_path=f"{base_data_path}/functional_trials/",
        q_regularization_weight=1,
        skip_if_existing=True,
        animate_model_flag=True,
    )

    # --- Ignorer certains marqueurs et canaux ---
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

    # --- Ajouter la vitesse du tapis à experimental_data ---
    results.experimental_data.treadmill_speed = treadmill_speed

    # --- Détection des événements cycliques ---
    results.add_cyclic_events(force_plate_sides=[Side.LEFT, Side.RIGHT], skip_if_existing=True, plot_phases_flag=False)

    # --- Reconstruction de la cinématique ---
    results.reconstruct_kinematics(
        reconstruction_type=[
            ReconstructionType.LSQ,
            ReconstructionType.ONLY_LM,
            ReconstructionType.LM,
            ReconstructionType.TRF,
        ],
        animate_kinematics_flag=True,
        plot_kinematics_flag=True,
        skip_if_existing=True,
    )

    # --- Calculs biomécaniques ---
    results.compute_angular_momentum()
    results.compute_mechanical_energy(skip_if_existing=False)
    calculator = results.mechanical_energy_calculator
    Em = calculator.Em
    Em_norm = calculator.Em_norm
    E_pot = calculator.E_pot_vec
    E_kin = calculator.E_kin_vec
    E_kin_com = calculator.E_kin_global_vec

    # ----------------------------------------------------------------------
    # Calcul du MoS avec vitesse du tapis
    # ----------------------------------------------------------------------
    AP_MoS, ML_MoS = results.compute_mos(skip_if_existing=False)

    if AP_MoS.size == 0 or ML_MoS.size == 0:
        print("⚠️ MoS empty — check heel strikes or marker availability")
    else:
        print(
            f"MoS computed: "
            f"AP range [{AP_MoS.min():.3f}, {AP_MoS.max():.3f}], "
            f"ML range [{ML_MoS.min():.3f}, {ML_MoS.max():.3f}]"
        )
    # ----------------------------------------------------------------------
    # Dynamique inverse
    # ----------------------------------------------------------------------
    results.perform_inverse_dynamics(
        skip_if_existing=False,
        reintegrate_flag=True,
        animate_dynamics_flag=True,
    )

    return results


def main():
    # --- Liste des sujets ---
    subjects_to_analyze = [
        Subject(
            subject_name="LAO01",
            subject_height=1.65,
        )
    ]

    # --- Exécution de l'analyse ---
    AnalysisPerformer(
        analysis_to_perform,
        subjects_to_analyze=subjects_to_analyze,
        cycles_to_analyze=None,
        result_folder="results",
        trails_to_analyze=["Cond0003"],
        skip_if_existing=False,
    )


if __name__ == "__main__":
    main()


    # # --- Example of how to create a OrganizedResult object --- #
    # organized_result = OrganizedResult(
    #     result_folder="results",
    #     conditions_to_compare=["_Cond0004"],
    #     # plot_type=PlotType.ANGULAR_MOMENTUM,
    #     nb_frames_interp=101,
    # )
    # organized_result.save("results/AngMom_organized.pkl")
    #
    # # --- Example of how to plot the angular momentum --- #
    # plot = PlotBiomechanicsQuantity(
    #     organized_result=organized_result,
    # )
    # plot.draw_plot()
    # plot.save("results/AngMom_temporary.svg")
    # plot.show()
    #
    # # --- Example of how compare peak-to-peak angular momentum with a paired t-test --- #
    # stats_results = StatsPerformer(
    #     organized_result=organized_result,
    #     stats_type=StatsType.PAIRED_T_TEST(QuantityToExtractType.PEAK_TO_PEAK),
    # )
    # stats_results.perform_stats()
    # stats_results.plot_stats(
    #     save_plot_name="results/AngMom_paired_t_test.svg", order=["_Marche050", "_Marche075", "_Marche100"]
    # )
