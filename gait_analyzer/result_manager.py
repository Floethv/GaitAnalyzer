from gait_analyzer.AngularMomentumCalculator import AngularMomentumCalculator
from gait_analyzer.model_creator import ModelCreator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.cyclic_events import CyclicEvents
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.optimal_estimator import OptimalEstimator
from gait_analyzer.subject import Subject, Side
from gait_analyzer.unique_events import UniqueEvents


class ResultManager:
    """
    This class contains all the results from the gait analysis and is the main class handling all types of analysis to perform on the experimental data.
    """

    def __init__(self, subject: Subject, cycles_to_analyze: range, static_trial: str, result_folder: str):
        """
        Initialize the ResultManager.
        .
        Parameters
        ----------
        subject: Subject
            The subject to analyze
        cycles_to_analyze: range
            The range of cycles to analyze
        static_trial: str
            The full file path of the static trial ([...]_static.c3d)
        result_folder: str
            The folder where the results will be saved. It will look like result_folder/subject_name.
        """
        # Checks:
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject")
        if not (isinstance(cycles_to_analyze, range) or cycles_to_analyze is None):
            raise ValueError(
                "cycles_to_analyze must be a range of cycles to analyze or None if all frames should be analyzed."
            )
        if not isinstance(static_trial, str):
            raise ValueError("static_trial must be a string")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")

        # Initial attributes
        self.subject = subject
        self.cycles_to_analyze = cycles_to_analyze
        self.result_folder = result_folder
        self.static_trial = static_trial

        # Extended attributes
        self.experimental_data = None
        self.model_creator = None
        self.gait_parameters = None
        self.events = None
        self.kinematics_reconstructor = None
        self.inverse_dynamics_performer = None
        self.optimal_estimator = None
        self.angular_momentum_calculator = None

    def create_model(self, osim_model_type, skip_if_existing: bool, skip_scaling:bool = False, animate_model_flag: bool = False):
        """
        Create and add the biorbd model to the ResultManager
        """

        # Checks
        if self.model_creator is not None:
            raise Exception("Biorbd model already added")

        # Add ModelCreator
        self.model_creator = ModelCreator(
            subject=self.subject,
            static_trial=self.static_trial,
            models_result_folder=f"{self.result_folder}/models",
            osim_model_type=osim_model_type,
            skip_if_existing=skip_if_existing,
            skip_scaling=skip_scaling,
            animate_model_flag=animate_model_flag,
        )

    def add_experimental_data(
        self, c3d_file_name: str, markers_to_ignore: list[str] = [], animate_c3d_flag: bool = False
    ):

        # Checks
        if self.experimental_data is not None:
            raise Exception("Experimental data already added")
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")

        # Add experimental data
        self.experimental_data = ExperimentalData(
            c3d_file_name=c3d_file_name,
            markers_to_ignore=markers_to_ignore,
            result_folder=self.result_folder,
            model_creator=self.model_creator,
            animate_c3d_flag=animate_c3d_flag,
        )

    def add_cyclic_events(self, force_plate_sides: list[Side], skip_if_existing: bool, plot_phases_flag: bool = False):

        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is not None:
            raise Exception("CyclicEvents or UniqueEvents were already added to the ResultManager")

        # Add events
        self.events = CyclicEvents(
            experimental_data=self.experimental_data,
            force_plate_sides=force_plate_sides,
            skip_if_existing=skip_if_existing,
            plot_phases_flag=plot_phases_flag,
        )

    def add_unique_events(self, skip_if_existing: bool, plot_phases_flag: bool = False):

        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is not None:
            raise Exception("CyclicEvents or UniqueEvents were already added to the ResultManager")

        # Add events
        self.events = UniqueEvents(
            experimental_data=self.experimental_data,
            skip_if_existing=skip_if_existing,
        )

    def reconstruct_kinematics(
        self,
        reconstruction_type=None,
        skip_if_existing: bool = False,
        animate_kinematics_flag: bool = False,
        plot_kinematics_flag: bool = False,
    ):
        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is None:
            raise Exception(
                "Please run the events detection first by running ResultManager.add_cyclic_events() or ResultManager.add_unique_events()"
            )
        if self.kinematics_reconstructor is not None:
            raise Exception("kinematics_reconstructor already added")

        # Reconstruct kinematics
        self.kinematics_reconstructor = KinematicsReconstructor(
            self.experimental_data,
            self.model_creator,
            self.events,
            self.cycles_to_analyze,
            reconstruction_type=reconstruction_type,
            skip_if_existing=skip_if_existing,
            animate_kinematics_flag=animate_kinematics_flag,
            plot_kinematics_flag=plot_kinematics_flag,
        )

    def compute_angular_momentum(self):
        """
        Fonction principale qui gère le calcul du moment angulaire.
        """
        # Vérifications
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is None:
            raise Exception("Please add the kinematics reconstructor first by running ResultManager.reconstruct_kinematics()")
        if self.angular_momentum_calculator is not None:
            raise Exception("Angular momentum was already calculate")
        self.angular_momentum_calculator = AngularMomentumCalculator(
            self.model_creator.biorbd_model, self.kinematics_reconstructor.q_filtered,
            self.kinematics_reconstructor.qdot, self.subject.subject_mass, self.subject.subject_height)

    def perform_inverse_dynamics(
        self, skip_if_existing: bool, reintegrate_flag: bool = True, animate_dynamics_flag: bool = False
    ):
        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please add the kinematics reconstructor first by running ResultManager.reconstruct_kinematics()"
            )
        if self.inverse_dynamics_performer is not None:
            raise Exception("inverse_dynamics_performer already added")

        # Perform inverse dynamics
        self.inverse_dynamics_performer = InverseDynamicsPerformer(
            self.experimental_data,
            self.kinematics_reconstructor,
            skip_if_existing=skip_if_existing,
            reintegrate_flag=reintegrate_flag,
            animate_dynamics_flag=animate_dynamics_flag,
        )

    def estimate_optimally(
        self, cycle_to_analyze: int, plot_solution_flag: bool = False, animate_solution_flag: bool = False
    ):

        # Checks
        if self.model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is None:
            raise Exception(
                "Please run the events detection first by running ResultManager.add_cyclic_events() or ResultManager.add_unique_events()"
            )
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please run the kinematics reconstruction first by running ResultManager.estimate_optimally()"
            )
        if self.inverse_dynamics_performer is None:
            raise Exception("Please run the inverse dynamics first by running ResultManager.perform_inverse_dynamics()")

        # Perform the optimal estimation optimization
        self.optimal_estimator = OptimalEstimator(
            cycle_to_analyze=cycle_to_analyze,
            subject=self.subject,
            biorbd_model_path=self.model_creator.biorbd_model_virtual_markers_full_path,
            experimental_data=self.experimental_data,
            events=self.events,
            kinematics_reconstructor=self.kinematics_reconstructor,
            inverse_dynamic_performer=self.inverse_dynamics_performer,
            plot_solution_flag=plot_solution_flag,
            animate_solution_flag=animate_solution_flag,
        )
