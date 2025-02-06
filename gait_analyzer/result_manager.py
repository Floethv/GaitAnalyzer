from gait_analyzer.biomod_model_creator import BiomodModelCreator
from gait_analyzer.experimental_data import ExperimentalData
from gait_analyzer.inverse_dynamics_performer import InverseDynamicsPerformer
from gait_analyzer.events import Events
from gait_analyzer.kinematics_reconstructor import KinematicsReconstructor
from gait_analyzer.optimal_estimator import OptimalEstimator


class ResultManager:
    """
    This class contains all the results from the gait analysis and is the main class handling all types of analysis to perform on the experimental data.
    """

    def __init__(self, subject_name: str, subject_mass: float, result_folder: str):
        """
        Initialize the ResultManager.
        .
        Parameters
        ----------
        subject_name: str
            The name of the subject
        subject_mass: float
            The mass of the subject
        result_folder: str
            The folder where the results will be saved. It will look like result_folder/subject_name.
        """
        # Checks:
        if not isinstance(subject_name, str):
            raise ValueError("subject_name must be a string")
        if not isinstance(subject_mass, float):
            raise ValueError("subject_mass must be an float")
        if not isinstance(result_folder, str):
            raise ValueError("result_folder must be a string")

        # Attributes
        self.subject_name = subject_name
        self.subject_mass = subject_mass
        self.result_folder = result_folder
        self.experimental_data = None
        self.biorbd_model_creator = None
        self.gait_parameters = None
        self.events = None
        self.kinematics_reconstructor = None
        self.inverse_dynamics_performer = None
        self.optimal_estimator = None


    def create_biorbd_model(self, osim_model_type, skip_if_existing: bool):
        """
        Create and add the biorbd model to the ResultManager
        """

        # Checks
        if self.biorbd_model_creator is not None:
            raise Exception("Biorbd model already added")

        # Add BiomodModelCreator
        self.biorbd_model_creator = BiomodModelCreator(self.subject_name, osim_model_type, skip_if_existing)

    def add_experimental_data(self, c3d_file_name: str, animate_c3d_flag: bool = False):

        # Checks
        if self.experimental_data is not None:
            raise Exception("Experimental data already added")
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")

        # Add experimental data
        self.experimental_data = ExperimentalData(
            c3d_file_name=c3d_file_name,
            subject_name=self.subject_name,
            result_folder=self.result_folder,
            biorbd_model_creator=self.biorbd_model_creator,
            animate_c3d_flag=animate_c3d_flag,
        )

    def add_events(self, plot_phases_flag):

        # Checks
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is not None:
            raise Exception("Events already added")

        # Add events
        self.events = Events(experimental_data=self.experimental_data, plot_phases_flag=plot_phases_flag)

    def reconstruct_kinematics(self,
                               reconstruction_type = None,
                               skip_if_existing: bool = False,
                               animate_kinematics_flag: bool = False,
                               plot_kinematics_flag: bool = False):
        # Checks
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is None:
            raise Exception("Please run the events detection first by running ResultManager.add_events()")
        if self.kinematics_reconstructor is not None:
            raise Exception("kinematics_reconstructor already added")

        # Reconstruct kinematics
        self.kinematics_reconstructor = KinematicsReconstructor(
            self.experimental_data,
            self.biorbd_model_creator,
            self.events,
            reconstruction_type=reconstruction_type,
            skip_if_existing=skip_if_existing,
            animate_kinematics_flag=animate_kinematics_flag,
            plot_kinematics_flag=plot_kinematics_flag,
        )


    def perform_inverse_dynamics(self):
        # Checks
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.kinematics_reconstructor is None:
            raise Exception("Please add the kinematics reconstructor first by running ResultManager.reconstruct_kinematics()")
        if self.inverse_dynamics_performer is not None:
            raise Exception("inverse_dynamics_performer already added")

        # Perform inverse dynamics
        self.inverse_dynamics_performer = InverseDynamicsPerformer(
            self.experimental_data,
            self.biorbd_model_creator.biorbd_model,
            self.kinematics_reconstructor.q_filtered,
            self.kinematics_reconstructor.qdot,
            self.kinematics_reconstructor.qddot,
        )


    def estimate_optimally(self):
        # Checks
        if self.biorbd_model_creator is None:
            raise Exception("Please add the biorbd model first by running ResultManager.create_biorbd_model()")
        if self.experimental_data is None:
            raise Exception("Please add the experimental data first by running ResultManager.add_experimental_data()")
        if self.events is None:
            raise Exception("Please run the events detection first by running ResultManager.add_events()")
        if self.kinematics_reconstructor is None:
            raise Exception(
                "Please run the kinematics reconstruction first by running ResultManager.estimate_optimally()"
            )

        # Perform the optimal estimation optimization
        self.optimal_estimator = OptimalEstimator(
            biorbd_model_path=self.biorbd_model_creator.biorbd_model_full_path,
            experimental_data=self.experimental_data,
            q=self.kinematics_reconstructor.q,
            qdot=self.kinematics_reconstructor.qdot,
            phases=self.events.phases,
        )
