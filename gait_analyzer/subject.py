from enum import Enum


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class Subject:
    """
    This class contains all the subject information.
    """

    def __init__(self, subject_name: str, subject_mass: float, dominant_leg: Side, preferential_speed: float):
        """
        Initialize the SubjectList.
        """
        # Checks
        if not isinstance(subject_name, str):
            raise ValueError("subject_name must be a string")
        if not isinstance(subject_mass, float):
            raise ValueError("subject_mass must be an float")
        # Check to make sure no child is analyzed (since scaling would not be adapted) and mass is not entered in pounds.
        if subject_mass < 30 or subject_mass > 100:
            raise ValueError(f"Mass of subject {subject_name} must be a expressed in [30, 100] kg.")
        if not isinstance(dominant_leg, Side):
            raise ValueError("dominant_leg must be a Side")
        if not isinstance(preferential_speed, float):
            raise ValueError("preferential_speed must be a float")

        # Initial attributes
        self.subject_name = subject_name
        self.subject_mass = subject_mass
        self.dominant_leg = dominant_leg
        self.preferential_speed = preferential_speed


    def outputs(self):
        return {
            "subject_name": self.subject_name,
            "subject_mass": self.subject_mass,
            "dominant_leg": self.dominant_leg,
            "preferential_speed": self.preferential_speed,
        }
