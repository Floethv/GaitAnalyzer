import os
import shutil
from xml.etree import ElementTree as ET
import numpy as np
import biorbd

from gait_analyzer.subject import Subject


class OsimModels:
    # TODO: Charbie -> Do we have the right to add the OpenSim models to a public repository?
    # TODO: Charbie -> Otherwise, can Floethv give the link to the OpenSim model?
    @property
    def osim_model_name(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].osim_model_name."
        )

    @property
    def original_osim_model_full_path(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].original_osim_model_full_path."
        )

    @property
    def xml_setup_file(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].xml_setup_file."
        )

    @property
    def muscles_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].muscles_to_ignore."
        )

    @property
    def markers_to_ignore(self):
        raise RuntimeError(
            "This method is implemented in the child class. You should call OsimModels.[mode type name].markers_to_ignore."
        )

    # Child classes acting as an enum
    class WholeBody:
        """This is a hole body model that consists of 23 bodies, 42 degrees of freedom and 30 muscles.
        The whole-body geometric model and the lower limbs, pelvis and upper limbs anthropometry are based on the running model of Hammer et al. 2010 which consists of 12 segments and 29 degrees of freedom.
        Extra segments and degrees of freedom were later added based on Dumas et al. 2007.
        Each lower extremity had seven degrees-of-freedom; the hip was modeled as a ball-and-socket joint (3 DoFs), the knee was modeled as a revolute joint with 1 dof, the ankle was modeled as 2 revolute joints and feet toes with one revolute joint.
        The pelvis joint was model as a free flyer joint (6 DoFs) to allow the model to translate and rotate in the 3D space, the lumbar motion was modeled as a ball-and-socket joint (3 DoFs) (Anderson and Pandy, 1999) and the neck joint was modeled as a ball-and-socket joint (3 DoFs).
        Mass properties of the torso and head (including the neck) segments were estimated from Dumas et al., 2007. Each arm consisted of 8 degrees-of-freedom; the shoulder was modeled as a ball-and-socket joint (3 DoFs), the elbow and forearm rotation were each modeled with revolute joints (1 dof) (Holzbaur et al., 2005), the wrist flexion and deviation were each modeled with revolute joints and the hand fingers were modeled with 1 revolute joint.
        Mass properties for the arms were estimated from 1 and de Leva, 1996. The model also include 30 superficial muscles of the whole body.
        [Charbie -> Link ?]
        """

        @property
        def osim_model_name(self):
            return "wholebody"

        @property
        def original_osim_model_full_path(self):
            return "../models/OpenSim_models/wholebody.osim"

        @property
        def xml_setup_file(self):
            return "../models/OpenSim_models/wholebody.xml"

        @property
        def muscles_to_ignore(self):
            return [
                "ant_delt_r",
                "ant_delt_l",
                "medial_delt_l",
                "post_delt_r",
                "post_delt_l",
                "medial_delt_r",
                "ercspn_r",
                "ercspn_l",
                "rect_abd_r",
                "rect_abd_l",
                "r_stern_mast",
                "l_stern_mast",
                "r_trap_acr",
                "l_trap_acr",
                "TRIlong",
                "TRIlong_l",
                "TRIlat",
                "TRIlat_l",
                "BIClong",
                "BIClong_l",
                "BRD",
                "BRD_l",
                "FCR",
                "FCR_l",
                "ECRL",
                "ECRL_l",
                "PT",
                "PT_l",
                "LAT2",
                "LAT2_l",
                "PECM2",
                "PECM2_l",
            ] + [
                "glut_med1_r",
                "semiten_r",
                "bifemlh_r",
                "sar_r",
                "tfl_r",
                "vas_med_r",
                "vas_lat_r",
                "glut_med1_l",
                "semiten_l",
                "bifemlh_l",
                "sar_l",
                "tfl_l",
                "vas_med_l",
                "vas_lat_l",
            ]

        @property
        def markers_to_ignore(self):
            return []


class ModelCreator:
    def __init__(
        self,
        subject: Subject,
        static_trial: str,
        models_result_folder: str,
        osim_model_type,
        skip_if_existing: bool,
        skip_scaling: bool,
        animate_model_flag: bool,
    ):
        """
        Initialize the ModelCreator.
        .
        Parameters
        ----------
        subject: Subject
            The subject to create the model for.
        static_trial: str
            The path to the static trial c3d file to use to create the model.
        models_result_folder: str
            The folder where the models will be saved.
        osim_model_type: OsimModels
            The type of model to create.
        skip_if_existing: bool
            If the model already exists, skip the creation.
        skip_scaling: bool
            If the model is already scaled in OpenSim's GUI, skip the scaling and only create the biomod.
        animate_model_flag: bool
            If True, animate the model after creating it.
        """

        # Checks
        if not isinstance(subject, Subject):
            raise ValueError("subject must be a Subject.")
        if not isinstance(static_trial, str):
            raise ValueError("static_trial must be a string.")
        if not isinstance(models_result_folder, str):
            raise ValueError("models_result_folder must be a string.")
        if not isinstance(skip_if_existing, bool):
            raise ValueError("skip_if_existing must be a boolean.")
        if not isinstance(skip_scaling, bool):
            raise ValueError("skip_scaling must be a boolean.")
        if not isinstance(animate_model_flag, bool):
            raise ValueError("animate_model_flag must be a boolean.")

        # Initial attributes
        self.subject = subject
        self.osim_model_type = osim_model_type
        self.static_trial = static_trial
        self.models_result_folder = models_result_folder

        # Extended attributes
        self.trc_file_path = None
        self.new_xml_path = None
        # TODO: Charbie -> can we point to the Opensim folder where all opensim's vtp files are stored
        self.vtp_geometry_path = "../../Geometry"
        self.osim_model_full_path = (
            self.models_result_folder
            + "/"
            + osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + ".osim"
        )
        self.biorbd_model_full_path = (
            self.models_result_folder
            + "/"
            + osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + ".bioMod"
        )
        self.biorbd_model_virtual_markers_full_path = (
            self.models_result_folder
            + "/"
            + osim_model_type.osim_model_name
            + "_"
            + self.subject.subject_name
            + "_virtual_markers.bioMod"
        )
        self.new_model_created = False

        # Create the models
        if not (skip_if_existing and os.path.isfile(self.biorbd_model_full_path)):
            if not skip_scaling:
                print(f"The model {self.biorbd_model_full_path} is being created...")
                self.convert_c3d_to_trc()
                self.personalize_xml_file_hacky()
                self.scale_opensim_model()
            else:
                self.create_biorbd_model()
        else:
            print(f"The model {self.biorbd_model_full_path} already exists, so it is being used.")
        self.biorbd_model = biorbd.Model(self.biorbd_model_full_path)

        if not (skip_if_existing and os.path.isfile(self.biorbd_model_virtual_markers_full_path)):
            self.extended_model_for_EKF()

        if animate_model_flag:
            self.animate_model()

    def convert_c3d_to_trc(self):
        """
        This function reads the c3d static file and converts it into a trc file that will be used to scale the model in OpenSim.
        The trc file is saved at the same place as the original c3d file.
        """
        self.trc_file_path = self.static_trial.replace(".c3d", ".trc")

        # Read the c3d file
        import ezc3d

        c3d = ezc3d.c3d(self.static_trial)
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]

        # TODO: FloEthv -> When this study is completed, please remove this hacky fix
        if "STER" in labels:
            labels[labels.index("STER")] = "STR"

        frame_rate = c3d["header"]["points"]["frame_rate"]
        marker_data = c3d["data"]["points"][:3, :, :] / 1000  # Convert in meters

        with open(self.trc_file_path, "w") as f:
            trc_file_name = os.path.basename(self.trc_file_path)
            f.write(f"PathFileType\t4\t(X/Y/Z)\t{trc_file_name}\n")
            f.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            )
            f.write(
                "{:.2f}\t{:.2f}\t{}\t{}\tm\t{:.2f}\t{}\t{}\n".format(
                    frame_rate,
                    frame_rate,
                    c3d["header"]["points"]["last_frame"],
                    len(labels),
                    frame_rate,
                    c3d["header"]["points"]["first_frame"],
                    c3d["header"]["points"]["last_frame"],
                )
            )
            f.write("Frame#\tTime\t" + "\t".join(labels) + "\n")
            f.write("\t\t" + "\t".join([f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(len(labels))]) + "\n")
            for frame in range(marker_data.shape[2]):
                time = frame / frame_rate
                frame_data = [f"{frame + 1}\t{time:.5f}"]
                for marker_idx in range(len(labels)):
                    pos = marker_data[:, marker_idx, frame]
                    frame_data.extend([f"{pos[0]:.5f}", f"{pos[1]:.5f}", f"{pos[2]:.5f}"])
                f.write("\t".join(frame_data) + "\n")

    # def personalize_xml_file(self):
    #     """
    #     This function should not be used right now, but it is still the real way to personalize the xml file.
    #     We are waiting for OpenSim to fix their path bug.
    #     """
    #     self.new_xml_path = self.osim_model_type.xml_setup_file.replace(".xml", f"_{self.subject_name}.xml")
    #     # Modify the original xml with the participants information
    #     tree = ET.parse(self.osim_model_type.xml_setup_file)
    #     root = tree.getroot()
    #     for elem in root.iter():
    #         if elem.tag == "model_file":
    #             elem.text = os.path.abspath(self.osim_model_type.original_osim_model_full_path)
    #         elif elem.tag == "output_model_file":
    #             # Due to OpenSim, this path must be relative to xml
    #             rel_path = os.path.relpath(self.osim_model_full_path, os.path.dirname(self.new_xml_path))
    #             elem.text = rel_path
    #         elif elem.tag == "mass":
    #             elem.text = f"{self.subject.subject_mass}"
    #         elif elem.tag == "marker_file":
    #             # Due to OpenSim, this path must be relative to original_osim_model_full_path
    #             trc_file_relative_path = os.path.relpath(
    #                 self.trc_file_path, os.path.abspath(self.osim_model_type.original_osim_model_full_path)
    #             )[3:]
    #             elem.text = trc_file_relative_path
    #     tree.write(self.new_xml_path)

    def personalize_xml_file_hacky(self):
        """
        This function is the one used in the process for now, but should be removed whenever we have the chance.
        """
        shutil.copyfile("../models/OpenSim_models/wholebody.xml", "wholebody.xml")
        shutil.copyfile("../models/OpenSim_models/wholebody.osim", "wholebody.osim")
        shutil.copyfile(self.trc_file_path, os.path.basename(self.trc_file_path))

        # Modify the original xml with the participants information
        tree = ET.parse("wholebody.xml")
        root = tree.getroot()
        for elem in root.iter():
            if elem.tag == "model_file":
                elem.text = "wholebody.osim"
            elif elem.tag == "output_model_file":
                rel_path = f"wholebody_{self.subject.subject_name}.osim"
                elem.text = rel_path
            elif elem.tag == "mass":
                elem.text = f"{self.subject.subject_mass}"
            elif elem.tag == "marker_file":
                elem.text = os.path.basename(self.trc_file_path)
        tree.write(f"wholebody_{self.subject.subject_name}.xml")

    def scale_opensim_model(self):
        try:
            import opensim as osim
        except:
            raise RuntimeError("To scale the model, you must install OpenSim.")

        self.new_xml_path = self.osim_model_full_path.replace(".osim", f".xml")
        # tool = osim.ScaleTool(self.new_xml_path)
        tool = osim.ScaleTool(f"wholebody_{self.subject.subject_name}.xml")
        tool.run()

        # Copy the output to the right place
        shutil.copyfile(f"wholebody_{self.subject.subject_name}.osim", self.osim_model_full_path)
        shutil.copyfile(f"wholebody_{self.subject.subject_name}.xml", self.new_xml_path)

        # Delete the temporary files
        os.remove(f"wholebody_{self.subject.subject_name}.osim")
        os.remove(f"wholebody_{self.subject.subject_name}.xml")
        os.remove(os.path.basename(self.trc_file_path))
        os.remove("wholebody.xml")
        os.remove("wholebody.osim")

    def create_biorbd_model(self):
        try:
            import osim_to_biomod as otb
        except:
            raise RuntimeError("To converet the osim model into a biomod, you must install osim_to_biomod.")

        # Convert the osim model to a biorbd model
        converter = otb.Converter(
            self.biorbd_model_full_path,  # .bioMod file to export to
            self.osim_model_full_path,  # .osim file to convert from
            ignore_muscle_applied_tag=False,
            ignore_fixed_dof_tag=False,
            ignore_clamped_dof_tag=False,
            mesh_dir=self.vtp_geometry_path,
            muscle_type=otb.MuscleType.HILL,
            state_type=otb.MuscleStateType.DEGROOTE,
            print_warnings=True,
            print_general_informations=False,
            vtp_polygons_to_triangles=True,
            muscles_to_ignore=self.osim_model_type.muscles_to_ignore,
            markers_to_ignore=self.osim_model_type.markers_to_ignore,
        )
        converter.convert_file()
        self.sketchy_replace_biomod_lines()
        self.new_model_created = True

    def sketchy_replace_biomod_lines(self):
        """
        This method is a temporary fix to replace the lines in the bioMod file.
        It should be done with the read feature of biorbd.model_creator.
        """

        with open(self.biorbd_model_full_path, "r+") as file:
            file_lines = file.readlines()

        with open(self.biorbd_model_full_path, "w") as file:
            for i_line, line in enumerate(file_lines):
                if i_line + 1 == 27:  # Turn the model so it is not alignes with the gimbal lock
                    file.write(
                        line.replace("\t\tRT\t0 0 0\txyz\t0 0 0\n", "\t\tRT\t1.57079633 1.57079633 0\txyz\t0 0 0\n")
                    )
                elif i_line + 1 == 42:  # Translation X
                    file.write(line.replace("-10 10", "-3 3"))
                elif i_line + 1 == 43:  # Translation Y
                    file.write(line.replace("-6 6", "-3 3"))
                elif i_line + 1 == 44:  # Translation Z
                    file.write(line.replace("-5 5", "-3 3"))
                elif i_line + 1 == 59:  # Pelvis Rotation X
                    file.write(line.replace("-3.1415999999999999 3.1415999999999999", f"{-np.pi/4} {np.pi/4}"))
                elif i_line + 1 == 60:  # Pelvis Rotation Y
                    file.write(line.replace("-3.1415999999999999 3.1415999999999999", f"{-np.pi/4} {np.pi/4}"))
                elif i_line + 1 == 177:  # Hip Rotation X
                    file.write(
                        line.replace("-2.6179999999999999 2.0943950999999998", f"{-40*np.pi/180} {120*np.pi/180}")
                    )
                elif i_line + 1 == 178:  # Hip Rotation Y
                    file.write(
                        line.replace("-2.0943950999999998 2.0943950999999998", f"{-60*np.pi/180} {30*np.pi/180}")
                    )
                elif i_line + 1 == 179:  # Hip Rotation Z
                    file.write(
                        line.replace(
                            "-2.0943950999999998 2.0943950999999998", f"{-30 * np.pi / 180} {30 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 262:  # Knee Rotation X
                    file.write(line.replace("-3.1415999999999999 0.34910000000000002", f"{-150 * np.pi / 180} {0.0}"))
                elif i_line + 1 == 358:  # Ankle Flexion
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-50 * np.pi / 180} {30 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 451:  # Ankle Inversion
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-15 * np.pi / 180} {15 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 560:  # Toes Flexion
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-50 * np.pi / 180} {60 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 633:  # Hip Rotation X
                    file.write(
                        line.replace(
                            "-2.6179999999999999 2.0943950999999998", f"{-40 * np.pi / 180} {120 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 634:  # Hip Rotation Y
                    file.write(
                        line.replace(
                            "-2.0943950999999998 2.0943950999999998", f"{-60 * np.pi / 180} {30 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 635:  # Hip Rotation Z
                    file.write(
                        line.replace(
                            "-2.0943950999999998 2.0943950999999998", f"{-30 * np.pi / 180} {30 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 718:  # Knee Rotation X
                    file.write(line.replace("-3.1415999999999999 0.34910000000000002", f"{-150 * np.pi / 180} {0.0}"))
                elif i_line + 1 == 814:  # Ankle Flexion
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-50 * np.pi / 180} {30 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 907:  # Ankle Inversion
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-15 * np.pi / 180} {15 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 1016:  # Toes Flexion
                    file.write(
                        line.replace(
                            "-1.0471975499999999 1.0471975499999999", f"{-50 * np.pi / 180} {60 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 1089:  # Torso Rotation X
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-90 * np.pi / 180} {45 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 1090:  # Torso Rotation Y
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-35 * np.pi / 180} {35 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 1091:  # Torso Rotation Z
                    file.write(
                        line.replace(
                            "-1.5707963300000001 1.5707963300000001", f"{-45 * np.pi / 180} {45 * np.pi / 180}"
                        )
                    )
                elif i_line + 1 == 1198:  # Head and neck Rotation X
                    file.write(line.replace("-1.74533 1.0471975499999999", f"{-50 * np.pi / 180} {45 * np.pi / 180}"))
                elif i_line + 1 in [
                    1306,
                    1307,
                    1308,
                    1309,
                    1478,
                    1479,
                    2137,
                    2138,
                    2139,
                    2140,
                    2309,
                    2310,
                ]:  # Uncomment ranges
                    file.write(line.replace("// ", ""))
                else:
                    file.write(line)

    @property
    def markers_for_virtual(self):
        return {
            "pelvis-V1": ["RASIS", "LASIS", "LPSIS", "RPSIS"],  # Close to pelvis center
            "femur_r-V1": ["RLFE", "RMFE"],  # Close to knee center
            "femur_r-V2": ["RLFE", "RMFE", "RGT"],  # Close to femur center
            "tibia_r-V1": ["RSPH", "RLM"],  # Close to ankle center
            "tibia_r-V2": ["RSPH", "RLM", "RATT"],  # Close to tibia center
            "calcn_r-V1": ["RMFH1", "RMFH1"],  # Close to forefoot center
            "calcn_r-V2": ["RMFH1", "RMFH1", "RCAL"],  # Close to foot center
            "femur_l-V1": ["LLFE", "LMFE"],  # Close to knee center
            "femur_l-V2": ["LLFE", "LMFE", "LGT"],  # Close to femur center
            "tibia_l-V1": ["LSPH", "LLM"],  # Close to ankle center
            "tibia_l-V2": ["LSPH", "LLM", "LATT"],  # Close to tibia center
            "calcn_l-V1": ["LMFH1", "LMFH1"],  # Close to forefoot center
            "calcn_l-V2": ["LMFH1", "LMFH1", "LCAL"],  # Close to foot center
            "torso-V1": ["STR", "C7", "T10", "SUP"],  # Close to torso center
            "humerus_r-V1": ["RLHE", "RMHE"],  # Close to elbow center
            "radius_r-V1": ["RUS", "RRS"],  # Close to wrist center
            "hand_r-V1": ["RHMH5", "RHMH2"],  # Close to forehand center
            "humerus_l-V1": ["LLHE", "LMHE"],  # Close to elbow center
            "radius_l-V1": ["LUS", "LRS"],  # Close to wrist center
            "hand_l-V1": ["LHMH5", "LHMH2"],  # Close to forehand center
        }

    def extended_model_for_EKF(self):
        """
        This function adds virtual markers to the original biomod to improve the kinematic reconstruction.
        First, joint markers are added between the bone landmark markers.
        Second, mid-segment markers are added between the joint markers.
        """
        # Copy the biomod file
        shutil.copy2(self.biorbd_model_full_path, self.biorbd_model_virtual_markers_full_path)

        all_marker_names = [m.to_string() for m in self.biorbd_model.markerNames()]
        for marker in all_marker_names:
            if "-" in marker:
                raise ValueError(
                    f"Marker {marker} contains a dash. Please avoid this character as the virtual marker system uses this separator to identify parents."
                )

        # When no Q are provided to biorbd.markers, the makers are expressed in the local reference frame
        all_marker_positions = [m.to_array() for m in self.biorbd_model.markers()]

        # Extend the new biomod file with the virtual markers
        with open(self.biorbd_model_virtual_markers_full_path, "a+") as file:
            file.write("\n\n/*-------------- VIRTUAL MARKERS --------------- \n*/\n")
            for i_jcs, key in enumerate(self.markers_for_virtual):
                # Fid the virtual marker position
                marker_positions = np.zeros((3, len(self.markers_for_virtual[key])))
                for i_marker, marker_name in enumerate(self.markers_for_virtual[key]):
                    if marker_name not in all_marker_names:
                        raise ValueError(f"Marker {marker_name} not found in the model.")
                    marker_positions[:, i_marker] = all_marker_positions[all_marker_names.index(marker_name)]
                virtual_marker_position_in_local = np.mean(marker_positions, axis=1)
                # Write the virtual marker to the biomod file
                file.write(f"marker\t{key}JC\n")
                parent_name = key.split("-")[0]
                file.write(f"\tparent\t{parent_name}\n")
                file.write(
                    f"\tposition\t{virtual_marker_position_in_local[0]}\t{virtual_marker_position_in_local[1]}\t{virtual_marker_position_in_local[2]}\n"
                )
                file.write("endmarker\n")

    def animate_model(self):
        """
        Animate the model
        """
        try:
            from pyorerun import BiorbdModel, PhaseRerun
        except:
            raise RuntimeError("To animate the model, you must install Pyorerun.")

        # Model
        model = BiorbdModel(self.biorbd_model_virtual_markers_full_path)
        model.options.transparent_mesh = False
        model.options.show_gravity = True

        # Visualization
        viz = PhaseRerun(np.linspace(0, 1, 10))
        viz.add_animated_model(model, np.zeros((model.nb_q, 10)))
        viz.rerun_by_frame("Kinematics reconstruction")

    def inputs(self):
        return {
            "subject_name": self.subject.subject_name,
            "subject_mass": self.subject.subject_mass,
            "osim_model_type": self.osim_model_type,
            "static_trial": self.static_trial,
        }

    def outputs(self):
        return {
            "biorbd_model_full_path": self.biorbd_model_full_path,
            "open_sim_model_full_path": self.osim_model_full_path,
            "biorbd_model": self.biorbd_model,
            "new_model_created": self.new_model_created,
            "biorbd_model_virtual_markers_full_path": self.biorbd_model_virtual_markers_full_path,
            "markers_for_virtual": self.markers_for_virtual,
        }
