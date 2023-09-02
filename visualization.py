import os
import cv2
from datetime import datetime
from math import sin, cos
import yaml
import numpy as np
import open3d as o3d

WIDTH = 1600
HEIGHT = 900


def read_yaml(path):
    content = []
    assert os.path.isfile(path), f"{path} doesn't exist"
    with open(path, "r") as stream:
        content = yaml.safe_load(stream)

    return content


class Visualizer:
    """
    Semantic KITTI, NuScene, Waymo visualizer class

    """
    def __init__(self, kitti_color_map_path: str=os.path.join("visualization_configs", "color_maps", "kitti_color_map.yaml"), nuscene_color_map_path: str=os.path.join("visualization_configs", "color_maps", "nuscene_color_map.yaml"), 
                 waymo_color_map_path: str=os.path.join("visualization_configs", "color_maps", "waymo_color_map.yaml"), camera_trajectory_path: str=os.path.join("visualization_configs", "camera_trajectory.json"),
                 create_images: bool=False, create_video: bool=True) -> None:
        """
        :param kitti_color_map_path:  semanticKITTI color map path
        :param nuscene_color_map_path:  NuScene color map path
        :param waymo_color_map_path:  waymo color map path
        :param camera_trajectory_path:  TO put camera in BEV

        """
        assert type(kitti_color_map_path) == str, f"{kitti_color_map_path} is not string."
        assert type(nuscene_color_map_path) == str, f"{nuscene_color_map_path} is not string."
        assert type(waymo_color_map_path) == str, f"{waymo_color_map_path} is not string."

        # load color maps
        self.color_maps = {
            "pcd": {
                "kitti": read_yaml(kitti_color_map_path)["pcd_color_map"],
                "nuscene": read_yaml(nuscene_color_map_path)["pcd_color_map"],
                "waymo": read_yaml(waymo_color_map_path)["pcd_color_map"]
            },
            "bbox": {
                "kitti": read_yaml(kitti_color_map_path)["bbox_color_map"],
                "nuscene": read_yaml(nuscene_color_map_path)["bbox_color_map"],
                "waymo": read_yaml(waymo_color_map_path)["bbox_color_map"]
            }
        }
        
        self.bbox_label_maps = {
            "kitti": read_yaml(kitti_color_map_path)["bbox_label_inverse"],
            "nuscene": read_yaml(nuscene_color_map_path)["bbox_label_inverse"],
            "waymo": read_yaml(waymo_color_map_path)["bbox_label_inverse"]
        }
        
        self.camera_trajectory_path = camera_trajectory_path

        self.create_images = create_images
        self.create_video = create_video

        # Normalize color maps
        self.normalize_color_maps()
    
    @staticmethod
    def bgr_to_rgb(bgr):
        return bgr[::-1]
    
    def normalize_color_maps(self):
        """
        Normalize color maps from range 0 ~ 255 to range 0 ~ 1
        """
        for dataset_name in ["kitti", "nuscene", "waymo"]:  
            for label in self.color_maps["pcd"][dataset_name].keys():
                self.color_maps["pcd"][dataset_name][label] = [x / 256 for x in self.color_maps["pcd"][dataset_name][label]]
                self.color_maps["pcd"][dataset_name][label] = np.asarray(self.bgr_to_rgb(self.color_maps["pcd"][dataset_name][label])).T

            for label in self.color_maps["bbox"][dataset_name].keys():
                self.color_maps["bbox"][dataset_name][label] = [x / 256 for x in self.color_maps["bbox"][dataset_name][label]]
                self.color_maps["bbox"][dataset_name][label] = np.asarray(self.bgr_to_rgb(self.color_maps["bbox"][dataset_name][label])).T
                
    
    def map_colors(self, np_label: np.array, geometry_type, dataset_name: str) -> np.array:
        """
        Map colors to pcd using labels

        :param np_label:
        :param dataset_name:
        :return np_color:
        """
        assert dataset_name in ["kitti", "nuscene", "waymo"], "Invalid dataset name"
        assert geometry_type in ["pcd", "bbox"], "Invalid geometry type"
        color_map = self.color_maps[geometry_type][dataset_name]
        np_color = np.vectorize(color_map.__getitem__, otypes=[np.ndarray])(np_label+1)  # TODO: bbox에 맞는지 확인
        np_color = np_color.tolist()
        np_color = np.array(np_color).reshape(-1, 3)

        return np_color

    def map_bbox_label(self, list_of_bbox_label: list[str], dataset_name: str) -> np.array:
        """
        Map bbox string labels to integer labels
        
        :param list_of_bbox_label:
        :param dataset_name:
        :return np_bbox_label: 
        """

        assert dataset_name in ["kitti", "nuscene", "waymo"], "Invalid dataset name"
        bbox_label_map = self.bbox_label_maps[dataset_name]
        np_bbox_label = np.array([int(bbox_label_map[string_label]) for string_label in list_of_bbox_label])
        
        return np_bbox_label
        
    def preprocess_pcd(self, list_of_np_pcd: list[np.array], list_of_np_pcd_label: list[np.array], dataset_name: str) -> list[np.array]:
        """
        Receive list of numpy.array pcds and pcd labels and convert it to list of open3d.geometry.PointCloud()
        
        :param list_of_np_pcd:
        :param list_of_np_pcd_label:
        :param dataset_name:
        :return list_of_pcd: 
        """
        list_of_pcd = []

        for np_pcd, np_pcd_label in zip(list_of_np_pcd, list_of_np_pcd_label):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcd)
            np_color = self.map_colors(np_pcd_label, "pcd", dataset_name)
            pcd.colors = o3d.utility.Vector3dVector(np_color)  # map colors to corresponding points

            list_of_pcd.append(pcd)

        return list_of_pcd

    @staticmethod
    def calculate_bounds(np_bbox: np.array) -> list[np.array]:
        """
        Caculate min bound and max bounds from bbox numpy format

        :param np_bbox:
        :return [min_bound, max_bound]: 

        """
        center_x, center_y, center_z = np_bbox[0], np_bbox[1], np_bbox[2]
        length, width, height = np_bbox[3], np_bbox[4], np_bbox[5]
        heading = np_bbox[6]

        min_bound = np.array(
            [center_x - ((width / 2) * cos(heading) - (length / 2) * sin(heading)),
             center_y - ((width / 2) * sin(heading) + (length / 2) * cos(heading)),
             center_z - height / 2 
            ]
        )

        max_bound = np.array(
            [center_x + (width / 2) * cos(heading) - (length / 2) * sin(heading), 
             center_y + (width / 2) * sin(heading) + (length / 2) * cos(heading), 
             center_z + height / 2]
             )

        return [min_bound, max_bound]

    def preprocess_bbox(self, list_of_np_bboxes: list[np.array], list_of_np_bboxes_label: list[np.array], dataset_name: str) -> list[np.array]:
        """
        Receive list of numpy.array bboxes and bbox labels and convert it to list of open3d.geometry.AxisAlignedBoundingBox
        
        :param list_of_np_bbox:
        :param list_of_np_bbox_label:
        :param dataset_name:
        :return list_of_bbox: 
        """
        
        list_of_bboxes = []

        for np_bboxes, np_bbox_labels in zip(list_of_np_bboxes, list_of_np_bboxes_label):
            list_of_bbox = []
            for np_bbox, np_bbox_label in zip(np_bboxes, np_bbox_labels):
                min_bound, max_bound = self.calculate_bounds(np_bbox)
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
                np_color = self.map_colors(np_bbox_label, "bbox", dataset_name)
                bbox.color = o3d.utility.Vector3dVector(np_color)
                list_of_bbox.append(bbox)

            list_of_bboxes.append(list_of_bbox)
        
        return list_of_bboxes

    def visualize(self, np_pcd, np_pcd_label, np_bbox=None, np_bbox_label=None, dataset_name: str="", height: float=50.0, video_name: str="") -> None:
        """
        Receive list of, or one numpy.array pcds and label and Visualize sequently.

        NOTE: can put list of np.arrays or one np.array at np_pcd and np_label
        :param np_pcd:
        :param np_label:
        :param np_bbox:
        :param np_bbox_label:
        :param dataset_name:
        :param height:
        :param video_name:
        :param dataset_name:
        """

        assert type(np_pcd) == np.array or type(np_pcd) == list, f"Invalid np_pcd(type {type(np_pcd)}) received."
        assert type(np_pcd_label) == np.array or type(np_pcd_label) == list, f"Invalid np_pcd_label(type {type(np_pcd_label)}) received."
        if np_bbox is not None:
            assert type(np_bbox) == np.array or type(np_bbox) == list, f"Invalid np_bbox(type {type(np_bbox)}) received."
            assert type(np_bbox_label) == np.array or type(np_bbox_label) == list, f"Invalid np_bbox_label(type {type(np_bbox_label)}) received."

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=WIDTH, height=HEIGHT)
        ctr = vis.get_view_control()
        trajectory = o3d.io.read_pinhole_camera_trajectory(self.camera_trajectory_path)
        
        # Modify extrisic to set height
        parameter = trajectory.parameters[0]
        _extrinsic = np.copy(parameter.extrinsic)
        _extrinsic[2, 3] = height
        parameter.extrinsic = _extrinsic
        trajectory.parameters[0] = parameter

        list_of_pcd = []
        list_of_bboxes = []
        list_of_image_path = []
        
        if type(np_pcd) == np.array:
            list_of_pcd = self.preprocess_pcd([np_pcd], [np_pcd_label], dataset_name)

        elif type(np_pcd) == list:
            list_of_pcd = self.preprocess_pcd(np_pcd, np_pcd_label, dataset_name)
        
        if np_bbox is not None:
            if type(np_bbox) == np.array:
                list_of_bboxes = self.preprocess_bbox([np_bbox], [np_bbox_label], dataset_name)

            elif type(np_bbox) == list:
                list_of_bboxes = self.preprocess_bbox(np_bbox, np_bbox_label, dataset_name)


        for i in range(len(list_of_pcd)):

            vis.clear_geometries()  # Clear pcd and reset camera pose
            vis.add_geometry(list_of_pcd[i])  # Add pcd seqencnce
            if np_bbox is not None:
                for bbox in list_of_bboxes[i]:
                    vis.add_geometry(bbox)

            ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0], allow_arbitrary=True)  # Set camera pose
            vis.poll_events()
            vis.update_renderer()
            
            # Create video
            if self.create_video:
                if not os.path.isdir("images"):
                    os.mkdir("images")

                image_name = os.path.join("images", video_name + "_%04d.jpg" % i)
                list_of_image_path.append(image_name)
                vis.capture_screen_image(image_name)
        
        # Create video
        if self.create_video:
            if video_name == "":
                now = datetime.now()
                video_name = now.strftime("Visualization_%m-%d-%Y_%H_%M_%S.mp4")
            
            self.make_video(list_of_image_path, video_name)
        
        if not self.create_images:
            # Delete all images
            for image_path in list_of_image_path:
                if os.path.isfile(image_path):
                    os.remove(image_path)


    def make_video(self, list_of_image_path: list, video_name: str) -> None:
        """
        Create video from given scenes

        :param list_of_image_path:
        :param video_name:        
        """

        assert type(video_name) == str, f"{video_name} is not string."
        # Video fream size
        frame_size = (WIDTH, HEIGHT)
        out = cv2.VideoWriter(os.path.join("videos", video_name), cv2.VideoWriter_fourcc(*'mp4v'), 15, frame_size)
        
        for image_path in list_of_image_path:
            image = cv2.imread(image_path)
            out.write(image)
        
        out.release()
        print(f"Created video name: {video_name}")

if __name__ == "__main__":
    visualizer = Visualizer()

    from nuscene_reader import NuSceneReader
    NUSCENE_PATH = os.getcwd()
    nuscene_reader = NuSceneReader(NUSCENE_PATH)
    list_of_np_pcd, list_of_label = nuscene_reader.load_data_for_scene(1)
    record_video = True
    height = 25
    video_name = "test_video.mp4"
    visualizer.visualize(
        np_pcd=list_of_np_pcd,  # n x 3 np.array
        np_pcd_label=list_of_label, # n x 1 np.array
        np_bbox=None,  # n x 7 np.array
        np_bbox_label = None,  # n x 1 np.array, use visualizer.map_bbox_label(list_of_string_bbox_label, dataset_name) to convert string label to integer labels
        dataset_name="nuscene", # [kitti|nuscene|waymo]
        height=height,  # z-coord of BEV camera
        video_name=video_name  # name of video to be saved in videos/
        )
