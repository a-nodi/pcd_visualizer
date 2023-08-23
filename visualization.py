import os
import cv2
from datetime import datetime
import yaml
import numpy as np
import open3d as o3d



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
            "kitti": read_yaml(kitti_color_map_path)["color_map"],
            "nuscene": read_yaml(nuscene_color_map_path)["color_map"],
            "waymo": read_yaml(waymo_color_map_path)["color_map"]
        }
        self.camera_trajectory_path = camera_trajectory_path

        self.create_images = create_images
        self.create_video = create_video

        # Normalize color maps
        self.normalize_color_maps()

    def normalize_color_maps(self):
        """
        Normalize color maps from range 0 ~ 255 to range 0 ~ 1
        """
        for dataset_name in ["kitti", "nuscene", "waymo"]:  
            for label in self.color_maps[dataset_name].keys():
                self.color_maps[dataset_name][label] = [x / 256 for x in self.color_maps[dataset_name][label]]
                self.color_maps[dataset_name][label] = np.asarray(self.color_maps[dataset_name][label]).T

    def map_colors(self, np_label: np.array, dataset_name: str) -> np.array:
        """
        Map colors to pcd using labels

        :param np_label:
        :param dataset_name:
        :return np_color:
        """
        assert dataset_name in ["kitti", "nuscene", "waymo"], "Invalid dataset name"
        color_map = self.color_maps[dataset_name]
        np_color = np.vectorize(color_map.__getitem__, otypes=[np.ndarray])(np_label+1)
        np_color = np_color.tolist()
        np_color = np.array(np_color).reshape(-1, 3)

        return np_color

    def preprocess_scene(self, list_of_np_pcd: list[np.array], list_of_np_label: list[np.array], dataset_name: str) -> list[np.array]:
        """
        Receive list of numpy.array pcds and label and convert it to list of open3d.geometry.PointCloud()
        
        :param list_of_np_pcd:
        :param list_of_np_label:
        :param dataset_name:
        :return list_of_pcd: 
        """
        list_of_pcd = []

        for np_pcd, np_label in zip(list_of_np_pcd, list_of_np_label):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcd)
            np_color = self.map_colors(np_label, dataset_name)
            pcd.colors = o3d.utility.Vector3dVector(np_color)  # map colors to corresponding points

            list_of_pcd.append(pcd)

        return list_of_pcd

    def visualize(self, np_pcd, np_label, dataset_name: str, video_name: str="") -> None:
        """
        Receive list of, or one numpy.array pcds and label and Visualize sequently.

        NOTE: can put list of np.arrays or one np.array at np_pcd and np_label
        :param list_of_np_pcd:
        :param list_of_np_label:
        :param dataset_name:
        """

        assert type(np_pcd) == np.array or type(np_pcd) == list, f"Invalid np_pcd(type{type(np_pcd)}) received."
        assert type(np_label) == np.array or type(np_label) == list, f"Invalid np_label(type{type(np_label)}) received."
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        ctr = vis.get_view_control()
        trajectory = o3d.io.read_pinhole_camera_trajectory(self.camera_trajectory_path)
        
        list_of_pcd = []
        list_of_image_path = []
        
        if type(np_pcd) == np.array:
            list_of_pcd = self.preprocess_scene([np_pcd], [np_label], dataset_name)

        elif type(np_pcd) == list:
            list_of_pcd = self.preprocess_scene(np_pcd, np_label, dataset_name)

        for i in range(len(list_of_pcd)):

            vis.clear_geometries()  # Clear pcd and reset camera pose
            vis.add_geometry(list_of_pcd[i])  # Add pcd seqencnce
            ctr.convert_from_pinhole_camera_parameters(trajectory.parameters[0], allow_arbitrary=False)  # Set camera pose
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
        frame_size = (1920, 1080)
        out = cv2.VideoWriter(os.path.join("videos", video_name),cv2.VideoWriter_fourcc(*'mp4v'), 15, frame_size)
        
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
    video_name = "test_video.mp4"
    visualizer.visualize(list_of_np_pcd, list_of_label, "nuscene", video_name)