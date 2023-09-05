import os
from utils import read_yaml
import numpy as np
import open3d as o3d

class NuSceneLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.total_scene_num = len(os.listdir(data_path))
    
    def generate_scene_path(self, scene_num: int) -> dict:
        scene_file_name = "scene-" + "%04d" % scene_num
        
        return {
            "clip_path": os.path.join(self.data_path, scene_file_name, "clip"),
            "color_path": os.path.join(self.data_path, scene_file_name, "color"),
            "K": os.path.join(self.data_path, scene_file_name, "K"),
            "label": os.path.join(self.data_path, scene_file_name, "label"),
            "lidar": os.path.join(self.data_path, scene_file_name, "lidar"),
            "lidar_pose": os.path.join(self.data_path, scene_file_name, "lidar_pose"),
            "mask": os.path.join(self.data_path, scene_file_name, "mark"),
            "output": os.path.join(self.data_path, scene_file_name, "output"),
            "pose": os.path.join(self.data_path, scene_file_name, "pose")
        }
    
    @staticmethod
    def load_file_paths_for_scene(file_scene_path):
        list_of_file_path = []

        for file_name in os.listdir(file_scene_path):
            list_of_file_path.append(os.path.join(file_scene_path, file_name))

        return list_of_file_path

    def load_pcds_for_scene(self, scene_num: int):
        pcd_scene_path = self.generate_scene_path(scene_num)["lidar"]
        list_of_pcd_path = self.load_file_paths_for_scene(pcd_scene_path)            
        list_of_pcd = []

        for _path in list_of_pcd_path:
            pcd = np.fromfile(_path, dtype=np.float32)
            pcd = pcd.reshape(-1, 4)[:, : 3]
            list_of_pcd.append(pcd)  # read nuScene pcd and crop only x, y, z

        return list_of_pcd
    
    def load_labels_for_scene(self, scene_num: int):
        label_scene_path = self.generate_scene_path(scene_num)["label"]
        list_of_label_path = self.load_file_paths_for_scene(label_scene_path)
        list_of_label = []

        for _path in list_of_label_path:
            list_of_label.append(np.fromfile(_path, dtype=np.uint8))

        return list_of_label
    
    def load_lidar_poses_for_scene(self, scene_num: int):
        lidar_pose_scene_path = self.generate_scene_path(scene_num)["lidar_pose"]
        list_of_lidar_pose_path = self.load_file_paths_for_scene(lidar_pose_scene_path)
        list_of_lidar_pose = []

        for _path in list_of_lidar_pose_path:
            list_of_row_of_extrinsic = []
            with open(_path, "r") as f:
                for line in f.readlines():
                    list_of_row_of_extrinsic.append([float(x) for x in line.strip().split()])
                
                lidar_pose = np.array(list_of_row_of_extrinsic)
                list_of_lidar_pose.append(lidar_pose)

        return list_of_lidar_pose
    
    def load_data_for_scene(self, scene_num: int):
        list_of_pcd = self.load_pcds_for_scene(scene_num)
        list_of_label = self.load_labels_for_scene(scene_num)
        list_of_lidar_pose = self.load_lidar_poses_for_scene(scene_num)    
        for i in range(0, len(list_of_pcd)):
            # remove noises
            list_of_label[i] = list_of_label[i] & 0xFF  # filt out odd labels
            noise_indexs = np.where(list_of_label[i] < 24)[0].tolist()
            list_of_pcd[i] = np.delete(list_of_pcd[i], noise_indexs, axis=0)
            list_of_label[i] = np.delete(list_of_label[i], noise_indexs, axis=0)
            
            noise_indexs = np.where(list_of_label[i] > 31)[0].tolist()
            list_of_pcd[i] = np.delete(list_of_pcd[i], noise_indexs, axis=0)
            list_of_label[i] = np.delete(list_of_label[i], noise_indexs, axis=0)


            list_of_label[i] -= 24  # shift label to start from 0

        return list_of_pcd, list_of_label, list_of_lidar_pose
    