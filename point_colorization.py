import os
import numpy as np
import open3d as o3d
import argparse
from nuscene_reader import NuSceneReader
from utils import read_yaml

NUSCENE_PATH = os.getcwd()
COLOR_MAP_PATH = "color_map.yaml"


class Visualizer:
    def __init__(self, data_path, color_map_path):
        self.nuscene_reader = NuSceneReader(data_path)
        self.color_map_path = color_map_path
        self.zero_base_color_map = read_yaml(color_map_path)['zero_base_color_map']
        self.color_map = read_yaml(color_map_path)['color_map']
        self.normailize_color_maps()  # normailize each color

    def load_scene(self, scene_num):
        list_of_pcd = []
        list_of_np_pcd, list_of_label = self.nuscene_reader.load_data_for_scene(scene_num)
        
        for np_pcd, np_label in zip(list_of_np_pcd, list_of_label):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_pcd)  # change format from np.array to o3d.geometry.PointCloud
            np_color = self.map_colors(np_label)
            pcd.colors = o3d.utility.Vector3dVector(np_color)  # map colors to corresponding points
            
            list_of_pcd.append(pcd)

        return list_of_pcd
        
    def map_colors(self, np_label, is_zero_base=True):
        color_map = self.zero_base_color_map if is_zero_base else self.color_map
        np_color = np.vectorize(color_map.__getitem__, otypes=[np.ndarray])(np_label)

        return np_color

    def normailize_color_maps(self):
        for label in self.zero_base_color_map.keys():
            self.zero_base_color_map[label] = [x / 256 for x in self.zero_base_color_map[label]]
            self.zero_base_color_map[label] = np.asarray(self.zero_base_color_map[label]).T
        
        for label in self.color_map.keys():
            self.color_map[label] = [x / 256 for x in self.color_map[label]]
            self.color_map[label] = np.asarray(self.color_map[label]).T

    def visualize(self, scene_num):
        list_of_pcd = self.load_scene(scene_num)
        # TODO: set to bird eye view!
        o3d.visualization.draw_geometries([list_of_pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', default=NUSCENE_PATH, type=str)
    parser.add_argument('--color_map_path', dest='color_map_path', default=COLOR_MAP_PATH)
    parser.add_argument('--scene_num', dest='scene_num', default=1, type=int)
    config = parser.parse_args()
    visualizer = Visualizer(config.data_path, config.color_map_path)
    visualizer.visualize(config.scene_num)