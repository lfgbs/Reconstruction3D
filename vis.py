import open3d as o3d
import copy
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Parser for mmc in 3D Reconstruction project - VC')
parser.add_argument('--clouds_path', type=str, required=True, help='Path to folder that contains the scenes pointclouds')
args = parser.parse_args()

pcd= pcd=o3d.io.read_point_cloud(args.clouds_path)

""" with np.load('camera.npz') as params:
    intrinsics= params["intrinsics"]
    distortion=params["distortion"]

print(intrinsics)
print(distortion) """

o3d.visualization.draw_geometries([pcd])