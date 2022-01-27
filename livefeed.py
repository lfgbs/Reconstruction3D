import numpy as np
import open3d as o3d
import argparse
import glob
from global_registration import *

def main():
     
    parser = argparse.ArgumentParser(description='Parser for mmc in 3D Reconstruction project - VC')
    parser.add_argument('--clouds_path', type=str, required=True, help='Path to folder that contains the scenes pointclouds')
    args = parser.parse_args()

    voxel_size=0.003
    
    pcds=[] #holds the clouds
    combined_clouds=o3d.geometry.PointCloud() #amalgamation de todas as nuvens

    #carregar os paths de todas as nuvens
    pcd_paths=glob.glob(args.clouds_path+'/*.pcd')
    #certificar que estão organizadas para que possa ser feito o alinhamento
    pcd_paths=sorted(pcd_paths)

    #carregar as nuvens usando o path
    for pcd_path in pcd_paths:
        pcd=o3d.io.read_point_cloud(pcd_path)
        pcds.append(pcd)

    count=0
    while pcds!=[]:
        if count==0:
            source_down=pcds[0]
            combined_clouds+=source_down.voxel_down_sample(voxel_size)
            count+=1
            continue

        source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(pcds[0], combined_clouds, voxel_size)
        
        """ result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_icp = refine_registration(pcds[0], combined_clouds, source_fpfh, target_fpfh, voxel_size, result_ransac)

        source_down.transform(result_icp.transformation)
        print(result_icp)
        print(result_ransac) """

        result_fast = execute_fast_global_registration(source_down, target_down,source_fpfh, target_fpfh,voxel_size)
        print(result_fast)

        source_down.transform(result_fast.transformation)

        combined_clouds+=source_down
        
        pcds=pcds[1:] #remove a cloud da lista para poupar memória 
        count+=1
        print(count)
        
    o3d.visualization.draw_geometries([combined_clouds])



if __name__ == "__main__":
    main()
   