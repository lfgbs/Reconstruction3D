import open3d as o3d
import copy
import numpy as np
import argparse
import glob

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def main():
    #loading and showing pcd
    #print("Load a ply point cloud, print it, and render it")

    parser = argparse.ArgumentParser(description='Parser for mmc in 3D Reconstruction project - VC')
    parser.add_argument('--clouds_path', type=str, required=True, help='Path to folder that contains the scenes pointclouds')
    args = parser.parse_args()

    pcds=[] #holds the clouds
    transformations=[] #holds the respective transformations
    combined_clouds=o3d.geometry.PointCloud() #amalgamation de todas as nuvens

    #carregar os paths de todas as nuvens
    pcd_paths=glob.glob(args.clouds_path+'/*.pcd')
    #certificar que estão organizadas para que possa ser feito o alinhamento
    pcd_paths=sorted(pcd_paths)

    #carregar as nuvens usando o path
    for pcd_path in pcd_paths:
        pcd=o3d.io.read_point_cloud(pcd_path)
        pcds.append(pcd)

    for i in range(len(pcd_paths)):
        # pick points from two point clouds and builds correspondences
        if i==0:
            transformations.append(np.identity(4))
        else:
            picked_id_source = pick_points(pcds[i])
            picked_id_target = pick_points(pcds[i-1])
            assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
            assert (len(picked_id_source) == len(picked_id_target))
            corr = np.zeros((len(picked_id_source), 2))
            corr[:, 0] = picked_id_source
            corr[:, 1] = picked_id_target

            # estimate rough transformation using correspondences
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(pcds[i], pcds[i-1], o3d.utility.Vector2iVector(corr))
            transformations.append(trans_init)

    #A cada nuvem aplicar todas as transformações que se lhe antecederam
    for index, cloud in enumerate(pcds):
        for i in range(index):
            cloud.transform(transformations[index-i])

        if index!=0:
            # point-to-point ICP for refinement
            print("Perform point-to-point ICP refinement")
            threshold = 0.0001 
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcds[i], pcds[i-1], threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

            
        combined_clouds+=cloud

    o3d.io.write_point_cloud(args.clouds_path+"/combined_clouds.pcd" ,combined_clouds)
    
    """  checking=o3d.io.read_point_cloud(args.clouds_path+"/combined_clouds.pcd")

    print("FINAL CLOUD!!!!!!!!!!!1")
    o3d.visualization.draw_geometries([checking]) """
    

if __name__ == "__main__":
    main()