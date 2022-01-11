import numpy as np
from numpy.lib.twodim_base import eye
import open3d as o3d
import copy
import math


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    #source_temp.paint_uniform_color([1, 0.706, 0])
    #target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def main():

    #loading and showing pcd
    print("Load a ply point cloud, print it, and render it")
    pcd2 = o3d.io.read_point_cloud("clouds/roomrgbd2.pcd")
    pcd1 = o3d.io.read_point_cloud("clouds/roomrgbd3.pcd")

    downpcd1 = pcd1.voxel_down_sample(0.001)
    downpcd2 = pcd2.voxel_down_sample(0.001)    

    trans_init = np.asarray([[math.cos(-math.pi/12),0.0, math.sin(-math.pi/12),0.0], [0.0,1.0, 0.0, 0.0], [-math.sin(-math.pi/12),  0.0, math.cos(-math.pi/12), 0.0], [0.0, 0.0, 0.0, 1.0] ])

    threshold=0.01

    draw_registration_result(pcd1, pcd2, trans_init)

    print("Initial alignment(downsampled)")
    evaluation = o3d.pipelines.registration.evaluate_registration(
    downpcd1, downpcd2, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        downpcd1, downpcd2, threshold,trans_init, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)

    #desenhamos com a transformação calculada com os downsampled
    draw_registration_result(downpcd1, downpcd2, reg_p2p.transformation)
    

if __name__ == "__main__":
    main()