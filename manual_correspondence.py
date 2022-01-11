import open3d as o3d
import copy
import numpy as np

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

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
    print("Load a ply point cloud, print it, and render it")
    pcd1 = o3d.io.read_point_cloud("clouds/room2.pcd")
    pcd2 = o3d.io.read_point_cloud("clouds/room3.pcd")

    downpcd1 = pcd1.voxel_down_sample(1)
    downpcd2 = pcd2.voxel_down_sample(1)

    draw_registration_result(downpcd1, downpcd2, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(downpcd1)
    picked_id_target = pick_points(downpcd2)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(downpcd1, downpcd2,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 1  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        downpcd1, downpcd2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(downpcd1, downpcd2, reg_p2p.transformation)
    print("")

if __name__ == "__main__":
    main()