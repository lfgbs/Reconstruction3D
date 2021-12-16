import freenect
import cv2
import numpy as np
import open3d as o3d
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()  
    array = array.astype(np.uint8)
    return array

def get_pointcloud(depth_map):
    #We must reproject depth map to 3d
    
    depth_map=np.float32(depth_map)
    depth_map=depth_map.reshape(-1,3)

    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(depth_map)
    
    o3d.visualization.draw_geometries([pcd])

    return pcd
 
if __name__ == "__main__":
    while True:
        #get a frame from RGB camera
        frame = get_video()
        frame = cv2.flip(frame, 1)  # the second arguments value of 1 indicates that we want to flip horizontally
        #get a frame from depth sensor
        depth = get_depth()
        depth = cv2.flip(depth, 1)  # the second arguments value of 1 indicates that we want to flip horizontally
        #display RGB image
        cv2.imshow('RGB image',frame)
        #display depth image
        cv2.imshow('Depth image',depth)
 
        # quit program when 'esc' key is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("p"):
            depth_img=get_depth
            print(depth_img)
            print(depth_img.shape)
            #pcd=get_pointcloud(depth)
            #o3d.io.write_point_cloud("clouds/test.pcd", pcd)

    cv2.destroyAllWindows()