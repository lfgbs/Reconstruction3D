import freenect
import cv2
import numpy as np
import open3d as o3d
from datetime import *
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()  
    array = array.astype(np.uint8)
    return array

def main():
     
    #parâmetros intrínsecos da câmara, num formato que será utilizado para a conversão depth para pcd 
    intrinsics = o3d.camera.PinholeCameraIntrinsic(680, 420, 594.21, 591.04, 339.5, 242.7)
    count=0


    while True:
        #get a frame from RGB camera
        frame = get_video()
        frame_bgr = cv2.flip(frame, 1)  # the second arguments value of 1 indicates that we want to flip horizontally
        #get a frame from depth sensor
        depth = get_depth()
        #depth = cv2.flip(depth, 1)  # the second arguments value of 1 indicates that we want to flip horizontally
        #display RGB image
        frame_bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        cv2.imshow('RGB image',frame_bgr)
        #display depth image
        cv2.imshow('Depth image',depth)
 
        # quit program when 'esc' key is pressed
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("p"):
            
            cv2.imwrite( "send/frame"+str(count)+".png" , frame)
            cv2.imwrite(  "send/depth"+str(count)+".png", depth)


            rgb_img=o3d.geometry.Image(frame)
            depth_img=np.float32(depth)
            depth_map=o3d.geometry.Image(depth_img)
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_map, convert_rgb_to_intensity=False)
            #pcd_depth=o3d.geometry.PointCloud.create_from_depth_image(depth_map, intrinsics)
            pcd_rgbd=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

            date_img = datetime.now().strftime("%H:%M:%S_%Y")
            
            #pcd_depth.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #flip it upside down
            pcd_rgbd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) #flip it upside down
            #o3d.io.write_point_cloud("clouds/roomdepth"+date_img+".pcd", pcd_depth)
            #o3d.io.write_point_cloud("clouds/roomrgbd"+date_img+".pcd", pcd_rgbd)
            o3d.io.write_point_cloud("clouds/resi/roomrgbd"+str(count)+".pcd", pcd_rgbd)
            #o3d.visualization.draw_geometries([pcd_rgbd])

            count+=1

    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
   