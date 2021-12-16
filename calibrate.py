import cv2
from datetime import *
import glob
import numpy as np
import freenect
import open3d as o3d

# Board Size
board_h = 9
board_w = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w*board_h,3), np.float32)
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array

def  FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (board_w,board_h),None)

    # If found, display image with corners
    if ret == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

    return ret, corners


def main():

    while True:
        frame = get_video()

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)

        if key == ord("p"):
            print("Image Captured")
            date_img = datetime.now().strftime("%H:%M:%S_%Y")
            cv2.imwrite('cal_imgs/cal_img_' + date_img + '.jpg', frame)

        if key == ord("q"):
            break

    # Read images
    images = glob.glob('cal_imgs/cal_img*')
    dimensions=None

    for image in images:
        img = cv2.imread(image)
        dimensions=img.shape[:2]
        ret, corners = FindAndDisplayChessboard(img)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    #ret, matriz, distorção, rotacao, translação
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, dimensions, None, None)

    #save parameters
    np.savez("camera.npz" , intrinsics = mtx , distortion = dist)

    cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()