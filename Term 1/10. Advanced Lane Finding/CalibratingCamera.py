import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# prepare object points
nx = 8
ny = 6


# Make a list of calibration images
# typically you want a list of 20 calibration images
# so 20 images of a chessboard from different distances and angles
# also have 1 extra image as a final test for your calibration

# Glob lets you get multible image names with the same structure in the specified path
images = glob.glob('calibration_wide/GO*.jpg')


# Map the points of the 2D image (Image Points), to points in the real undistorted 3D Image (Object Points)
#Object Points
# Object points are (x,y,x) cordinates of the real image.
# they are just the known co-ordinates for a (nx,ny) shaped board
# Because the chess board is on a flat surface, all z points are 0
# X and Y values are just the points of the corners (x = 0 to nx-1, y = 0 to ny-1)

objpoints = [] # 3D points in real world space

# Image Points
# Points in the distorted 2D image

imgpoints = [] # 2D points in image plance

# we want objpoints to be x = 0 to nx-1 and y = 0 to ny -1
# initialize object points as an array of 3 columns and length of nx*ny (Same shape as 3D image)
objp = np.zeros((nx*ny,3), np.float32)
# we want all the values in the z array to stay zero, so we change the first 2  arrays 
#mgrid function returns the co-ordinate values for a given size, then we reshape it into the correct format of 2 columns for x and y
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

print('Number of Calibration Images: {}'.format(len(images)))
# LOOP FROM HERE FOR ALL CALIBRATION IMAGES
for idx, fname, in enumerate(images):
	img = cv2.imread(fname)

	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	# the image points are the points of the corners of the distorted calibration image (ouput of find chessboard corners)

	# Find the chessboard corners
	# input must be an 8bit grayscale image
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

	# If found, add the object points and image points
	if ret == True:
		# append these values for every calibration image 
		# all these ponints can be used to calibrate the camera with an opencv function
		imgpoints.append(corners)
		objpoints.append(objp)

	    # Draw and display the corners
	    # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
	    # plt.imshow(img)
	    # plt.show()

# now that we have our Object points and image points, we can use them to calculate the 
# distortion coefficients and camera matrix required to undistort the image

#import a distorted image to test on
img = cv2.imread('calibration_wide/test_image.jpg')
print('ObjPoints: {}. ImgPoints: {}'.format(len(objpoints), len(imgpoints)))
print('Shape of Test Image: {}'.format(img.shape[0:2]))
# takes the object points, image points, and the shape of the image (only height and width)
# it returns the distortion coefficients and the camera matrix needed to transform
# 3D object points to 2D image points, it also returns the position of the camera in the world
# with rotation and translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2],None,None)
# another function can return the undistorted image using the output camera matrix
# and the distortion coefficients

# output of undistort is often called the destination image
dst = cv2.undistort(img, mtx, dist, None, mtx)

f, axis = plt.subplots(1,2)
axis[0].imshow(img)
axis[1].imshow(dst)

plt.show()

# we can save the Camera calibration results into a pickle so that we can use them later

# import pickle


# dist_pickle = {}
# dist_pickle["mtx"] = mtx
# dist_pickle["dist"] = dist
# pickle.dump( dist_pickle, open("calibration_wide/wide_dist_pickle.p", "wb" ))


# print('Camera Matrix and Distortion Coefficients written to calibration_wide/wide_dist_pickle.p')