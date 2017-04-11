import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

def calibrate_camera(pickleFile):

	# Get Camera Matrix an distortion coefficients
	nx = 9
	ny = 6

	images = glob.glob('CarND-Advanced-Lane-Lines/camera_cal/calibration*.jpg')

	objpoints = [] # 3D points in real world space
	imgpoints = [] # 2D points in image plance

	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

	image_shape = ()
	print('Number of Calibration Images: {}'.format(len(images)))

	for idx, fname, in enumerate(images):
		img = mpimg.imread(fname)
		image_shape = img.shape
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

		ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)

		if ret == True:
			imgpoints.append(corners)
			objpoints.append(objp)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[0:2],None,None)

	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	pickle.dump( dist_pickle, open(pickleFile, "wb" ))

	print('OUTPUT:       Camera Matrix: mtx     Distortion Coefficients: dist')

	return dist_pickle

def plot2Images(title1, image1, title2, image2, cmap=''):
	f, (axis1, axis2) = plt.subplots(1,2, figsize=(24,9))
	axis1.imshow(image1)
	axis1.set_title(title1, fontsize=50)
	axis2.imshow(image2, cmap=cmap)
	axis2.set_title(title2, fontsize=50)
	plt.show()

def get_Saturation(image):
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	saturation = hls[:,:,2]
	return saturation

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
	# Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel= cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    grad_binary = sxbinary
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(100, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sobelmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = 255*sobelmag/np.max(sobelmag)
    # 5) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1 
    # 6) Return this mask as your binary_output image
    mag_binary = sxbinary
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.3)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely= cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobely,abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_mask = np.zeros_like(dir_grad)
    binary_mask[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    dir_binary = binary_mask
    return dir_binary

cameraCalibrationFile = 'CalibrationValues.p'

try:
    dist_pickle = pickle.load(open(cameraCalibrationFile, "rb"))
    print('Using {}'.format(cameraCalibrationFile))
except (OSError, IOError) as e:
	print("{} doesn't exist. Creating Pickle Now".format(cameraCalibrationFile))
	dist_pickle = calibrate_camera(cameraCalibrationFile)

mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

img = mpimg.imread('CarND-Advanced-Lane-Lines/test_images/test2.jpg')

undist_img = cv2.undistort(img, mtx, dist, None, mtx)

# plot2Images('Original Image', img, 'Undistorted Image', undist_img)


#Thresholds

ksize = 15

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(undist_img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
grady = abs_sobel_thresh(undist_img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
mag_binary = mag_thresh(undist_img, sobel_kernel=ksize, mag_thresh=(30, 100))
dir_binary = dir_threshold(undist_img, sobel_kernel=ksize, thresh=(0.7, 1.3))


combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

plot2Images('Undistorted Image', undist_img, 'Threshold Grad. Dir', combined, cmap='gray')