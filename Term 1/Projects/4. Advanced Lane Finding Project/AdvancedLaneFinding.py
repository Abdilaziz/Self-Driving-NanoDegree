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

def plotImage(title, image, cmap=None):

    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()

def plotMultiImage(*titles, images):
    info = zip(titles,images)
    for i in info:
        if(len(i[1].shape)==2):
            cmap='gray'
        else:
            cmap=None

        plotImage(i[0], i[1],cmap)

def plotImages(*titles , images):
    numbOfimages = len(images)
    f, disp = plt.subplots(numbOfimages, 1)
    i = 0
    cmap = None
    for title in titles:
        if len(images[i].shape)==2:
            cmap='gray'
        else:
            cmap=None
        disp[i].imshow(images[i], cmap=cmap)
        disp[i].set_title(title, fontsize=25)
        i= i +1
    plt.show()

def get_Saturation(image, thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    saturation = hls[:,:,2]
    binary = np.zeros_like(saturation)
    binary[(saturation>=thresh[0]) & (saturation<= thresh[1])]=1
    return binary

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # gray = img
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

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    
    # gray = image
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
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    # 6) Return this mask as your binary_output image
    mag_binary = sxbinary
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # gray = image
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


def process_Image(img, mtx, dist):
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    #Thresholds

    ksize = 15

    # Apply each of the thresholding functions
    saturation = get_Saturation(undist_img, thresh=(175,255))
    gradx = abs_sobel_thresh(undist_img, orient='x', sobel_kernel=ksize, thresh=(40, 100))
    grady = abs_sobel_thresh(undist_img, orient='y', sobel_kernel=ksize, thresh=(40, 100))
    mag_binary = mag_thresh(undist_img, sobel_kernel=ksize, thresh=(50, 100))
    dir_binary = dir_threshold(undist_img, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)

    combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1) | (saturation==1))) ] = 1

    img_size = (combined.shape[1], combined.shape[0])
    offset = 300
    src = np.float32([(550,480), (730,480), (1180,img_size[1]), (190,img_size[1])])
    dest = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    M = cv2.getPerspectiveTransform(src,dest)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_CUBIC)

    return combined, undist_img, warped





cameraCalibrationFile = 'CalibrationValues.p'

try:
    dist_pickle = pickle.load(open(cameraCalibrationFile, "rb"))
    print('Using {}'.format(cameraCalibrationFile))
except (OSError, IOError) as e:
	print("{} doesn't exist. Creating Pickle Now".format(cameraCalibrationFile))
	dist_pickle = calibrate_camera(cameraCalibrationFile)

mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


img = mpimg.imread('CarND-Advanced-Lane-Lines/test_images/straight_lines1.jpg')


output, undist_img, warped = process_Image(img, mtx, dist)


plotImages('Undistorted Image', 'Output', images=[undist_img,warped])

mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/processed_image.jpg',output, cmap='gray')
mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/warped_image.jpg',warped, cmap='gray')


# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vdertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows    
    for level in range(0,len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    print(l_points.shape)
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# Display the final results
plt.imshow(output)
plt.title('window fitting results')
plt.show()
