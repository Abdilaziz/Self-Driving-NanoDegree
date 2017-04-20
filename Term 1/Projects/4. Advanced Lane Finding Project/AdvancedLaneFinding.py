import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
import os

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

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):

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

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    
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

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    
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

def findLanes(binary_warped, left_lines, right_lines):

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # plt.plot(histogram)
    # plt.show()

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    if (len(left_lines.recent_xfitted) !=0):
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit = left_lines.current_fit[-1]
        right_fit= right_lines.current_fit[-1]
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    else:
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)







    leftlane_detected = False
    rightlane_detected = False
    if (len(left_lane_inds) > 0) :
        leftlane_detected = True
    if (len(right_lane_inds) > 0 ):
        rightlane_detected = True


    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]  







    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = binary_warped.shape[0]
    # left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    # right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    # print('Histogram Method Pixel Curvature Values: Left Lane: {} , Right Lane: {}'.format(left_curverad, right_curverad))
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


    # get distance in meters from the cars position (center of the image)

    leftlanes_centerPos = left_fit_cr[0]*(binary_warped.shape[0]*ym_per_pix)**2 + left_fit_cr[1]*(binary_warped.shape[0]*ym_per_pix) + left_fit_cr[2]
    rightlanes_centerPos = right_fit_cr[0]*(binary_warped.shape[0]*ym_per_pix)**2 + right_fit_cr[1]*(binary_warped.shape[0]*ym_per_pix) + right_fit_cr[2]

    carPosition = (binary_warped.shape[1]/2)*xm_per_pix
    leftDistance = carPosition - leftlanes_centerPos
    rightDistance = rightlanes_centerPos -carPosition


    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.title('Histogram Window Sliding')
    # plt.savefig('CarND-Advanced-Lane-Lines/output_images/HistogramOutput.jpg')

    # plt.show()

    # Create an image to draw on and an image to show the selection window
    # out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    # result = cv2.addWeighted(out_img, 2, window_img, 0.3, 0)
    # plt.imshow(result)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)

    # plt.title('Histogram Window Sliding')
    # plt.savefig('CarND-Advanced-Lane-Lines/output_images/HistogramOutput.jpg')

    # plt.show()
    left_lines.update(leftlane_detected, left_fitx, left_fit, left_curverad, leftDistance,   leftx, lefty )
    right_lines.update(rightlane_detected, right_fitx, right_fit, right_curverad, rightDistance, rightx, righty)

    return left_lines, right_lines

def conv_findLanes(warped):

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
        leftx = np.zeros(int(warped.shape[0]/window_height))
        rightx = np.zeros(int(warped.shape[0]/window_height))
        y_val = np.zeros(int(warped.shape[0]/window_height))

        for i in range(len(y_val)):
            y_val[i] = warped.shape[0]- i*window_height

        # Go through each level and draw the windows    
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            leftx[level] = window_centroids[level][0]
            rightx[level] = window_centroids[level][1]
            l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results

        left_fit = np.polyfit(y_val, leftx, 2)
        right_fit = np.polyfit(y_val, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Define y-value where we want radius of curvature
        # I'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        print('Convolutional Method Pixel Curvature Values: Left Lane: {} , Right Lane: {}'.format(left_curverad, right_curverad))
        # Example values: 1926.74 1908.48

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(y_val*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(y_val*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print('Convolutional Method Curvature Values: Left Lane: {} m , Right Lane: {} m'.format(left_curverad, right_curverad))
        # Example values: 632.1 m    626.2 m


        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

    # Display the final results
    # plt.imshow(output)
    # # plt.imshow(warped, cmap='gray')
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 1280)
    # plt.ylim(720, 0)
    # plt.title('Convolutional Window Sliding')
    # plt.savefig('CarND-Advanced-Lane-Lines/output_images/ConvolutionalOutput.jpg')
    # plt.show()

def process_Image(img, mtx, dist, left_lines, right_lines, path=''):
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    ksize = 15

    gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)

    # Apply each of the thresholding functions
    # saturation = get_Saturation(undist_img, thresh=(175,255))
    # gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 100))
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(40, 100))
    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(50, 100))
    # dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    saturation = get_Saturation(undist_img, thresh=(160,255))
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(40, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(40, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(50, 100))
    dir_binary = dir_threshold(saturation, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)

    combined[(((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1) | (saturation==1))) ] = 1

    # kernel_size =5
    # combined = cv2.GaussianBlur(combined,(kernel_size, kernel_size),0)

    img_size = (combined.shape[1], combined.shape[0])
    offset = 300

    src = np.float32([(608,440), (672,440), (1150,img_size[1]), (180,img_size[1])])
    # src = np.float32([(550,480), (740,480), (1150,img_size[1]), (190,img_size[1])])
    dest = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset, img_size[1]]])

    M = cv2.getPerspectiveTransform(src,dest)
    Minv = cv2.getPerspectiveTransform(dest,src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_NEAREST) # INTER_NEAREST


    if path !='':
        # Testing Prespective Transforms Output
        line_image = np.copy(undist_img)*0
        undist_warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_NEAREST) # INTER_NEAREST
        cv2.line(line_image,(offset,0),(offset,img_size[1]),(255,0,0),10) 
        cv2.line(line_image,(img_size[0] - offset,0),(img_size[0] - offset,img_size[1]),(0,0,255),10) 
        lines_edges = cv2.addWeighted(undist_warped, 0.8, line_image, 1, 0)

        # proc_undist = cv2.warpPerspective(lines_edges, Minv, img_size, flags=cv2.INTER_CUBIC)
        # proc_undist = cv2.addWeighted(undist_img, 0.8, proc_undist, 1, 0)

        # plotImages('Undistorted Image', 'Output', images=[proc_undist,lines_edges])

        # plotImages('Gray', 'Saturation', images=[gray,saturation])

        # plotImages('Undistorted Image', 'Output', images=[undist_img,warped])

        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path +'undistorted_image.jpg',undist_img, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path +'thresholded_image.jpg',combined, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path + 'warped_image.jpg',warped, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path + 'allignment_check.jpg',lines_edges, cmap='gray')

    left_lines, right_lines = findLanes(warped, left_lines, right_lines)
    # conv_findLanes(warped)


    # Create an image to draw the lines on


    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    pts_left = np.array([np.transpose(np.vstack([left_lines.recent_xfitted[len(left_lines.recent_xfitted)-1], ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lines.recent_xfitted[len(right_lines.recent_xfitted)-1], ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    # mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/Fully_Proccessed_Image.jpg',result, cmap='gray')


    return result, left_lines, right_lines

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

    def update(self, detected, xValues, fitCoeff, radius_of_curvature, lanes_distance, detectedx, detectedy):
        self.detected = detected

        n = 20
        self.recent_xfitted.append(xValues)
        if (len(self.recent_xfitted) > n):
            self.recent_xfitted = self.recent_xfitted[1:]
        self.bestx = np.mean(self.recent_xfitted, axis=0)

        self.current_fit.append(fitCoeff)
        if (len(self.current_fit) > n):
            self.current_fit = self.current_fit[1:]
        self.best_fit = np.mean(self.current_fit, axis=0)

        self.radius_of_curvature = radius_of_curvature

        self.line_base_pos = lanes_distance

        if(len(self.current_fit)>1):
            self.diffs = np.diff(np.vstack([fitCoeff, self.current_fit[-2]] ), axis=0)

        self.allx = detectedx

        self.ally = detectedy

def runOnImages(mtx, dist):
    images = glob.glob('CarND-Advanced-Lane-Lines/test_images/straight_lines*.jpg')
    images.extend(glob.glob('CarND-Advanced-Lane-Lines/test_images/test*.jpg'))

    for idx, fname, in enumerate(images):

        inputImgName = os.path.basename(fname)[:-4]
        print(inputImgName)
        inputImgPath = 'CarND-Advanced-Lane-Lines/test_images/'+ inputImgName +'.jpg'
        outputImgPath = 'CarND-Advanced-Lane-Lines/output_images/'+ inputImgName+ '/' +'Fully_Proccessed_Image.jpg'
        os.makedirs(os.path.dirname(outputImgPath), exist_ok=True)
        img = mpimg.imread(inputImgPath)

        left_lines = Line()
        right_lines = Line()

        folder = inputImgName + '/'
        result, left_lines, right_lines = process_Image(img, mtx, dist, left_lines, right_lines, path=folder)

        # Now our radius of curvature is in meters
        print('Histogram Method Curvature Values: Left Lane: {} m , Right Lane: {} m'.format(left_lines.radius_of_curvature, right_lines.radius_of_curvature))
        # Example values: 632.1 m    626.2 m

        mpimg.imsave(outputImgPath,result, cmap='gray')

        # plt.imshow(result)
        # plt.show()

        # print(left_lines.line_base_pos)

def runOnVideo(mtx, dist):
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip, ImageSequenceClip
    # from IPython.display import HTML

    inputVideo = 'project_video'
    video_output = inputVideo + 'output.mp4'
    inputVideoPath = "CarND-Advanced-Lane-Lines/"+inputVideo+".mp4"
    clip1 = VideoFileClip(inputVideoPath)

    left_lines = Line()
    right_lines = Line()

    new_frames = []
    # i = 0

    # 20 to 26 is the difficult area of the video
    clip1 = clip1.subclip(0, 3)

    print('Processing Each Frame of the Video')
    for frame in clip1.iter_frames():
        # i = i +1

        result, left_lines, right_lines = process_Image(frame, mtx, dist, left_lines, right_lines)
        # print('Completed: {}'.format(i))
        new_frames.append(result)

    new_clip = ImageSequenceClip(new_frames, fps=clip1.fps)
    new_clip.write_videofile(video_output)




cameraCalibrationFile = 'CalibrationValues.p'

try:
    dist_pickle = pickle.load(open(cameraCalibrationFile, "rb"))
    print('Using {}'.format(cameraCalibrationFile))
except (OSError, IOError) as e:
    print("{} doesn't exist. Creating Pickle Now".format(cameraCalibrationFile))
    dist_pickle = calibrate_camera(cameraCalibrationFile)


mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


runOnImages(mtx, dist)

# runOnVideo(mtx, dist)