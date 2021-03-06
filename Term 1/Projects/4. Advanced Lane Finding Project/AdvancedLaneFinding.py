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
	# maps 3D points to 2D points and returns the required distortion matrix and distortion coefficients
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape[0:2],None,None)
	dst = cv2.undistort(mpimg.imread(images[0]), mtx, dist, None, mtx)
	mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/calibrationAfter1.jpg',dst)
	# writes camera calibration values to a pickle, so that it can be reused

	dist_pickle = {}
	dist_pickle["mtx"] = mtx
	dist_pickle["dist"] = dist
	pickle.dump( dist_pickle, open(pickleFile, "wb" ))

	print('OUTPUT:       Camera Matrix: mtx     Distortion Coefficients: dist')

	return dist_pickle

def plotImage(title, image, cmap=None):
    # Displays one image
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.show()

def plotMultiImage(*titles, images):
    # Displays a variable number of Images in seperate windows
    info = zip(titles,images)
    for i in info:
        if(len(i[1].shape)==2):
            cmap='gray'
        else:
            cmap=None

        plotImage(i[0], i[1],cmap)

def plotImages(*titles , images):
    # Displays multiple images in the same window
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
    # gets the saturation values of an image and thresholds it
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    saturation = hls[:,:,2]
    binary = np.zeros_like(saturation)
    binary[(saturation>=thresh[0]) & (saturation<= thresh[1])]=1
    return binary

def get_Lightness(image, thresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    lightness = hls[:,:,1]
    binary = np.zeros_like(lightness)
    binary[(lightness>=thresh[0]) & (lightness<= thresh[1])]=1
    return binary

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):

    # applys the Sobel Operator to a gray scale image to take the dirivative in direction of the orient parameter, then applys thresholding
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel= cv2.Sobel(gray,cv2.CV_64F,0,1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255)):
    
    # apply the sobel operator in both the x and y direction and thresholds the overall magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)
     
    sobelmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = 255*sobelmag/np.max(sobelmag)
    
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1 
    
    return mag_binary

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Gets the driection of the gradient through the use of arctan and thresholds the angle values to be within the parameter values
    sobelx = cv2.Sobel(gray, cv2.CV_64F,1,0, ksize=sobel_kernel)
    sobely= cv2.Sobel(gray, cv2.CV_64F,0,1, ksize=sobel_kernel)
    
    abs_sobelx = np.sqrt(sobelx**2)
    abs_sobely = np.sqrt(sobely**2)
    # Calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobely,abs_sobelx)
    
    dir_binary = np.zeros_like(dir_grad)
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    return dir_binary

def findLanes(binary_warped, left_lines, right_lines):

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # if (len(left_lines.recent_xfitted) !=0) & (left_lines.detected & right_lines.detected):
    if (len(left_lines.recent_xfitted) !=0) & (left_lines.detected & right_lines.detected):
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
        print('Blind Search')

        # This Blind searches the image for the center of the lane line values
        # print('Blind Searching')
        # Take a histogram of the bottom half of the warped and thresholded image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

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
            # print('Left Rect. win_xleft_low: {}, win_y_low: {}, win_xleft_high: {}, win_y_high: {}'.format(win_xleft_low, win_y_low, win_xleft_high, win_y_high))
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

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = binary_warped.shape[0]

    # Define conversions in x and y from pixels space to meters
    # Avg lane is 30 meters long and 3.7 meters wide
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # keep track of wether a lane was found from frame to frame
    leftlane_detected = False
    rightlane_detected = False
    if (left_lane_inds.nonzero()[0].size != 0 ) & (right_lane_inds.nonzero()[0].size != 0 ) :
        leftlane_detected = True
        rightlane_detected = True


        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]  

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting, These values are in pixel space
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])


        # get distance in meters from the cars position (center of the image at the bottom)
        leftlanes_centerPos = left_fit_cr[0]*(y_eval*ym_per_pix)**2 + left_fit_cr[1]*(y_eval*ym_per_pix) + left_fit_cr[2]
        rightlanes_centerPos = right_fit_cr[0]*(y_eval*ym_per_pix)**2 + right_fit_cr[1]*(y_eval*ym_per_pix) + right_fit_cr[2]

        carPosition = (binary_warped.shape[1]/2)*xm_per_pix
        leftDistance = carPosition - leftlanes_centerPos
        rightDistance = rightlanes_centerPos -carPosition

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_lines.update(leftlane_detected, left_fitx, left_fit, left_curverad, leftDistance, leftx, lefty )
        right_lines.update(rightlane_detected, right_fitx, right_fit, right_curverad, rightDistance, rightx, righty)
    else:



        avgLeftlanex = left_lines.bestx
        avgLeftcoeff = left_lines.best_fit

        avgRightlanex = right_lines.bestx
        avgRightcoeff = right_lines.best_fit

        left_curverad = left_lines.radius_of_curvature
        leftDistance = left_lines.line_base_pos

        right_curverad = right_lines.radius_of_curvature
        rightDistance = right_lines.line_base_pos



        lefty = np.array(left_lines.ally)
        leftx = np.array(left_lines.allx)
        righty = np.array(right_lines.ally)
        rightx = np.array(right_lines.allx)


        left_lines.update(leftlane_detected, avgLeftlanex, avgLeftcoeff, left_curverad, leftDistance, leftx, lefty )
        right_lines.update(rightlane_detected, avgRightlanex, avgRightcoeff, right_curverad, rightDistance, rightx, righty)
    return out_img, left_lines, right_lines

def process_Image(img, mtx, dist, left_lines, right_lines, path=''):

    # Undistort the image with the distribution parameters and the camera matrix
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)

    ksize = 15

    # Apply each of the thresholding functions to find lane lines
    gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
    saturation = get_Saturation(undist_img, thresh=(120,255))
    thresh_lightness = get_Lightness(undist_img, thresh=(40,255))
    lightness = get_Lightness(undist_img)

    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 100))
    # mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(20, 255))
    # dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

    # combine all the thresholds into one image.
    combined = np.zeros_like(saturation)
    # combined[(   (thresh_lightness == 1) & (saturation == 1) |  ((gradx==1) & (grady==1))| ((mag_binary == 1) & (dir_binary == 1)) )] = 1
    combined[(   (thresh_lightness == 1) & (saturation == 1) |  (gradx==1)  )] = 1


    # Transform the Perspective of the conbined image from the current view, to the birds eye view needed to find lane lines well
    img_size = (combined.shape[1], combined.shape[0])
    # offset = 300
    # src values are intended to be parrallel with the lane lines in an image with straight lane lines. These values where gained from playing with what gave the best output
    offset = 320
    src = np.float32([(585,460), (695,460), (1127,img_size[1]), (203,img_size[1])])
    # src = np.float32([(589,457), (698,457), (1145,img_size[1]), (190,img_size[1])])
    dest = np.float32([[offset, 0], [img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]], [offset, img_size[1]]])
    M = cv2.getPerspectiveTransform(src,dest)
    # inverse matrix is also calculated to transform back from the birds eye view to the original view.
    Minv = cv2.getPerspectiveTransform(dest,src)
    warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_NEAREST)

    # find the lane lines in the birds eye view image, left lines and right lines keep track of lane information from frame to frame of a video
    out_img, left_lines, right_lines = findLanes(warped, left_lines, right_lines)

    # draw the average of the found lane lines
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    pts_left = np.array([np.transpose(np.vstack([left_lines.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lines.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image, the lane is green
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Perspective transform from the birds eye view back to the original perspective
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    # print('Histogram Method Curvature Values: Left Lane: {} m , Right Lane: {} m'.format(left_lines.radius_of_curvature, right_lines.radius_of_curvature))
    curvature = (round(left_lines.radius_of_curvature,2) + round(right_lines.radius_of_curvature))*0.5
    text1 = "Avg Curvature of Both Lanes: " + str(curvature) + ' m'
    cv2.putText(result,text1, (430,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    lane_width = 3.7 # Avg Lane width is 3.7 meters
    distfromcenter = round(0.5*(right_lines.line_base_pos-lane_width/2) +  0.5*(abs(left_lines.line_base_pos)-lane_width/2),2)
    text2 = 'Distance from center lane: ' + str(distfromcenter) + ' m'
    cv2.putText(result,text2, (430,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

    # saving images needed if images are used.
    if path !='':
        # Testing Prespective Transforms Output
        line_image = np.copy(undist_img)*0
        undist_warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_NEAREST) # INTER_NEAREST
        cv2.line(line_image,(offset,0),(offset,img_size[1]),(255,0,0),10) 
        cv2.line(line_image,(img_size[0] - offset,0),(img_size[0] - offset,img_size[1]),(0,0,255),10) 
        lines_edges = cv2.addWeighted(undist_warped, 0.8, line_image, 1, 0)

        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_lines.recent_xfitted[-1], ploty, color='yellow')
        plt.plot(right_lines.recent_xfitted[-1], ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title('Histogram Window Sliding')
        plt.savefig('CarND-Advanced-Lane-Lines/output_images/'+ path +'HistogramOutput.jpg')
        plt.close()

        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path +'undistorted_image.jpg',undist_img, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path +'thresholded_image.jpg',combined, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path + 'warped_image.jpg',warped, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path + 'allignment_check.jpg',lines_edges, cmap='gray')
        mpimg.imsave('CarND-Advanced-Lane-Lines/output_images/'+ path + 'Fully_Proccessed_Image.jpg',result, cmap='gray')


    return result, left_lines, right_lines

# Define a class to receive the characteristics of each lane detected
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

        n = 10
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
        img = mpimg.imread(inputImgPath)

        if (idx>=8):
            img = img[:,:,:3]

        left_lines = Line()
        right_lines = Line()

        folder = inputImgName + '/'
        outPath = 'CarND-Advanced-Lane-Lines/output_images/' + folder
        os.makedirs(os.path.dirname(outPath), exist_ok=True)

        result, left_lines, right_lines = process_Image(img, mtx, dist, left_lines, right_lines, path=folder)

        # Now our radius of curvature is in meters
        print('Histogram Method Curvature Values: Left Lane: {} m , Right Lane: {} m'.format(left_lines.radius_of_curvature, right_lines.radius_of_curvature))

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
    
    # testing processing on a short section of the video
    # 20 to 26 is the difficult area of the video
    # clip1 = clip1.subclip(38, 42)

    print('Processing Each Frame of the Video')
    for frame in clip1.iter_frames():

        result, left_lines, right_lines = process_Image(frame, mtx, dist, left_lines, right_lines)
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


# runOnImages(mtx, dist)

runOnVideo(mtx, dist)