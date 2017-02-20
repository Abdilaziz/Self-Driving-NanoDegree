#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



def process_image(image):

	#printing out some stats and plotting
	print('This image is:', type(image), 'with dimesions:', image.shape)

	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 7
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# Filter by color

	low_color_threshold = 180
	high_color_threshold = 255
	color_mask = cv2.inRange(blur_gray,low_color_threshold,high_color_threshold)
	mask_res = cv2.bitwise_and(blur_gray,blur_gray, mask=color_mask)


	# Define our parameters for Canny and apply
	low_threshold = 90
	high_threshold = 150
	edges = cv2.Canny(mask_res, low_threshold, high_threshold)


	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)
	ignore_mask_color = 255

	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)

	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 2 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 15     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 30 # minimum number of pixels making up a line
	max_line_gap = 20    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on a blank image
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

	# Create a "color" binary image to combine with line image
	color_edges = np.dstack((edges, edges, edges)) 

	# Draw the lines on the edge image
	#lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
	lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)


	return lines_edges



i = 0
inputImg=''

while(i<6):
	if(i==0):
		inputImg = 'solidWhiteCurve'
	elif(i==1):
		inputImg = 'solidWhiteRight'
	elif(i==2):
		inputImg = 'solidYellowCurve'
	elif(i==3):
		inputImg = 'solidYellowCurve2'
	elif(i==4):
		inputImg = 'solidYellowLeft'
	elif(i==5):
		inputImg = 'whiteCarLaneSwitch'
	#reading in an image
	inputPath = 'CarND-LaneLines-P1/test_images/'+inputImg+'.jpg'
	outputPath = 'CarND-LaneLines-P1/test_images/'+inputImg + 'Output'+'.jpg'
	image = mpimg.imread(inputPath)
	outputImage = process_image(image)
	mpimg.imsave(outputPath,outputImage)
	i = i +1




# # Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

# white_output = 'white.mp4'
# clip1 = VideoFileClip("CarND-LaneLines-P1/solidWhiteRight.mp4")
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(white_output, audio=False)




# #Create a new video with `moviepy` by processing each frame to [YUV](https://en.wikipedia.org/wiki/YUV) color space.

# new_clip_output = 'test_output.mp4'
# test_clip = VideoFileClip("test.mp4")
# new_clip = test_clip.fl_image(lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2YUV)) #NOTE: this function expects color images!!
# new_clip.write_videofile(new_clip_output, audio=False)



