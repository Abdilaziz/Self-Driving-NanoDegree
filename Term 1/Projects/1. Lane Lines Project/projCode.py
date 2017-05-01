#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



def process_image(image):

	#printing out some stats and plotting
	# print('This image is:', type(image), 'with dimesions:', image.shape)

	gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)


	# Define a kernel size and apply Gaussian smoothing
	kernel_size = 5
	blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# Filter by color

	low_color_threshold = 165
	high_color_threshold = 255
	color_mask = cv2.inRange(blur_gray,low_color_threshold,high_color_threshold)
	color_masked = cv2.bitwise_and(blur_gray,blur_gray, mask=color_mask)



	# Define our parameters for Canny and apply
	low_threshold = 50
	high_threshold = 150
	edges = cv2.Canny(color_masked, low_threshold, high_threshold)
	# Next we'll create a masked edges image using cv2.fillPoly()
	mask = np.zeros_like(edges)
	ignore_mask_color = 255

	# This time we are defining a four sided polygon to mask
	imshape = image.shape
	# vertices = np.array([[(0+130,imshape[0]),(460, 315), (500, 315), (imshape[1]-60,imshape[0])]], dtype=np.int32)
	vertices = np.array([[(0,imshape[0]),(475, 310), (465, 310), (imshape[1],imshape[0])]], dtype=np.int32)

	cv2.fillPoly(mask, vertices, ignore_mask_color)
	masked_edges = cv2.bitwise_and(edges, mask)


	# Define the Hough transform parameters
	# Make a blank the same size as our image to draw on
	rho = 2 # distance resolution in pixels of the Hough grid
	theta = np.pi/180 # angular resolution in radians of the Hough grid
	threshold = 25     # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 3 # minimum number of pixels making up a line
	max_line_gap = 5    # maximum gap in pixels between connectable line segments
	line_image = np.copy(image)*0 # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
	                            min_line_length, max_line_gap)

	# Iterate over the output "lines" and draw lines on a blank image
	leftSlopeTotal = 0
	rightSlopeTotal = 0

	leftYintTotal = 0
	rightYintTotal = 0

	leftAmount = 0
	rightAmount = 0

	for line in lines:
	    for x1,y1,x2,y2 in line:

	    	if((x2-x1)==0):
	    		print('One of the values divides by 0')
	    	else:
	    		curSlope = (y2-y1)/(x2-x1)
	    		curYint1 = -1*curSlope*x1 + y1
	    		curYint2 = -1*curSlope*x2 + y2
		    	if(curSlope<=-0.5):
		    		# Left Side
		    		leftSlopeTotal = leftSlopeTotal + curSlope
		    		leftYintTotal = leftYintTotal + curYint1
		    		leftAmount = leftAmount + 1
		    		
		    	elif(curSlope>=0.5):
		    		# Right Side
		    		rightSlopeTotal = rightSlopeTotal + curSlope
		    		rightYintTotal = rightYintTotal + curYint1
		    		rightAmount= rightAmount + 1

	avgLeftSlope =leftSlopeTotal/leftAmount
	avgRightSlope = rightSlopeTotal/rightAmount
	avgLeftYint = leftYintTotal/leftAmount
	avgRightYint = rightYintTotal/rightAmount

	leftYpoint1 = imshape[0]
	leftXpoint1 = int((leftYpoint1-avgLeftYint)/avgLeftSlope)
	print(avgLeftSlope)

	leftYpoint2 = 310
	leftXpoint2 = int((leftYpoint2-avgLeftYint)/avgLeftSlope)
	print(avgRightSlope)

	cv2.line(line_image,(leftXpoint1,leftYpoint1),(leftXpoint2,leftYpoint2),(255,0,0),10) 

	rightYpoint1 = imshape[0]
	rightXpoint1 = int((rightYpoint1-avgRightYint)/avgRightSlope)

	rightYpoint2 = 310
	rightXpoint2 = int((rightYpoint2-avgRightYint)/avgRightSlope)
	cv2.line(line_image,(rightXpoint1,rightYpoint1),(rightXpoint2,rightYpoint2),(0,0,255),10) 


	# Create a "color" binary image to combine with line image
	# color_edges = np.dstack((masked_edges, masked_edges, masked_edges)) 

	# Draw the lines on the edge image
	lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
	# lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)


	return lines_edges



# i = 0
# inputImg=''

# while(i<6):
# 	if(i==0):
# 		inputImg = 'solidWhiteCurve'
# 	elif(i==1):
# 		inputImg = 'solidWhiteRight'
# 	elif(i==2):
# 		inputImg = 'solidYellowCurve'
# 	elif(i==3):
# 		inputImg = 'solidYellowCurve2'
# 	elif(i==4):
# 		inputImg = 'solidYellowLeft'
# 	elif(i==5):
# 		inputImg = 'whiteCarLaneSwitch'
# 	#reading in an image
# 	inputPath = 'CarND-LaneLines-P1/test_images/'+inputImg+'.jpg'
# 	outputPath = 'CarND-LaneLines-P1/test_images/'+inputImg + 'Output2'+'.jpg'
# 	image = mpimg.imread(inputPath)
# 	outputImage = process_image(image)
# 	mpimg.imsave(outputPath,outputImage)
# 	i = i +1


inputPath = 'CarND-LaneLines-P1/test_images/'+inputImg+'.jpg'
outputPath = 'CarND-LaneLines-P1/test_images/'+inputImg + 'Output2'+'.jpg'
image = mpimg.imread(inputPath)
outputImage = process_image(image)
mpimg.imsave(outputPath,outputImage)
i = i +1



# Import everything needed to edit/save/watch video clips
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML

# inputVideo = 'solidYellowLeft'
# video_output = inputVideo + 'output.mp4'
# inputVideoPath = "CarND-LaneLines-P1/"+inputVideo+".mp4"
# clip1 = VideoFileClip(inputVideoPath)
# white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# white_clip.write_videofile(video_output, audio=False)
