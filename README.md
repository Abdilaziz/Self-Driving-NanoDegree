# Self-Driving-NanoDegree

## Objective:
This Repo is to track all the materials acquired throughout
the Self-Driving-NanoDegree Program on Udacity.

## Projects Included:
##1) Lane Lines Project

Finding the Lane Lines in Images and Videos by thresholding color values and choosing a region of interest. 
Then using Canny Edge Detection and Hough Transform to recognize lines in the image. 

Output: Video with detected lane lines drawn on each frame.
Used: OpenCV (Python)

##2) Traffic Sign Classifier Project

Used a Convolutional Neural Network to train a classifier that can classify 43 different traffic signs.
Implemented the LeNet Architecture.
Used Dropout to handle Overfitting.

Output: Softmax of 5 test images
Used: Tensorflow (Python)

##3) Behavioural Cloning Project

Used Udacity's Vehicle Simulator to aquire data correlating Images from 3 cameras on a car with steering angle measurements.
Trained the Nvidia Self-Driving Vehicle Architecture (5 Convolutional Layers followed by 4 Fully Connected Layers). 
Ran the trained model on the simulator that can autonomously drive on the track.

Output: Video of vehicle autonomously driving around a track.
Used: Keras (Python)

##4) Advanced Lane Finding Project

Calibrated Camera Images and performed a Perspective transform to find Lane Lines.
Calculate the Lanes Curvature and fit a second order polynomial to each lane.
Tracked detected lane measurements for each frame of a video to smoothen the detected lane lines.

Output: Video displaying lane lanes found, with measurements of the vehicles position from the center of the lane, and the curvature of the lane.
Used: OpenCV (Python)

##5) Vehicle Detection and Tracking

Find vehicles in Images using feature extraction (spatial binning, histogram of Colors, and Histogram of Oriented Gradients), and training a Support Vector Machine to classify whether or not there is a vehicle in the image.

Use Sliding windows to detect vehciles in a full image. Windows where a vehicle is detected with a good level of confidance is drawn onto the image.

False positives and multiple detection for the same vehicle are corrected by using a Heatmap.

Output: Video of vehicle detections being drawn on the image, with their position values displayed.
Used: OpenCV, Sklearn. (Python)

