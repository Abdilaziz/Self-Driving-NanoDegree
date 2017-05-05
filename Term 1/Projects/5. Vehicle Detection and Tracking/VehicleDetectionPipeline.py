#!/usr/bin/env python -W ignore::DeprecationWarning


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
import cv2
import time
import glob
import pickle

# Define a function to return some characteristics of the dataset 
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] =  len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    test_image = cv2.imread(car_list[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = test_image.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = test_image.dtype
    # Return data_dict
    return data_dict

def plot3d(pixels, colors_rgb, axis_labels=list("RGB"), axis_limits=[(0, 255), (0, 255), (0, 255)]):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    # ravel turns the matrix into a 1 dimensional vector
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')


    return ax  # return Axes3D object for further manipulation

def plotColorSpaces(image, shape, imgCategory):
	h, w, d = shape

	# Select a small fraction of pixels to plot by subsampling it
	scale = max(h, w, 64) / 64  # at most 64 rows and columns
	img_small = cv2.resize(image, (np.int(h / scale), np.int(w / scale)), interpolation=cv2.INTER_NEAREST)

	# Convert subsampled image to desired color space(s)
	img_small_RGB = img_small # cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
	img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
	img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2LUV)
	img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting


	# Plot and show
	plot3d(img_small_RGB, img_small_rgb)
	plt.savefig(imgCategory +'RGB.jpg')

	plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
	plt.savefig(imgCategory + 'HSV.jpg')

	plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"))
	plt.savefig(imgCategory+ 'LUV.jpg')

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), feature_vec=True):
    # Create a list to append feature vectors to
    features = []
    histImg = 1
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=feature_vec))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=feature_vec)

        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors
    return features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, cspace='RGB', spatial_size=(32, 32), hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, hist_range=(0, 256),spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

def normalize_Scale_Features(features, scaler, reshape = False):
    if reshape == True:
        features = scaler.transform(np.array(features).reshape(1, -1))
    # Apply the scaler to X
    scaled_X = scaler.transform(features)
    return scaled_X

def train_Classifier(pickleName, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256), feature_vec=True):

    # Read all the images downloaded
    notcars = glob.glob('Datasets/non-vehicles/GTI/*.png')
    notcars.extend(glob.glob('Datasets/non-vehicles/Extras/*.png'))

    cars = glob.glob('Datasets/vehicles/GTI_Far/*.png')
    cars.extend(glob.glob('Datasets/vehicles/GTI_Left/*.png'))
    cars.extend(glob.glob('Datasets/vehicles/GTI_MiddleClose/*.png'))
    cars.extend(glob.glob('Datasets/vehicles/GTI_Right/*.png'))
    cars.extend(glob.glob('Datasets/vehicles/GTI_Left/*.png'))
    cars.extend(glob.glob('Datasets/vehicles/KITTI_extracted/*.png'))



    data_info = data_look(cars, notcars)



    print('Your function returned a count of', 
          data_info["n_cars"], ' cars and', 
          data_info["n_notcars"], ' non-cars')
    print('of size: ',data_info["image_shape"], ' and data type:', 
          data_info["data_type"])



    t = time.time()


    # Extract all features that will be used by the classifier.
    car_features = extract_features(cars, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, feature_vec=feature_vec)
    notcar_features = extract_features(notcars, cspace=cspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, feature_vec=feature_vec)


    # car_features, car_hog_image = get_hog_features(car_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    # notcar_features, notcar_hog_image = get_hog_features(notcar_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)

    # plt.imshow(car_hog_image,cmap='gray')
    # plt.show()

    # plt.imshow(notcar_hog_image,cmap='gray')
    # plt.show()


    t2 = time.time()
    #Generally (RGB Colorspace) 48.31s to Extract All Features for all images.
    # LUV Colorspace is 52.8s
    print(round(t2-t, 2), 'Seconds to extract All features...')

    # Training the Classifier

    # Normalize Data and turn the features vector into an array of each feature stacked
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)


    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))


    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    print('Saving Model in '+pickleName)


    dist_pickle = {}

    dist_pickle['scaler'] = X_scaler
    dist_pickle['classifier'] = svc

    pickle.dump( dist_pickle, open(pickleName, "wb" ))

    return svc, X_scaler

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def detect_vehicles(image, classifier, scaler):

    # Notes: when y is roughly 400px, scale should be roughly 40x40 or smaller.
    # have the multi scale windows overlapping due to depth of vehciles in lanes isn't only apparent through a vertical pixel measurement

    # window_scales = 5
    windows = []

    small_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 450], xy_window=(32, 32), xy_overlap=(0.25, 0.25))

    medium_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[430, 500], xy_window=(64, 64), xy_overlap=(0.25, 0.25))

    large_windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None], xy_window=(128, 128), xy_overlap=(0.25, 0.25))

    windows.extend(small_windows)
    windows.extend(medium_windows)
    windows.extend(large_windows)

    detectedWindows = []
    # t = time.time()
    for window in windows:

        #get image of the window
        window_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # # Extract all features that will be used by the classifier at every window.
        features = single_img_features(window_img, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
        
        normalized_features = normalize_Scale_Features(features, scaler=scaler, reshape= True)
        prediction = classifier.predict(normalized_features)
        if prediction[0] == 1.:
            detectedWindows.append(window)


    # t2 = time.time()
    # print('The number of windows searched is ', len(windows))
    # print('The number of windows with vehicles detected is ', len(detectedWindows))
    # print(round(t2-t, 5), 'Seconds to predict one whole image Image')

    return detectedWindows

def draw_detections(image,classifier,scaler, plot=False):

    detectedWindows = detect_vehicles(image, classifier, scaler)

    window_img = draw_boxes(image, detectedWindows, color=(0, 0, 255), thick=6)

    if plot == True:
        plt.imshow(window_img)
        plt.show()
    return window_img

def run_Image(classifier, scaler):
    image = mpimg.imread('CarND-Vehicle-Detection-master/test_images/test5.jpg')

    draw_detections(image, classifier, scaler, plot=True)


def run_Video(classifier, scaler):

    from moviepy.editor import VideoFileClip, ImageSequenceClip

    inputVideo = 'test_video'
    video_output = inputVideo + 'output.mp4'
    inputVideoPath = "CarND-Vehicle-Detection-master/"+inputVideo+".mp4"
    clip1 = VideoFileClip(inputVideoPath)

    new_frames = []
    
    # testing processing on a short section of the video
    # clip1 = clip1.subclip(38, 42)

    print('Processing Each Frame of the Video')
    for frame in clip1.iter_frames():

        result = draw_detections(frame, classifier, scaler)

        new_frames.append(result)



    new_clip = ImageSequenceClip(new_frames, fps=clip1.fps)
    new_clip.write_videofile(video_output)


# Keep track of windows from frame to frame
# Dont Draw boxes of cars if they are not consistant from frame to frame to filter out false positives
# combine multiple boxes around a vehicle around its centroid to have one bounding box per vehicle.


# class Vehicle():
#     def __init__(self):
        

#         self.latest_windows = []
#         self.latest_window = None 

#         self.filtered_windows = 

    

#     def update(windows):
#         self.latest_window = windows
#         n= 10
#         self.latest_windows = self.latest_windows.append(self.latest_window)
#         if (len(self.latest_windows) > n):
#             self.latest_windows = self.latest_windows[1:]



VehicleDetectionPickle = 'VehicleDetection.p'

print('')
# Initialize Values used for feature extraction
# Be sure to delete the current pickle file in order to reconfigure the trained classifier.

#Spatial Binning
spatial_size = (32, 32)

# Histogram of Color
hist_bins = 32
hist_range = (0, 256)

# Histogram of Orientations Parameters
colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"


try:
    dist_pickle = pickle.load(open(VehicleDetectionPickle, "rb"))
    classifier = dist_pickle['classifier']
    scaler = dist_pickle['scaler']
    print('Using {}'.format(VehicleDetectionPickle))
except (OSError, IOError) as e:
    print("{} doesn't exist. Training the Classifier Now".format(VehicleDetectionPickle))
    classifier, scaler= train_Classifier(VehicleDetectionPickle, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)


run_Video(classifier, scaler)


# images = glob.glob('VehiclesAndNonVehicles/*.png')


# t = time.time()
# # Extract all features that will be used by the classifier.
# features = extract_features(images, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
# normalized_features = normalize_Scale_Features(features, scaler=scaler, reshape= False)
# predictions = classifier.predict(normalized_features)
# t2 = time.time()


# n_predict = len(images)

# print(round(t2-t, 5), 'Seconds to predict', n_predict, 'Images')
# print('My SVC predicts: ', predictions)
# print('For these',n_predict, 'labels: ', [image.split('\\')[-1] for image in images])







