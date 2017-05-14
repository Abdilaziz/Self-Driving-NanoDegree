# Vehicle Detection: Feature Extraction -> Training Classifier (Linear SVM) -> Make the classification work on full images (Sliding Window) -> Filter false positives (Heat Mapping)

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

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    # transform_sqrt is for gamma normalization. It helps reduce the effect of shadows and other illumination
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


# EXTRACT FEATURES FOR THE ENTIRE DATASET

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
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

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
    model_score = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', model_score)
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
    dist_pickle['cspace'] = cspace
    dist_pickle['orient'] = orient
    dist_pickle['pix_per_cell'] = pix_per_cell
    dist_pickle['cell_per_block'] = cell_per_block
    dist_pickle['hog_channel'] = hog_channel
    dist_pickle['spatial_size'] = spatial_size
    dist_pickle['hist_bins'] = hist_bins
    dist_pickle['hist_range'] = hist_range

    dist_pickle['model_score'] = model_score
    dist_pickle['data_info'] = data_info

    pickle.dump( dist_pickle, open(pickleName, "wb" ))

    return svc, X_scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, model_score, data_info


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


# THIS BOTTOM PORTION IS FOR THE HOG SUBSAMPLINGS

def convert_color(image, cspace):

    if cspace != 'RGB':
        if cspace == 'HSV':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            cvt_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: cvt_image = np.copy(image)
    return cvt_image

# Algorithm for finding HOG features once for the whole image rather than per cell, and finding the other features per cell and classifying it rather than


def detect_vehicles2(img, ystart, ystop, scale, classifier, scaler, cspace, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_size=(32, 32), hist_bins=32, hist_range= (0,256)):
    

    windows = []

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    ctrans_tosearch = convert_color(img_tosearch, cspace)

    # scales the whole image down making the search window larger
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    # with a classification window being a 64x64 image, each window has (64/8) = 8 cells per row. Meaning 8-2+1 = 7 blocks per row
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
 
    if hog_channel == 0:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 1:
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    elif hog_channel == 2:
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    else:
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            # Extract HOG for this patch
            if hog_channel == 0:
                hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            elif hog_channel == 1:
                hog_features = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            elif hog_channel == 2:
                hog_features = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            else:
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell


            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                


            # xbox_left = np.int(xleft*scale)
            # ytop_draw = np.int(ytop*scale)
            # win_draw = np.int(window*scale)
            # windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))




            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=hist_range)

            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = classifier.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    
    # print('Windows Searched is : ' , (nxsteps*nysteps))
    # print('Number of cars found is: ', len(windows))
    return windows


from scipy.ndimage.measurements import label
def heatMap_Detections(image, box_list, heatMaps=None):

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    def add_heat(heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap


    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    # Apply threshold to help remove false positives
    # heat = apply_threshold(heat, 0) # 3
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    if (heatMaps != None):
        heatMaps.updateHeatMap(heatmap)
        heatmap = heatMaps.avg_heatmap

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, 3)

    # plt.imshow(heatmap, cmap='hot')
    # plt.show()


    return heatmap, heatMaps

def draw_labeled_bboxes(img, labels, heatMaps=None):
    # Iterate through all detected cars
    bboxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox)

            
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)

    if (heatMaps!=None):
        heatMaps.updateCentroid(bboxes, labels[1], img)

    # Return the image
    return img, heatMaps

def draw_detections2(image, classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, plot=False, heatMaps=None):

    ystart = 400 # 420
    ystop = 656 # 548 
    scale = 2 # 1

    # we can detect cars with multiple scaled windows by running with multiple scale values
    # the smaller the scale, the larger the searched image, meaning more windows are searched, meaning the detection window in the original image is smaller

    detectedWindows = detect_vehicles2(image, ystart, ystop, scale, classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

    ystart = 400
    ystop = 550
    scale = 1.5 # 96
    detectedWindows1 = detect_vehicles2(image, ystart, ystop, scale, classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

    ystart = 400
    ystop = 500
    scale = 1
    detectedWindows2 = detect_vehicles2(image, ystart, ystop, scale, classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)


    detectedWindows.extend(detectedWindows1)

    detectedWindows.extend(detectedWindows2)

    # Itegrate the heatMap from frame to frame so areas get cool and hot over time.
    heatmap, heatMaps = heatMap_Detections(image, detectedWindows, heatMaps)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    Final_Image, heatMaps = draw_labeled_bboxes(np.copy(image), labels, heatMaps)
    n_cars = labels[1]
    text1 = 'The Number of cars detected is: ' + str(n_cars)
    cv2.putText(Final_Image, text1, (430,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)


    if plot == True:
        plt.imshow(Final_Image)
        plt.show()
    return Final_Image, heatMaps


def run_Image2(classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, heatMaps=None, plot=False):
    
    test_images = glob.glob('CarND-Vehicle-Detection-master/test_images/*.jpg')
    results = []

    for fname in test_images:
        image = mpimg.imread(fname)

        image = image[:,:,:3]

        print(fname.split('\\')[-1])
        t = time.time()
        result,_ = draw_detections2(image, classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, heatMaps=heatMaps, plot=plot)
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict one whole image Image with HOG window Subsampling')
        print('')
        results.append(result)
    for i in range(len(test_images)):
        if (plot==False):
            out =  'CarND-Vehicle-Detection-master/output_images/' + (test_images[i].split('\\')[-1]).split('.')[0] + '_OutputImage.jpg'
            print(out)
            mpimg.imsave(out,results[i])

def run_Video(classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range):

    from moviepy.editor import VideoFileClip, ImageSequenceClip

    inputVideo = 'project_video'
    video_output = inputVideo + 'output.mp4'
    inputVideoPath = "CarND-Vehicle-Detection-master/" + inputVideo + ".mp4"
    clip1 = VideoFileClip(inputVideoPath)

    new_frames = []


    heatMaps = HeatMapTracker()
    
    # testing processing on a short section of the video
    # clip1 = clip1.subclip(5, 15)


    print('Processing Each Frame of the Video')
    for frame in clip1.iter_frames():

        result, heatMaps = draw_detections2(frame, classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, heatMaps=heatMaps)
        text = 'The centroids for the vehicles is ' + str(heatMaps.avg_centroid)
        cv2.putText(result, text, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)

        new_frames.append(result)

    new_clip = ImageSequenceClip(new_frames, fps=clip1.fps)
    new_clip.write_videofile(video_output)


class HeatMapTracker():
    def __init__(self):

        self.recent_heatmap = None
        self.recent_n_heatmaps = []

        self.recent_centroids = [] # recent centroids. Array length is the number of cars detected in the image
        self.recent_n_centroids = [] # a 2D array with recent centroid values of each car

        self.avg_heatmap = None
        self.avg_centroid = []

        self.numb_totalCars = 0

    def getArrayPlacement(avg_centroid, new_centroid):
        pixel_window = 50

        curPos = 0
        for cur_centroid in avg_centroid:
            lower_centroid_thresh = [cur_centroid[0] - pixel_window , cur_centroid[1] - pixel_window]
            upper_centroid_thresh = [cur_centroid[0] + pixel_window , cur_centroid[1] + pixel_window]

            if (lower_centroid_thresh[0] <= new_centroid[0]) & (lower_centroid_thresh[1] < new_centroid[1]) & (upper_centroid_thresh[0] > new_centroid[0]) & (upper_centroid_thresh[1] > new_centroid[1]):
                return curPos
            curPos += 1

        return None

    def updateHeatMap(self, heatmap):

        n = 25

        self.recent_heatmap = heatmap
        self.recent_n_heatmaps.append(heatmap)
        if ( len(self.recent_n_heatmaps) > n ):
            self.recent_n_heatmaps = self.recent_n_heatmaps[1:]

        self.avg_heatmap = np.mean(self.recent_n_heatmaps, axis=0)

    def updateCentroid(self, bboxes, total_cars, image):

        n = 20

        self.numb_totalCars = total_cars

        for bbox in bboxes:
            # centroid is (xmin + (xmax - xmin)/2), (ymin + (ymax - ymin)/2)
            centroid = [ int(bbox[0][0] + (bbox[1][0] - bbox[0][0])/2) , int(bbox[0][1] +  (bbox[1][1] - bbox[0][1])/2)]


            arrayPos = HeatMapTracker.getArrayPlacement(self.avg_centroid, centroid)

            if arrayPos == None:
                arrayPos = len(self.recent_centroids)          
                self.recent_centroids.append(centroid)
                self.recent_n_centroids.append([centroid])
                self.avg_centroid.append(centroid)
                # print('Length of incresed size centroid array: ', len(self.avg_centroid))
            else:
                self.recent_centroids[arrayPos] = centroid
                self.recent_n_centroids[arrayPos].append(centroid)
                if (len(self.recent_n_centroids[arrayPos]) > n):
                    self.recent_n_centroids[arrayPos] = self.recent_n_centroids[arrayPos][1:]

                avg = []
                for x in zip(*(self.recent_n_centroids[arrayPos])):
                    avg.append( int( sum(x) / len(x)) )
                
                self.avg_centroid[arrayPos] = avg


            # print(arrayPos, ' position of centroid is ', self.avg_centroid[arrayPos])
            HeatMapTracker.drawNumber(image, self.avg_centroid[arrayPos],  arrayPos)
        if (len(self.avg_centroid) > total_cars):
            for numb_to_remove in range( (len(self.avg_centroid) - total_cars) ):
                # print('This is check number: ', numb_to_remove)
                # a car has left (remove from array) I am removing all the data for now
                removePos = HeatMapTracker.removeVehicle(self.avg_centroid,bboxes)
                if (removePos != None):
                    del self.recent_n_centroids[removePos]
                    del self.recent_centroids[removePos]
                    del self.avg_centroid[removePos]
        # print('')

    def removeVehicle(avg_centroid, bboxes):
        curPos = 0
        # print('Need to remove centroid values for cars not detected')

        for cur_centroid in avg_centroid:
            pixel_window = 30
            lower_centroid_thresh = [cur_centroid[0] - pixel_window , cur_centroid[1] - pixel_window]
            upper_centroid_thresh = [cur_centroid[0] + pixel_window , cur_centroid[1] + pixel_window]

            found = False

            for bbox in bboxes:
                new_centroid = [ int(bbox[0][0] + (bbox[1][0] - bbox[0][0])/2) , int(bbox[0][1] +  (bbox[1][1] - bbox[0][1])/2)]
                
                # print('Lower Thresh: ', lower_centroid_thresh)
                # print('Upper Thresh: ', upper_centroid_thresh)
                # print('New Centroid: ', new_centroid)

                if  ((lower_centroid_thresh[0] < new_centroid[0]) & (lower_centroid_thresh[1] < new_centroid[1]) & (upper_centroid_thresh[0] > new_centroid[0]) & (upper_centroid_thresh[1] > new_centroid[1])):
                    # print('Found the current box in the avg')
                    found = True
            if found != True:
                # print('Removing:  ',curPos)
                return curPos
            curPos += 1
        return None


    def drawNumber(image, centroid ,arrayPos):
        pos  = centroid
        text = str(arrayPos)
        cv2.putText(image, text, (pos[0],pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)
        return image



VehicleDetectionPickle = 'VehicleDetection.p'

print('')
# Initialize Values used for feature extraction
# Be sure to delete the current pickle file in order to reconfigure the trained classifier.

#Spatial Binning
new_spatial_size = (32, 32)

# Histogram of Color
new_hist_bins = 32
new_hist_range = (0, 256)

# Histogram of Orientations Parameters
new_colorspace = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
new_orient = 9
new_pix_per_cell = 8
new_cell_per_block = 2
new_hog_channel = 0 # Can be 0, 1, 2, or "ALL"


try:
    dist_pickle = pickle.load(open(VehicleDetectionPickle, "rb"))
    classifier = dist_pickle['classifier']
    scaler = dist_pickle['scaler']

    cspace = dist_pickle['cspace'] # color space of image used to get all 3 features
    orient = dist_pickle['orient'] # number of orientation bins that are used in the histogram of oriented gradients
    pix_per_cell = dist_pickle['pix_per_cell'] # number of pixels per cell
    cell_per_block = dist_pickle['cell_per_block'] # number of cells included per block that is locally normalized
    hog_channel = dist_pickle['hog_channel'] # Number the channels used when calculating the histogram of oriented gradients
    spatial_size = dist_pickle['spatial_size'] # spatial size that is a feature
    hist_bins = dist_pickle['hist_bins'] # From Histogram of Colors
    hist_range = dist_pickle['hist_range'] # range of pixel values considered for in the histogram of colors
    

    model_score = dist_pickle['model_score']
    data_info = dist_pickle['data_info']


    if (cspace != new_colorspace) | (orient!=new_orient) | (pix_per_cell != new_pix_per_cell) | (cell_per_block != new_cell_per_block) | (hog_channel != new_hog_channel) | (spatial_size != new_spatial_size) | (hist_bins != new_hist_bins) | (hist_range != new_hist_range):
        print('New Parameters have been choosen. ')
        print(' Now re-training the classifier')
        print('')

        classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, model_score, data_info = train_Classifier(VehicleDetectionPickle, cspace=new_colorspace, orient=new_orient, pix_per_cell=new_pix_per_cell, cell_per_block=new_cell_per_block, hog_channel=new_hog_channel, spatial_size=new_spatial_size, hist_bins=new_hist_bins, hist_range=new_hist_range)

    print('Using {}'.format(VehicleDetectionPickle))
except (OSError, IOError) as e:
    print("{} doesn't exist. Training the Classifier Now".format(VehicleDetectionPickle))
    classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, model_score, data_info = train_Classifier(VehicleDetectionPickle, cspace=new_colorspace, orient=new_orient, pix_per_cell=new_pix_per_cell, cell_per_block=new_cell_per_block, hog_channel=new_hog_channel, spatial_size=new_spatial_size, hist_bins=new_hist_bins, hist_range=new_hist_range)


print('Your Training Data has a count of', 
      data_info["n_cars"], ' cars and', 
      data_info["n_notcars"], ' non-cars')
print('of size: ',data_info["image_shape"], ' and data type:', 
      data_info["data_type"])
print('The Models accuracy is', model_score)


print('')
print('Using:', cspace,'as the colorspace,',  orient,'orientations,',pix_per_cell,
    'pixels per cell and,', cell_per_block,'cells per block,', hog_channel, 'as the hog_channel,', spatial_size, 'as the spatial_size,', 
    hist_bins, 'as the hist_bins, and', hist_range, 'as the hist_range')
print('')


# run_Image2(classifier,scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel,spatial_size, hist_bins, hist_range, plot=False)


run_Video(classifier, scaler, cspace, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range)

