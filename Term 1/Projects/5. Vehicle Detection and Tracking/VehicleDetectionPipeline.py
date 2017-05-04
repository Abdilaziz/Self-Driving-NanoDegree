import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
import numpy as np
import cv2
import time
import glob

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

# Just for fun choose random car / not-car indices and plot example images   
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))
    
# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])




#Spatial Binning
spatial_size = (32, 32)

# Histogram of Color
hist_bins = 32
hist_range = (0, 256)
feature_vec = True

# Histogram of Orientations Parameters
colorspace = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"

t = time.time()


# Extract all features that will be used by the classifier.
car_features = extract_features(cars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, feature_vec=feature_vec)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range, feature_vec=feature_vec)


# car_features, car_hog_image = get_hog_features(car_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)
# notcar_features, notcar_hog_image = get_hog_features(notcar_image[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=True)

# plt.imshow(car_hog_image,cmap='gray')
# plt.show()

# plt.imshow(notcar_hog_image,cmap='gray')
# plt.show()




t2 = time.time()
#Generally (RGB Colorspace) 48.31s to Extract HOG Features for all images.
# LUV Colorspace is 52.8s
print(round(t2-t, 2), 'Seconds to extract HOG features...')

print( ( len(car_features)*len(car_features[0]) ) )


# Training the Classifier










