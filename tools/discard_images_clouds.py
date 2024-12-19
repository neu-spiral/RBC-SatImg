import os

import numpy as np
from image_reader import ReadSentinel2
from osgeo import gdal
import matplotlib.pyplot as plt

# Configuration
dim_x = 256
dim_y = 256
path_multiearth = "/Users/helena/Documents/Research/Recursive_Bayesian_Image_Classification/MultiEarth2023/Dataset/sent2"
bands_to_plot = ['B2']

# Space Coordinate

# Coord i
#lat_vect = ["-54.70", "-54.60", "-54.60", "-54.58", "-54.58"]
#lon_vect = ["-3.95", "-4.05", "-3.73", "-3.89", "-3.43"]
lat_vect = ["-54.60"]
lon_vect = ["-4.05"]

# ****************************************************************
# PLOT HISTOGRAM
# ****************************************************************
# Space coordinate loop
for coord_idx, _ in enumerate(lat_vect):
    lat = lat_vect[coord_idx]
    lon = lon_vect[coord_idx]

    path_multiearth_coord = os.path.join(path_multiearth, lat + '_' + lon)
    path_label_images = os.path.join(path_multiearth_coord, 'deforestation_labels')
    path_band_folder = os.path.join(path_multiearth_coord, 'B2')

    # Instance of Image Reader object
    image_reader = ReadSentinel2(dim_x, dim_y)

    list_images = os.listdir(path_band_folder)
    list_images.sort()

    sum_pixels = []
    for image_idx_loop, image_date_i in enumerate(list_images):
        # Read image
        path_image_date_i = os.path.join(path_band_folder, image_date_i)
        band = gdal.Open(path_image_date_i).ReadAsArray()
        # Normalize image
        band_normalized = (band - np.min(band))/(np.max(band)-np.min(band))
        # Add up normalized pixels
        sum_pixels.append(np.sum(band_normalized))
        print('image_idx '+str(image_idx_loop)+' sum_pixels '+str(np.sum(band_normalized)))
    print(sum_pixels)
    plt.hist(sum_pixels, bins=100)
    plt.title('Histogram summed pixels normalized images B2 [-54.60, -4.05]')


# ****************************************************************
# SELECT IMAGES WITHOUT TOO MANY CLOUDS
# ****************************************************************
threshold = 12e3
selected_images = []
# Space coordinate loop
for coord_idx, _ in enumerate(lat_vect):
    lat = lat_vect[coord_idx]
    lon = lon_vect[coord_idx]

    path_multiearth_coord = os.path.join(path_multiearth, lat + '_' + lon)
    path_label_images = os.path.join(path_multiearth_coord, 'deforestation_labels')
    path_band_folder = os.path.join(path_multiearth_coord, 'B2')

    # Instance of Image Reader object
    image_reader = ReadSentinel2(dim_x, dim_y)

    list_images = os.listdir(path_band_folder)
    list_images.sort()

    sum_pixels = []
    for image_idx_loop, image_date_i in enumerate(list_images):
        # Read image
        path_image_date_i = os.path.join(path_band_folder, image_date_i)
        band = gdal.Open(path_image_date_i).ReadAsArray()
        # Check if image has too many clouds
        above_threshold, sum_pixels = image_reader.check_clouds(image = band, threshold=threshold, dim_x=dim_x, dim_y=dim_y)
        print(sum_pixels)
        # Check clouds for this image
        if not above_threshold:
            print(path_image_date_i)
            print('Image '+str(image_idx_loop)+' with date '+image_date_i[-15:-5]+' does not have too many clouds')
            selected_images.append(image_idx_loop)
print(selected_images)