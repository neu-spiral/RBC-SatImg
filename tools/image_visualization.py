import os

import numpy as np
from image_reader import ReadSentinel2
from osgeo import gdal
import matplotlib.pyplot as plt

# Configuration
dim_x = 256
dim_y = 256
path_multiearth = "/Users/helena/Documents/Research/Recursive_Bayesian_Image_Classification/MultiEarth2023/Dataset/sent2"
#num_images_to_plot = 50
bands_to_plot = ['B2', 'B3', 'B4', 'B8A', 'B8', 'B11', 'QA60']
images_to_plot = np.arange(0,220)
label_images_to_plot = [40, 46, 107, 118, 172, 189]
label_images_to_plot = [41, 46, 108, 118, 173, 189] # changes for clouds - threshold 12e3 manual
offset_labels = 5
num_bands_to_plot = len(bands_to_plot)
num_images_to_plot = len(images_to_plot)
"""
# Space Coordinates
# Coord i
lat_vect = ["-54.70", "-54.60", "-54.60", "-54.58", "-54.58"]
lon_vect = ["-3.95", "-4.05", "-3.73", "-3.89", "-3.43"]

# Create image
fig, axs = plt.subplots(num_images_to_plot, num_bands_to_plot+1, figsize=(5,100))

"""
# Configuration to plot selected images
# After discarding images with too many clouds
images_to_plot = [0, 3, 12, 19, 21, 24, 25, 29, 32, 41, 42, 50, 51, 52, 54, 58, 63, 64, 65, 71, 90, 91, 92, 93,
                  108, 109, 113, 114, 117, 119, 124, 125, 126, 127, 128, 130, 131, 132, 133, 135, 142, 144, 145,
                  151, 173, 177, 178, 183, 185, 187, 190, 197, 202, 204, 218] # threshold 10e3
images_to_plot = [0, 3, 12, 17, 19, 21, 24, 25, 29, 32, 35, 41, 42, 45, 46, 50, 51, 52, 54, 58, 61, 63, 64, 65,
                  67, 71, 90, 91, 92, 93, 94, 108, 109, 111, 113, 114, 117, 118, 119, 120, 121, 122, 124, 125,
                  126, 127, 128, 130, 131, 132, 133, 135, 142, 144, 145, 146, 147, 150, 151, 154, 173, 177, 178,
                  183, 185, 187, 190, 197, 202, 203, 204, 218] # threshold 12e3
images_to_plot = [0, 3, 12, 17, 19, 21, 24, 25, 29, 32, 35, 41, 42, 45, 46, 48, 50, 51, 52, 54, 58, 61, 63, 64, 65,
                  67, 71, 90, 91, 92, 93, 94, 108, 109, 111, 113, 114, 117, 118, 119, 120, 121, 122, 124, 125,
                  126, 127, 128, 130, 131, 132, 133, 135, 142, 144, 145, 146, 147, 150, 151, 154, 173, 177, 178,
                  183, 185, 187, 188, 189, 190, 193, 197, 202, 203, 204, 218] # threshold 12e3 + manual additions
num_images_to_plot = len(images_to_plot)
# Selecting a specific coordinate
lat_vect = ["-54.60"]
lon_vect = ["-4.05"]
# Create figure specific size
fig, axs = plt.subplots(num_images_to_plot, num_bands_to_plot+1, figsize=(5,50))


# Space coordinate loop
for coord_idx, _ in enumerate(lat_vect):
    lat = lat_vect[coord_idx]
    lon = lon_vect[coord_idx]

    path_multiearth_coord = os.path.join(path_multiearth, lat + '_' + lon)
    path_label_images = os.path.join(path_multiearth_coord, 'deforestation_labels')

    # Instance of Image Reader object
    image_reader = ReadSentinel2(dim_x, dim_y)

    # Plot Satellite Images
    # Iterate folders (bands B1 to B12 and cloud mask QA60)
    band_idx = 0
    band_list = os.listdir(path_multiearth_coord)
    band_list.sort()
    for band_folder in band_list:
        path_band_folder = os.path.join(path_multiearth_coord, band_folder)
        #if os.path.isdir(path_band_folder):
        if band_folder in bands_to_plot:
            print('Showing images from band ' + band_folder + ' for lat ' + lat + ' and lon ' + lon)
            image_idx = 0
            list_images = os.listdir(path_band_folder)
            list_images.sort()
            for image_idx_loop, image_date_i in enumerate(list_images):
                if image_idx_loop in images_to_plot:
                    path_image_date_i = os.path.join(path_band_folder, image_date_i)
                    band = gdal.Open(path_image_date_i).ReadAsArray()
                    axs[image_idx, band_idx].imshow(band)
                    axs[image_idx, band_idx].get_yaxis().set_ticks([])
                    axs[image_idx, band_idx].get_xaxis().set_ticks([])
                    for axis in ['top', 'bottom', 'left', 'right']:
                        axs[image_idx, band_idx].spines[axis].set_linewidth(0)
                    if image_idx == 0:
                        # Title band
                        axs[image_idx, band_idx].title.set_text(band_folder)

                    if band_idx == 0:
                        # Title Image
                        axs[image_idx, band_idx].set_ylabel('Image ' + str(image_idx_loop)+"\n"+image_date_i[-15:-5], rotation=0, fontsize=8, fontfamily='Times New Roman')
                        axs[image_idx, band_idx].yaxis.set_label_coords(-1.5, 0.05)
                    """
                    # Title Image
                    axs[image_idx, band_idx].set_ylabel('Image ' + str(image_idx)+"\n"+image_date_i[-15:-5], rotation=0, fontsize=8, fontfamily='Times New Roman')
                    axs[image_idx, band_idx].yaxis.set_label_coords(-1.5, 0.05)
                    """
                    image_idx = image_idx + 1
            band_idx = band_idx + 1

    # Plot Ground-Truth Images
    axs[0, band_idx].title.set_text('Label')
    list_label_images = os.listdir(path_label_images)
    list_label_images.sort()
    if '.DS_Store' in list_label_images:
        list_label_images.remove('.DS_Store')
    label_images_plotted = 0
    for idx_loop, date_idx in enumerate(images_to_plot):
        if date_idx in label_images_to_plot:
            path_label_image = os.path.join(path_label_images, list_label_images[label_images_plotted+offset_labels])
            label_image = gdal.Open(path_label_image).ReadAsArray()
            axs[idx_loop, band_idx].imshow(label_image)
            axs[idx_loop, band_idx].get_yaxis().set_ticks([])
            axs[idx_loop, band_idx].get_xaxis().set_ticks([])
            for axis in ['top', 'bottom', 'left', 'right']:
                axs[idx_loop, band_idx].spines[axis].set_linewidth(0)
            axs[idx_loop, band_idx].set_ylabel('Label image ' + str(label_images_plotted) + "\n" + list_label_images[label_images_plotted+offset_labels][-15:-5], rotation=0,
                                                fontsize=8, fontfamily='Times New Roman')
            axs[idx_loop, band_idx].yaxis.set_label_coords(2.5, 0.05)
            label_images_plotted = label_images_plotted + 1
        else:
            axs[idx_loop, band_idx].get_yaxis().set_ticks([])
            axs[idx_loop, band_idx].get_xaxis().set_ticks([])

    # Save pdf image
    plt.savefig(os.path.join(path_multiearth_coord, lat + '_' + lon+'_visualization_results_threshold_12000_manual.pdf'),format="pdf", bbox_inches="tight", dpi=1000)
    print('pdf file saved')