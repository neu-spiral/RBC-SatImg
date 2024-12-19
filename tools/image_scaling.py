import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from image_reader import ReadSentinel2
from configuration import Config
from tools.cloud_filtering import check_cloud_threshold, scale_image
from tools.operations import normalize_image
from tools.spectral_index import get_broadband_index
from training import plot_labels_training_images

# Create image reader object
Config.shifting_factor_available = False
image_reader = ReadSentinel2(Config.image_dimensions[Config.scenario]['dim_x'],
                             Config.image_dimensions[Config.scenario]['dim_y'])

# ----------------------------------------------------------------------------
# ------------------------ STORE REFERENCE IMAGE SHIFTING VALUE
# ----------------------------------------------------------------------------
index_ref_image = 0
path_training_images = os.path.join(Config.path_sentinel_images, 'training')  # we get the reference image from the
# training folder
coord_scale_x = Config.coord_scale_x
coord_scale_y = Config.coord_scale_y
dim_x = Config.image_dimensions[Config.scenario]['dim_x']
dim_y = Config.image_dimensions[Config.scenario]['dim_y']

# Read reference image
ref_image, date_string = image_reader.read_image(path=path_training_images,
                                             image_idx=index_ref_image)

offset_ref = []
for idx, band_i in enumerate(Config.bands_to_read):
    # cut image to area without clouds or cloud shadows
    pixels_no_clouds = ref_image[:, idx].reshape(dim_x, dim_y)[coord_scale_x[0]:coord_scale_x[1],
                       coord_scale_y[0]:coord_scale_y[1]].flatten()
    offset_ref.append(np.mean(pixels_no_clouds))

#check_cloud_threshold(image=ref_image, image_date=date_string)

# ----------------------------------------------------------------------------
# ------------------------ TRAINING IMAGES
# ----------------------------------------------------------------------------
# Paths
path_training_images = os.path.join(Config.path_sentinel_images, 'training')

# Specify images to plot
index_images_to_evaluate = [0, 1, 5]
index_images_to_evaluate = [0, 1, 2]
# shifting_factor_training = np.ndarray([2, len(index_images_to_evaluate)])
shifting_factor_training = dict()

for loop_idx, image_idx in enumerate(index_images_to_evaluate):
    # All bands of the image with index *image_idx* are stored in *image_all_bands*
    image, date_string = image_reader.read_image(path=path_training_images,
                                                 image_idx=image_idx)

    # scaled_image, hist_offset = scale_image(image=image, image_date=date_string)
    scaled_image, shifting_factor_training[image_idx] = scale_image(image=image, offset_ref=offset_ref)

    index = get_broadband_index(data=image, bands=['3', '8'])
    index_scaled = get_broadband_index(data=scaled_image, bands=['3', '8'])

    #accepted_cloud_mask = check_cloud_threshold(image=scaled_image, image_date=date_string)
    """
    # Debug
    plt.figure(), plt.hist(index, bins=100)
    plt.figure(), plt.hist(index_scaled, bins=100)
    # Check cloud mask (after scaling)
    accepted_cloud_mask = check_cloud_threshold(image=image, image_date=date_string)
    accepted_cloud_mask = check_cloud_threshold(image=scaled_image, image_date=date_string)
    print('image has been scaled')
    print(date_string)
    #shifting_factor_training[:, loop_idx] = [image_idx, hist_offset]
    plot_labels_training_images(training_images=scaled_image,
                                bands_index=Config.bands_spectral_index[Config.scenario],
                                threshold=Config.threshold_index_labels[Config.scenario], mask=np.zeros(256*256))
    """

pickle_file_path = os.path.join(Config.path_sentinel_images, 'shifting_factor_training.pkl')
pickle.dump(shifting_factor_training, open(pickle_file_path, 'wb'))

# ----------------------------------------------------------------------------
# ------------------------ EVALUATION IMAGES
# ----------------------------------------------------------------------------
# Paths
path_evaluation_images = os.path.join(Config.path_sentinel_images, 'evaluation')

# Specify images to plot
index_images_to_evaluate = [0, 2, 3, 5, 7, 9, 11, 21, 49, 66, 67, 69, 71, 72, 73, 74, 75, 76, 77, 78,
                        81, 82, 83, 89, 90, 101, 103, 108, 130, 134, 135, 140, 142, 144, 145, 146, 147,
                        148, 150, 161, 175]  # original vector (from cloud filter)
index_images_to_evaluate = [0, 2, 3, 5, 7, 9, 11, 21, 66, 67, 69, 71, 72, 73, 75, 76, 77, 78,
                        82, 90, 101, 103, 108, 134, 145, 146,
                        150, 161, 175]  # some images have been discarded because there were clouds in the cropped area

# shifting_factor_evaluation = np.ndarray([2, len(index_images_to_evaluate)])
shifting_factor_evaluation = dict()

for loop_idx, image_idx in enumerate(index_images_to_evaluate):
    # All bands of the image with index *image_idx* are stored in *image_all_bands*
    image, date_string = image_reader.read_image(path=path_evaluation_images,
                                                 image_idx=image_idx)

    # scaled_image, hist_offset = scale_image(image=image, image_date=date_string)
    scaled_image, shifting_factor_evaluation[image_idx] = scale_image(image=image, offset_ref=offset_ref)

    # Check cloud mask (after scaling)
    # accepted_cloud_mask = check_cloud_threshold(image=scaled_image, image_date=date_string)
    print('image has been scaled')
    # shifting_factor_evaluation[:, loop_idx] = [image_idx, hist_offset]

pickle_file_path = os.path.join(Config.path_sentinel_images, 'shifting_factor_evaluation.pkl')
pickle.dump(shifting_factor_evaluation, open(pickle_file_path, 'wb'))

"""
# Get reference image (i.e., first image from the evaluation images folder)
ref_image_idx = 0
ref_image, _ = image_reader.read_image(path=path_evaluation_images, image_idx=ref_image_idx)

# Scale evaluation images
for image_idx in index_images_to_evaluate:
    # All bands of the image with index *image_idx* are stored in *image_all_bands*
    image, date_string = image_reader.read_image(path=path_evaluation_images,
                                                           image_idx=image_idx)
    # Check cloud mask (before scaling)
    # We check the cloud mask because the check_cloud_threshold() function plots the pixel histograms for
    #different bands and the RGB image as well
    accepted_cloud_mask = check_cloud_threshold(image=image, image_date=date_string)

    # Scale image
    normalized_image = normalize_image(image_all_bands=image)
    scaled_image = scale_image(ref_image=ref_image, image=image, image_date=date_string)


    # Check cloud mask (after scaling)
    accepted_cloud_mask = check_cloud_threshold(image=scaled_image, image_date=date_string)
    print('image has been scaled')
"""
