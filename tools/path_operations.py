import os


def get_num_images_in_folder(path_folder: str, image_type: str, file_extension: str):
    """ Returns the number of images with type *image_type* and file extension *file_extension*
    in the folder with path *path_folder*.

    Parameters
    ----------
    path_folder : str
        path of the folder from which images are counted
    image_type : str
        type that images must have to be counted
    file_extension : str
        file extension that the image files must have to be counted

    Returns
    -------
    image_counter : int
        number of images with type *image_type* and file extension *file_extension*
        in the folder with path *path_folder*.

    """
    file_counter = 0  # only images with specified type and file extension are counted
    for file_name in os.listdir(path_folder):
        if file_name.endswith(image_type + file_extension):
            file_counter = file_counter + 1
    return file_counter


def get_path_image(path_folder: str, image_type: str, file_extension: str, image_index: int):
    """ Returns the path of the image stored in the folder *path_folder*, with type
    *image_type* and file extension *file_extension*. If sorting by file_name in ascending order,
    and only considering the images of the specified type and file extension, the returned
    path is linked to the image with index *image_index*.

    Parameters
    ----------
    path_folder: str
        path of the folder where the target image is stored
    image_type: str
        type of the target image
    file_extension: str
        extension of the target image file
    image_index: int
        index of the target image within the folder with path *path_folder*

    Returns
    -------
    output_path : str
        path of the target image

    """
    output_path = -1  # if image with specified type, file extension and index is not found, -1 is returned
    file_counter = 0  # only images with specified type and file extension are counted
    for file_name in os.listdir(path_folder):
        if file_name.endswith(image_type + file_extension):
            if file_counter == image_index:  # the counter of images is compared to the specified image index
                output_path = os.path.join(path_folder, file_name)
                break  # if the corresponding image is found, the loop is automatically stopped
            file_counter = file_counter + 1
    return output_path
