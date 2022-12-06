import numpy as np

from benchmark_models.deepwatermap_main import deepwatermap


def find_padding(v, divisor=32):
    """ Function extracted from the inference.py script in the ./deepwatermap_main/ folder.
    No changes have been applied to this function.

    """
    v_divisible = max(divisor, int(divisor * np.ceil(v / divisor)))
    total_pad = v_divisible - v
    pad_1 = total_pad // 2
    pad_2 = total_pad - pad_1
    return pad_1, pad_2


def main_deepwater(checkpoint_path: str, image: str):
    """ Function extracted from the inference.py script in the ./deepwatermap_main/ folder.
    The function has been changed so that the image is passed as a parameter, instead of the image
    path. Therefore, there is no need to read the image in this function.

    """
    # load the model
    model = deepwatermap.model()
    model.load_weights(checkpoint_path)

    # load and preprocess the input image
    # image = tiff.imread(image_path)  # the function has been changed so that the image
    # is directly passed as a parameter
    #     print(image.shape,"__1___")
    pad_r = find_padding(image.shape[0])
    pad_c = find_padding(image.shape[1])
    image = np.pad(image, ((pad_r[0], pad_r[1]), (pad_c[0], pad_c[1]), (0, 0)), 'reflect')

    # solve no-pad index issue after inference
    if pad_r[1] == 0:
        pad_r = (pad_r[0], 1)
    if pad_c[1] == 0:
        pad_c = (pad_c[0], 1)

    image = image.astype(np.float32)

    # remove nans (and infinity) - replace with 0s
    image = np.nan_to_num(image, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    image = image - np.min(image)
    image = image / np.maximum(np.max(image), 1)

    # run inference
    image = np.expand_dims(image, axis=0)
    dwm = model.predict(image)
    #     print(dwm.shape,"__model_res__")
    dwm = np.squeeze(dwm)
    dwm = dwm[pad_r[0]:-pad_r[1], pad_c[0]:-pad_c[1]]

    # soft threshold
    dwm = 1. / (1 + np.exp(-(16 * (dwm - 0.5))))
    dwm = np.clip(dwm, 0, 1)
    return dwm
