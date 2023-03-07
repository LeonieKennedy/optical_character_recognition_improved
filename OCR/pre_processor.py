import numpy as np
from PIL import Image
from scipy.ndimage import interpolation as inter
import cv2


# Get a score for each angle tested
def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=True, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


# Straighten image
def skew_correction(image_file_path):
    # convert to binary
    img = Image.open(image_file_path)

    delta = 1
    limit = 180  # max rotation allowed
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle: {}'.format(best_angle))
    img_g = cv2.imread(image_file_path)

    # correct skew
    data = inter.rotate(img_g, best_angle, reshape=True, order=0)

    return data


# Remove noise
def noise_removal(image_file):
    img = cv2.imread(image_file)
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

    return dst


# Binarise the image
def adaptive_thresholding(image_file):
    img = cv2.imread(image_file, 0)
    new_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)  # works better

    return new_img


# remove shadows
def remove_shadows(image_file):
    print("here2", image_file)
    # convert PIL to OpenCV
    pil_image = image_file.convert("RGB")
    open_cv_image = np.array(pil_image)
    img = open_cv_image[:, :, ::-1].copy()


    # img = cv2.imread(image_file, -1)
    rgb_planes = cv2.split(img)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    result = cv2.merge(result_planes)
    result = cv2.fastNlMeansDenoisingColored(result, None, 1, 10, 7, 15)

    # shadowless_image_name = "shadowless_image.jpg"
    # cv2.imwrite(shadowless_image_name, result)

    return result


# Create a new pre-processed image based on user input
def pre_process_image(image_file, thresholding, skew, noise):

    # convert PIL to OpenCV
    print("Image" , image_file)
    pil_image = image_file.convert("RGB")
    open_cv_image = np.array(pil_image)
    img = open_cv_image[:, :, ::-1].copy()

    # Skew correction
    if skew is True:
        new_img = skew_correction(img)
    # Adaptive thresholding
    elif thresholding is True:
        new_img = adaptive_thresholding(img)

    # Noise removal
    elif noise is True:
        new_img = noise_removal(img)

    else:
        return img
    print(type(new_img))

    return new_img
