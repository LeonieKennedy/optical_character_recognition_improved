import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
def correct_orientation(image):
    image = cv2.imread(image)
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the angle of the largest contour
    angles = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        angles.append(angle)
    angle = np.median(angles)

    # Rotate the image to correct the orientation
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    # find the new width and height bounds
    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    rotated = cv2.warpAffine(image, M, (bound_w, bound_h))#, flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REPLICATE)

    return rotated

image = correct_orientation("/home/iduadmin/PycharmProjects/OCR (another copy)/Task3_images/document/document7.png")
cv2.imwrite("output.jpg", image)

