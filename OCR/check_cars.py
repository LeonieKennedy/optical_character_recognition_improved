# File to be deleted once complete


from difflib import SequenceMatcher

from mod_keras import KerasModel
from mod_easyocr import EasyOCRModel
from ocr_cars import ExtractLicencePlatesModel, ExtractLicencePlates
import os
import keras_ocr
from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_mean
from skimage.filters import threshold_otsu

from skimage.util import invert
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import gaussian
from scipy.ndimage import interpolation as inter
import scipy
from pre_processor import remove_shadows

model = ExtractLicencePlatesModel()
count = 0
h_count = 0
total = 0
path_to_images = "../Task3_images/car/"

for root, dirs, file_names in os.walk(path_to_images):
    for file_name in file_names:
        print(file_name)

        deshadowed_image = remove_shadows(path_to_images + file_name)
        output = ExtractLicencePlatesModel.get_text(model, "no_shadow.jpg")["text"]
        print("output:", output)
        
        reference = str(file_name.split('.')[0].lower())

        if output == None:
            output = ""

        output = output.lower()

        s = SequenceMatcher(None, reference, output)

        print("\n"+reference)
        print(output)        
        print(s.ratio()*100)
        print("\n")
        total = total + (s.ratio()*100)
        count = count + 1

        if s.ratio()*100 == 100:
            h_count = h_count + 1
        
print("Average:", total / count)
print("100%:", (h_count / count) * 100)
print("Count:", count)