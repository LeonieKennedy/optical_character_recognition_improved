import os

import keras_ocr
from pydantic import BaseModel
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow
import pickle
from keras_ocr.detection import build_keras_model
import string 


class Keras(BaseModel):
    source_file: str
    text: str
    confidence: float
    detection_time: float


class KerasModel:
    def __init__(self):
        # Create detector and recogniser models and assign weights to them
        self.detector = keras_ocr.detection.Detector(weights=None)
        self.detector.model.load_weights(f"./models/keras_detector.h5")

        self.recognizer = keras_ocr.recognition.Recognizer(alphabet=(string.digits + string.ascii_lowercase),
                                                           weights=None)
        self.recognizer.model.load_weights("./models/keras_recognizer.h5")

    # Sort words into the correct order based on x-coordinate and add it to overall text output
    def add_line_to_complete(line, extracted_text):
        line = dict(sorted(line.items()))
        words = ''
        for word in line.values():
            words = words + ' ' + word

        extracted_text = extracted_text + words + '\n'
        
        return extracted_text
    
    def get_text(self, image_file_path):
        start_time = datetime.now()

        height_variation = 2  # +/- half of overall height of box to buffer
        # Extract text from input image
        pipeline = keras_ocr.pipeline.Pipeline(detector=self.detector, recognizer=self.recognizer)
        image = keras_ocr.tools.read(image_file_path)
        prediction = pipeline.recognize([image])[0]
        # Annotate input image with boxes
        fig, ax = plt.subplots()
        image = keras_ocr.tools.drawAnnotations(image=image, predictions=prediction)
        plt.savefig("./images/annotated_keras.jpg")
        
        # Sort words into the correct order
        line = {}
        y_buffer = 0
        prev_y_max = 0
        prev_y_min = 0
        extracted_text = ""

        for text, box in prediction: 
            y_max = box[:, 1].max()  # highest y coordinate
            y_min = box[:, 1].min()  # lowest y coordinate
            x_min = box[:, 0].min()  # lowest x coordinate

            # sort the text into the correct lines based on y-coordinates
            if (prev_y_min - y_buffer) < y_min and (prev_y_max + y_buffer) > y_max:
                prev_y_min = y_min
                y_buffer = (y_max - y_min) / height_variation
                prev_y_max = y_max
                line[x_min] = text
                
            else:
                extracted_text = KerasModel.add_line_to_complete(line, extracted_text)

                line = {x_min: text}

                prev_y_min = y_min
                prev_y_max = y_max
                y_buffer = (y_max - y_min) / height_variation

        extracted_text = KerasModel.add_line_to_complete(line, extracted_text)

        result = {
            'source_file': image_file_path,
            'text': extracted_text,
            'confidence': None,
            'detection_time': (datetime.now() - start_time).total_seconds()
        }
        
        return result
