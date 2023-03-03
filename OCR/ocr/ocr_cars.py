import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io
import plotly.express as px
from pydantic import BaseModel
from datetime import datetime


class ExtractLicencePlates(BaseModel):
    source_file: str
    plate_detected: bool
    text: str
    detection_time: float


# Extract car licence plates from images and get the reg number
class ExtractLicencePlatesModel:
    def __init__(self):
        # Load recogniser model
        self.reader = easyocr.Reader(['en'])
        
        # Constants
        self.INPUT_WIDTH = 640 
        self.INPUT_HEIGHT = 640

        # Load detector model
        self.model = cv2.dnn.readNetFromONNX('./models/license_plate_detector.onnx')
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Locate licence platese
    def detect_licence_plates(self, input_image):
        # 1/255 = scale factor
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.model.setInput(blob)
        predictions = self.model.forward()
        print(predictions.shape)
        licence_plate_coordinates = predictions[0]
        
        return input_image, licence_plate_coordinates

    # Filter boxes based on confidence and probability scores
    def filter_licence_coords(self, input_image, licence_plate_coordinates):
        # center x, center y, width, height, confidence, probability
        boxes = []
        confidences = []

        image_width, image_height = input_image.shape[:2]  # only get image width and height
        x_factor = image_width/self.INPUT_WIDTH
        y_factor = image_height/self.INPUT_HEIGHT

        for i in range(len(licence_plate_coordinates)):
            rows = licence_plate_coordinates[i]
            confidence = rows[4]  # confidence of detecting licence plate
            if confidence > 0.4:  # 40%
                class_score = rows[5]  # probability score of licence plate
                if class_score > 0.25:  # 25%
                    center_x, center_y, width, height = rows[0:4]

                    left = int((center_x - 0.5*width)*x_factor)
                    top = int((center_y-0.5*height)*y_factor)
                    width = int(width*x_factor)
                    height = int(height*y_factor)
                    box = np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)

        # Clean
        filtered_boxes = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()
        
        # non-maximum supression
        index = cv2.dnn.NMSBoxes(filtered_boxes, confidences_np, 0.25, 0.45)

        return filtered_boxes, confidences_np, index

    # extract text
    def extract_text(self, image, bbox):
        x, y, width, height = bbox
        licence_plate = image[y:y+height, x:x+width]  # isolate license plate from image
        extracted_text = ""

        # shape will be 0 if there were no licence plates detected
        if 0 in licence_plate.shape:
            print("no plate")
            return 'number plate not detected'
        else:
            results = self.reader.readtext(licence_plate, paragraph=True)

            for box, text in results:
                extracted_text = extracted_text + text
        
        return extracted_text

    # Annotate input image with boxes and car reg
    def annotate_image(self, image, licence_coords, confidences_np, index):
        licence_text = ""
        plate_detected = True

        # for each licence plate detected
        for i in index:
            x, y, width, height = licence_coords[i]

            # Plate detection confidence
            plate_confidence = confidences_np[i]
            confidence_text = 'plate: {:.0f}%'.format(plate_confidence*100)
            licence_text = ExtractLicencePlatesModel.extract_text(self, image, licence_coords[i])

        if licence_text == "number plate not detected" or licence_text == "":
            print("no coords")
            plate_detected = False
            licence_text = ""

            results = self.reader.readtext(image, paragraph=True)
            print("l:", results)

            for box, text in results:
                licence_text = licence_text + text

        else:
            # Highlight plate
            cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 255), 2)
            cv2.rectangle(image, (x, y-30), (x+width, y), (255, 0, 255), -1)
            cv2.rectangle(image, (x, y+height), (x+width, y+height+25), (0, 0, 0), -1)

            # Draw car reg
            cv2.putText(image, confidence_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(image, licence_text, (x, y+height+27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        return image, licence_text, plate_detected

    # Detect car licence plates and extract text
    def get_text(self, image_file_path):
        start_time = datetime.now()
        img = io.imread(image_file_path)    
        # detect licence plates
        input_image, licence_plate_coordinates = ExtractLicencePlatesModel.detect_licence_plates(self, img)
        # filter licence plate coordinates
        filtered_coords, confidences_np, index = \
            ExtractLicencePlatesModel.filter_licence_coords(self,
                                                            input_image,
                                                            licence_plate_coordinates)
        # annotate_image
        annotated_image, extracted_text, plate_detected = \
            ExtractLicencePlatesModel.annotate_image(self,
                                                     img,
                                                     filtered_coords,
                                                     confidences_np,
                                                     index)

        # fig = px.imshow(annotated_image)
        # fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
        # fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        # fig.show()

        result = {
            'source_file': image_file_path,
            'plate_detected': plate_detected,
            'text': extracted_text,
            'confidence': None,
            'detection_time': (datetime.now() - start_time).total_seconds(),
        }

        return result
