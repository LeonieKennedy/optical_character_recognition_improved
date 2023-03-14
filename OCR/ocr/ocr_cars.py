import os
import easyocr
import cv2
import numpy as np
import plotly.express as px
from pydantic import BaseModel
from datetime import datetime
from PIL import Image

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


    # extract text
    def extract_text(self, image, bbox):
        x, y, w, h = bbox
        licence_plate = image[y:y + h, x:x + w]  # isolate license plate from image

        # shape will be 0 if there were no licence plates detected
        if 0 in licence_plate.shape:
            print('no number')
            return ""
        else:
            extracted_text = ""
            results = self.reader.readtext(licence_plate, paragraph=True)

            for (box, text) in results:
                extracted_text = extracted_text + text + "\n"

            extracted_text = extracted_text[:-1]

            return extracted_text

    # Locate licence plates
    def detect_licence_plates(self, img):
        # Reshape image
        image = img.copy()
        rows, columns, channels = image.shape

        max_rc = max(rows, columns)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:rows, 0:columns] = image[:, :, :3]

        # 1/255 = scale factor
        blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (self.INPUT_WIDTH, self.INPUT_HEIGHT), swapRB=True, crop=False)
        self.model.setInput(blob)
        predictions = self.model.forward()
        detections = predictions[0]

        return input_image, detections

    # Filter boxes based on confidence and probability scores
    def filter_licence_coords(self, input_image, detections):
        # center x, center y, width, height, confidence, probability
        boxes = []
        confidences = []

        image_width, image_height = input_image.shape[:2]  # only get image width and height
        x_factor = image_width / self.INPUT_WIDTH
        y_factor = image_height / self.INPUT_HEIGHT

        for i in range(len(detections)):
            rows = detections[i]
            confidence = rows[4]  # confidence of detecting license plate
            if confidence > 0.4:  # 40%
                class_score = rows[5]  # probability score of license plate
                if class_score > 0.25:  # 25%
                    center_x, center_y, width, height = rows[0:4]

                    left = int((center_x - 0.5 * width) * x_factor)
                    top = int((center_y - 0.5 * height) * y_factor)
                    width = int(width * x_factor)
                    height = int(height * y_factor)
                    box = np.array([left, top, width, height])

                    confidences.append(confidence)
                    boxes.append(box)

        # clean
        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # non-maximum suppression
        index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
        return boxes_np, confidences_np, index

    # Annotate input image with boxes and car reg
    def annotate_image(self, image, licence_coords, confidences_np, index):
        plate_detected = True
        licence_text = ""

        # for each licence plate detected
        for i in index:
            x, y, w, h = licence_coords[i]

            # Plate detection confidence
            plate_confidence = confidences_np[i]
            confidence_text = 'plate: {:.0f}%'.format(plate_confidence * 100)
            index_licence_text = ExtractLicencePlatesModel.extract_text(self, image, licence_coords[i])

            if index_licence_text != "":
                licence_text = licence_text + index_licence_text + "\n"

            # Highlight plate
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.rectangle(image, (x, y - 30), (x + w, y), (255, 0, 255), -1)
            cv2.rectangle(image, (x, y + h), (x + w, y + h + 25), (0, 0, 0), -1)

            cv2.putText(image, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(image, licence_text, (x, y + h + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

        if licence_text == "":
            plate_detected = False
            results = self.reader.readtext(image, paragraph=True)

            for box, text in results:
                print(box)
                licence_text = licence_text + text + "\n"

            print("Licence plate not detected \nText found:", licence_text)

        licence_text = licence_text[:-1]
        return image, licence_text, plate_detected

    # Detect car licence plates and extract text
    def get_text(self, image_file_path):
        start_time = datetime.now()

        image_file_path = np.array(image_file_path, dtype="uint8")
        img = cv2.cvtColor(image_file_path, cv2.COLOR_BGR2RGB)

        input_image, detections = ExtractLicencePlatesModel.detect_licence_plates(self, img)
        boxes_np, confidences_np, index = ExtractLicencePlatesModel.filter_licence_coords(self, input_image, detections)
        result_img, extracted_text, plate_detected = ExtractLicencePlatesModel.annotate_image(self, img, boxes_np, confidences_np, index)

        fig = px.imshow(result_img)
        fig.update_layout(width=700, height=400, margin=dict(l=10, r=10, b=10, t=10))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.show()

        result = {
            'source_file': "",
            'plate_detected': plate_detected,
            'text': extracted_text,
            'confidence': None,
            'detection_time': (datetime.now() - start_time).total_seconds(),
        }

        return result


# model = ExtractLicencePlatesModel()
#
# img = Image.open("/home/iduadmin/PycharmProjects/OCR (another copy)/Task3_images/london_traffic.jpg")
#
# results = ExtractLicencePlatesModel.get_text(model, img)
# print(results)