import os
import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import plotly.express as px
from pydantic import BaseModel
from datetime import datetime
from ultralytics import YOLO

class ExtractMessages(BaseModel):
    source_file: str
    message_detected: bool
    text: str
    detection_time: float


# Extract message boxes from images and get the text
class ExtractMessagesModel:
    def __init__(self):
        # Load recogniser model
        self.reader = easyocr.Reader(['en'])

        # Constants
        self.INPUT_WIDTH = 640 
        self.INPUT_HEIGHT = 640
        self.CLASSES = ['group', 'message']

        # Load detector model
        self.model = YOLO("./models/message_detector.pt")


    # Locate message boxes
    def detect_messages(self, input_image):
        # image = img.copy()

        predictions = self.model(input_image)
        message_coordinates = list(predictions)[0].boxes.xyxy.cpu().tolist()
        class_values = list(predictions)[0].boxes.cls.cpu().tolist()
        class_confidence= list(predictions)[0].boxes.conf.cpu().tolist()

        print(predictions)
        return message_coordinates, class_values, class_confidence

    # Filter boxes based on confidence and probability scores
    def filter_message_coords(self, message_coordinates, class_values, class_confidence):
        boxes = []
        confidences = []
        names = []

        for i in range(len(message_coordinates)):
            confidence = class_confidence[i]  # confidence of detecting message
            if confidence > 0.4:  # 40%
                names.append(self.CLASSES[int(class_values[i])])
                confidences.append(confidence)
                boxes.append(message_coordinates[i])

        return boxes, confidences, names

    # extract text
    def extract_text(self, image, bbox):
        x, y, width, height = bbox

        message = image[int(y):int(height), int(x):int(width)]  # isolate message from image
        extracted_text = ""

        # shape will be 0 if there were no message  detected
        if 0 in message.shape:
            print("no message")
            return 'message not detected'
        else:
            results = self.reader.readtext(message, paragraph=True)

            for box, text in results:
                extracted_text = extracted_text + text
        
        return extracted_text

    # Annotate input image with boxes and car reg
    def annotate_image(self, image, message_coords, confidences_np, names):
        message_text = ""
        message_detected = True
        all_text = ""
        # message_array = []
        # for each message detected
        for i in range(len(names)):
            message_text = ExtractMessagesModel.extract_text(self, image, message_coords[i])
            # message_array.append(message_text)
            if names[i] == "message":
                all_text = all_text + "\n" + message_text
            else:
                all_text = "Group Name: " + message_text + all_text

        # if all_text != "":
        #     for i in range(len(names)):
        #         # message detection confidence
        #         message_confidence = confidences_np[i]
        #         confidence_text = names[i] + ": " + str((message_confidence * 100))[:5]
        #         x, y, width, height = message_coords[i]
        #         x = int(x)
        #         y = int(y)
        #         width = int(width)
        #         height = int(height)
        #
        #         cv2.rectangle(image, (x, y), (width, height), (255, 0, 255), 2)
        #         cv2.rectangle(image, (x, y - 30), (width, y), (255, 0, 255), -1)
        #         cv2.rectangle(image, (x, height), (width, height + 25), (0, 0, 0), -1)
        #
        #         # Draw car reg
        #         cv2.putText(image, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #         cv2.putText(image, message_array[i], (x, height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if all_text == "message not detected" or all_text == "":
            print("no coords")
            message_detected = False
            message_text = ""

            results = self.reader.readtext(image, paragraph=True)
            print("l:", results)

            for box, text in results:
                message_text = message_text + text


        return image, all_text, message_detected

    # Detect car message  and extract text
    def get_text(self, image_file_path):
        start_time = datetime.now()
        img = cv2.imread(image_file_path)
        # detect message
        message_coordinates, class_values, class_confidence = ExtractMessagesModel.detect_messages(self, img)
        # filter message  coordinates
        filtered_coords, confindences, names = \
            ExtractMessagesModel.filter_message_coords(self,
                                                            message_coordinates,
                                                            class_values,
                                                            class_confidence)
        # annotate_image
        annotated_image, extracted_text, message_detected = \
            ExtractMessagesModel.annotate_image(self,
                                                     img,
                                                     filtered_coords,
                                                     confindences,
                                                     names)

        # fig = px.imshow(annotated_image)
        # fig.update_layout(width=1400, height=800, margin=dict(l=10, r=10, b=10, t=10))
        # fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        # fig.show()
        # fig.write_image("plotly.png")

        result = {
            'source_file': image_file_path,
            'message_detected': message_detected,
            'text': extracted_text,
            'confidence': None,
            'detection_time': (datetime.now() - start_time).total_seconds(),
        }

        return result

model = ExtractMessagesModel()
result = ExtractMessagesModel.get_text(model, "/home/iduadmin/Downloads/telegram_chat.png")
print(result["text"])