import os
import easyocr
import cv2
import numpy as np
import plotly.express as px
from pydantic import BaseModel
from datetime import datetime
from ultralytics import YOLO
from PIL import Image

class ExtractMessages(BaseModel):
    source_file: str
    message_detected: bool
    app: str
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
        self.CLASSES = ['android', 'facebook', 'group', 'hangouts', 'imessage', 'instagram', 'line', 'received', 'sent', 'signal', 'skype', 'snapchat', 'telegram', 'twitter', 'wechat', 'whatsapp']
        # Load detector model
        self.model = YOLO("./models/message_detector.pt")


    # Order messages with top messages first and bottom messages last
    def order_messages(coordinates, categories, confidences):
        swapped = True
        while swapped is True:
            swapped = False

            for i in range(0, len(coordinates) - 1):
                if coordinates[i][1] > coordinates[i + 1][1]:
                    coordinates[i], coordinates[i + 1] = coordinates[i + 1], coordinates[i]
                    categories[i], categories[i+1] = categories[i+1], categories[i]
                    confidences[i], confidences[i+1] = confidences[i+1], confidences[i]
                    swapped = True


        return coordinates, categories, confidences

    # Locate message boxes
    def detect_messages(self, input_image):

        predictions = self.model(input_image)
        message_coordinates = list(predictions)[0].boxes.xyxy.cpu().tolist()
        class_values = list(predictions)[0].boxes.cls.cpu().tolist()
        class_confidence= list(predictions)[0].boxes.conf.cpu().tolist()
        message_coordinates, class_values, class_confidence = ExtractMessagesModel.order_messages(message_coordinates, class_values, class_confidence)
        return message_coordinates, class_values, class_confidence

    # Filter boxes based on confidence and probability scores
    def filter_message_coords(self, message_coordinates, class_values, class_confidence):
        boxes = []
        confidences = []
        names = []

        for i in range(len(message_coordinates)):
            confidence = class_confidence[i]  # confidence of detecting message
            print(f"""
            coords: {message_coordinates[i]}
            confidence: {confidence}
            class: {self.CLASSES[int(class_values[i])]}""")

            if confidence > 0.75:  # 80%
                names.append(self.CLASSES[int(class_values[i])])
                confidences.append(confidence)
                boxes.append(message_coordinates[i])

        return boxes, confidences, names

    # extract text
    def extract_text(self, image, bbox):
        x0, y0, x1, y1 = bbox

        message = image[int(y0):int(y1), int(x0):int(x1)]  # isolate message from image
        extracted_text = ""
        # shape will be 0 if there were no message  detected
        if 0 in message.shape:
            print("no message")
            return ""
        else:
            results = self.reader.readtext(message, paragraph=True)

            for box, text in results:
                extracted_text = extracted_text + text
        
        return extracted_text

    # Annotate input image with boxes and car reg
    def annotate_image(self, image, message_coords, confidences_np, names):
        message_detected = True
        app = "Unknown"
        all_text = ""
        message_array = []
        print(names)
        # for each message detected
        for i in range(len(names)):
            message_text = ExtractMessagesModel.extract_text(self, image, message_coords[i])
            message_array.append(message_text)
            if names[i] == "sent" or names[i] == "received":
                print("m ", message_text)
                all_text = all_text + "\n" + names[i] + ": " + message_text
            elif names[i] == "group":
                all_text = "Group Name: " + message_text + all_text
            else:
                app = names[i]

        print(all_text)
        if all_text != "":
            for i in range(len(names)):
                # message detection confidence
                message_confidence = confidences_np[i]
                confidence_text = names[i] + ": " + str((message_confidence * 100))[:5]
                x0, y0, x1, y1 = message_coords[i]
                x0 = int(x0)
                y0 = int(y0)
                x1 = int(x1)
                y1 = int(y1)

                cv2.rectangle(image, (x0, y0), (x1, y1), (30, 144, 250), 2)
                cv2.rectangle(image, (x0, y0 - 30), (x1, y0), (30, 144, 250), -1)
                cv2.rectangle(image, (x0, y1), (x1, y1 + 25), (0, 0, 0), -1)

                # Draw message
                cv2.putText(image, confidence_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(image, message_array[i], (x0, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)

        if all_text == "":
            print("no coords")
            message_detected = False

            results = self.reader.readtext(image, paragraph=True)

            for box, text in results:
                all_text = all_text + text + "\n"


        return image, all_text, message_detected, app

    # Detect car message  and extract text
    def get_text(self, img):
        start_time = datetime.now()

        # convert PIL to OpenCV
        pil_image = img.convert("RGB")
        open_cv_image = np.array(pil_image)
        img = open_cv_image[:, :, ::-1].copy()


        # detect message
        message_coordinates, class_values, class_confidence = ExtractMessagesModel.detect_messages(self, img)
        # filter message  coordinates
        filtered_coords, confindences, names = \
            ExtractMessagesModel.filter_message_coords(self,
                                                            message_coordinates,
                                                            class_values,
                                                            class_confidence)
        # annotate_image
        annotated_image, extracted_text, message_detected, app = \
            ExtractMessagesModel.annotate_image(self,
                                                     img,
                                                     filtered_coords,
                                                     confindences,
                                                     names)

        print("\n\n\n\n"+extracted_text)
        fig = px.imshow(annotated_image)
        fig.update_layout(width=1400, height=800, margin=dict(l=10, r=10, b=10, t=10))
        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.show()
        fig.write_image("plotly.png")

        result = {
            'source_file': "",
            'message_detected': message_detected,
            'app': app,
            'text': extracted_text,
            'confidence': None,
            'detection_time': (datetime.now() - start_time).total_seconds(),
        }

        return result

# model = ExtractMessagesModel()
#
# img = Image.open("/home/iduadmin/Downloads/telegram_chat.png")
#
# results = ExtractMessagesModel.get_text(model, img)