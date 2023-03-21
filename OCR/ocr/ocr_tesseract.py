from pytesseract import pytesseract
import cv2
from pydantic import BaseModel
from datetime import datetime

class Tesseract(BaseModel):
    source_file: str
    text: str
    confidence: float
    detection_time: float


class TesseractModel:
    def init(self):
        # path to tesseract on ubuntu
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
        
    def get_text(self, img, scale):
        start_time = datetime.now()

        # Scale image
        height, width, channels = img.shape
        img = cv2.resize(img, ((height * scale), (width * scale)))
        
        # Extract text
        text = pytesseract.image_to_string(img)

        result = {
            'source_file': "",
            'text': text,
            'confidence': None,
            'detection_time': (datetime.now() - start_time).total_seconds()
        }
        return result
