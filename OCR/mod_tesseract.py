from pytesseract import pytesseract
import cv2
from pydantic import BaseModel
from datetime import datetime


class Tesseract(BaseModel):
    source_file: str
    text: str
    detection_time: float


class TesseractModel:
    def init(self):
        pytesseract.pytesseract.tesseract_cmd = r"/usr/share/tesseract-ocr/4.00/tessdata"
        
    def get_text(self, image_file_path, scale):
        start_time = datetime.now()

        # Scale image
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape
        img = cv2.resize(img, ((height * scale), (width * scale)))
        
        # Extract text
        text = pytesseract.image_to_string(img)
        
        result = {
            'source_file': image_file_path,
            'text': text,
            'detection_time': (datetime.now() - start_time).total_seconds()
        }
        return result
