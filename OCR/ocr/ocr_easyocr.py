import easyocr
from .easyocr_languages import easyocr_languages
from datetime import datetime
from pydantic import BaseModel
import numpy as np

class EasyOCR(BaseModel):
    source_file: str
    text: str
    confidence: float
    detection_time: float


class EasyOCRModel:
    # Sort words into the correct order based on x-coordinate and add it to overall text output
    def add_line_to_extracted_text(line, extracted_text):
        line = dict(sorted(line.items()))
        words = ''
        for word in line.values():
            words = words + ' ' + word

        extracted_text = extracted_text + words + '\n'
        
        return extracted_text

    # Sort words into the correct order
    def order_words(box, text, extracted_text, prev_y_min, prev_y_max, y_buffer, line):
        y_max = box[3][1]  # top right coordinate
        y_min = box[2][1]  # bottom right coordinate
        x_min = box[0][0]  # bottom left coordinate
        height_variation = 2  # +/- half of overall height to buffer

        # sort the text into the correct lines based on y-coordinates
        if (prev_y_min - y_buffer) < y_min and (prev_y_max + y_buffer) > y_max:
            prev_y_min = y_min
            y_buffer = (y_max - y_min) / height_variation
            prev_y_max = y_max
            line[x_min] = text
        
        # Once end of line is reaached add it to the extracted text
        else:
            extracted_text = EasyOCRModel.add_line_to_extracted_text(line, extracted_text)

            line = {x_min: text}
            prev_y_min = y_min
            prev_y_max = y_max
            y_buffer = (y_max - y_min) / height_variation

        return extracted_text, prev_y_min, prev_y_max, y_buffer, line

    def get_text(self, image_file_path, language, paragraph):
        start_time = datetime.now()

        # Map language to language coding
        language_options = easyocr_languages()
        language_coding = language_options[language]

        # Get the boxes, text and confidences for the image
        reader = easyocr.Reader([language_coding])
        results = reader.readtext(image_file_path, paragraph=paragraph)

        # Sort the words into the correct order
        line = {}
        y_buffer = 0
        prev_y_max = 0
        prev_y_min = 0
        extracted_text = ""
        av_confidence = 0
        
        # If paragraph == False, confidence scores are generated
        if paragraph is False:
            count = 0
            for (box, text, confidence) in results:
                extracted_text, prev_y_min, prev_y_max, y_buffer, line = EasyOCRModel.order_words(box,
                                                                                                  text,
                                                                                                  extracted_text,
                                                                                                  prev_y_min,
                                                                                                  prev_y_max,
                                                                                                  y_buffer,
                                                                                                  line)
                av_confidence =+ confidence
                count =+ 1
            
            # Check to see if any words were detected
            try:
                av_confidence = av_confidence / count
            except ZeroDivisionError:
                av_confidence = 0

        # Calculate the overall confidence for output
        else:
            for (box, text) in results: 
                extracted_text, prev_y_min, prev_y_max, y_buffer, line = EasyOCRModel.order_words(box,
                                                                                                  text,
                                                                                                  extracted_text,
                                                                                                  prev_y_min,
                                                                                                  prev_y_max,
                                                                                                  y_buffer,
                                                                                                  line)
            av_confidence = 0.0

        extracted_text = EasyOCRModel.add_line_to_extracted_text(line, extracted_text)

        result = {
            'source_file': "",
            'text': extracted_text,
            'confidence': av_confidence,
            'detection_time': (datetime.now() - start_time).total_seconds()
        }

        return result
