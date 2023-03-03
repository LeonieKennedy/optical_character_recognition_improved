from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from tempfile import NamedTemporaryFile
from pathlib import Path
import shutil
import cv2

from ocr.ocr_messages import ExtractMessagesModel
from ocr.ocr_keras import KerasModel
from ocr.ocr_tesseract import TesseractModel
from ocr.ocr_easyocr import EasyOCRModel
from ocr.ocr_cars import ExtractLicencePlates, ExtractLicencePlatesModel
from classify_image import ClassifyImageModel
import pre_processor

app = FastAPIOffline()

origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://0.0.0.0:8000",
    "http://0.0.0.0:3000",
]

app.add_middleware(    
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# global models so that they only get loaded once
keras_model = None
tesseract_model = None
easyocr_model = None
car_reg_model = None
classify_model = None
processor_model = None
message_model = None


####################################### Pre-processing ######################################################
# Pre-process image: includes adaptive thresholding, skew correction and noise removal
def pre_process(image_file_path, thresholding, skew_correction, noise_removal):
    tmp_path = check_if_tmp_file(image_file_path)

    image = pre_processor.pre_process_image(str(tmp_path), thresholding, skew_correction, noise_removal)
    tmp_path = '/tmp/processed_image.png'
    cv2.imwrite(tmp_path, image)

    return str(tmp_path)

################################## Optical character recognition #######################################
# Extract text using Keras
@app.post("/keras")
def keras(image_file_path: UploadFile,
          thresholding: bool = False,
          skew_correction: bool = False,
          noise_removal: bool = False):

    global keras_model

    tmp_path = check_if_tmp_file(image_file_path)
    new_image = pre_process(tmp_path, thresholding, skew_correction, noise_removal)
    new_tmp_path = check_if_tmp_file(new_image)

    if keras_model is None:
        keras_model = KerasModel()

    results = KerasModel.get_text(keras_model, str(new_tmp_path))
    results["source_file"] = image_file_path.filename

    return results

# Extract text using Tesseract
@app.post("/tesseract")
def tesseract(image_file_path: UploadFile,
              scale: int = 1,
              thresholding: bool = False,
              skew_correction: bool = False,
              noise_removal: bool = False):

    global tesseract_model

    tmp_path = check_if_tmp_file(image_file_path)
    new_image = pre_process(tmp_path, thresholding, skew_correction, noise_removal)
    new_tmp_path = check_if_tmp_file(new_image)

    if tesseract_model is None:
        tesseract_model = TesseractModel()

    results = TesseractModel.get_text(tesseract_model, str(new_tmp_path), scale)
    results["source_file"] = image_file_path.filename

    return results


# Extract text using EasyOCR
@app.post("/easyocr")
def easyocr(image_file_path: UploadFile,
            language: str,
            paragraph: bool = False,
            thresholding: bool = False,
            skew_correction: bool = False,
            noise_removal: bool = False):
    global easyocr_model

    tmp_path = check_if_tmp_file(image_file_path)
    new_image = pre_process(tmp_path, thresholding, skew_correction, noise_removal)
    new_tmp_path = check_if_tmp_file(new_image)

    if easyocr_model is None:
        easyocr_model = EasyOCRModel()

    results = EasyOCRModel.get_text(easyocr_model, str(new_tmp_path), language, paragraph)
    results["source_file"] = image_file_path.filename

    return results

# Extract car registration plates from images
@app.post("/get_car_reg")
def get_car_reg(image_file_path: UploadFile):
    global car_reg_model

    tmp_path = check_if_tmp_file(image_file_path)

    if car_reg_model is None:
        car_reg_model = ExtractLicencePlatesModel()

    results = ExtractLicencePlatesModel.get_text(car_reg_model, str(tmp_path))
    results["source_file"] = image_file_path.filename

    return results

# Extract messages from screenshots
@app.post("/get_messages")
def get_messages(image_file_path: UploadFile):
    global message_model

    tmp_path = check_if_tmp_file(image_file_path)

    if message_model is None:
        message_model = ExtractMessagesModel()

    results = ExtractMessagesModel.get_text(message_model, str(tmp_path))
    results["source_file"] = image_file_path.filename
    with open("text.txt", "w+") as f:
        f.write(results["text"])

    return results


# Categorises image before deciding which ocr model to use.
@app.post("/submit_image")
async def submit_image(image_file_path: UploadFile = File()):
    global classify_model, easyocr_model, keras_model, car_reg_model, message_model

    tmp_path = check_if_tmp_file(image_file_path)

    # load models and save them so that they don't need to be loaded for every run
    if classify_model is None:
        classify_model = ClassifyImageModel(model=None, processor=None)

    category = ClassifyImageModel.classify_image(classify_model, str(tmp_path))

    # remove image shadows
    processed_image = pre_processor.remove_shadows(str(tmp_path))
    processed_tmp_path = check_if_tmp_file(processed_image)

    if category == "car":
        if car_reg_model is None:
            car_reg_model = ExtractLicencePlatesModel()
        results = ExtractLicencePlatesModel.get_text(car_reg_model, str(processed_tmp_path))

    elif category == "document":
        if keras_model is None:
            keras_model = KerasModel()
        results = KerasModel.get_text(keras_model, str(processed_tmp_path))

    elif category == "texting":
        if message_model is None:
            message_model = ExtractMessagesModel()
        results = ExtractMessagesModel.get_text(message_model, str(tmp_path))

    else:
        if easyocr_model is None:
            easyocr_model = EasyOCRModel()

        results = EasyOCRModel.get_text(easyocr_model, str(processed_tmp_path), "English", False)

    with open("text.txt", "w+") as f:
        f.write(results["text"])

    results["source_file"] = image_file_path.filename

    return results


# Load all the ocr and classification models - added to increase speed
@app.get("/load_all_models")
def load_all_models():
    global classify_model, keras_model, easyocr_model, tesseract_model, car_reg_model, message_model
    
    classify_model = ClassifyImageModel(model=None, processor=None)
    keras_model = KerasModel()
    easyocr_model = EasyOCRModel()
    tesseract_model = TesseractModel()
    car_reg_model = ExtractLicencePlatesModel()
    message_model = ExtractMessagesModel()

# Save file as temporary file
def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def check_if_tmp_file(query_file):
    try:
        tmp_path = save_upload_file_tmp(query_file)
    except AttributeError:
        tmp_path = query_file
    return tmp_path