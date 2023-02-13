from fastapi import UploadFile, Query, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from datetime import datetime
from tempfile import NamedTemporaryFile
from pathlib import Path
import shutil
import cv2

from mod_keras import KerasModel, Keras
from mod_tesseract import TesseractModel, Tesseract
from mod_easyocr import EasyOCRModel, EasyOCR
from ocr_cars import ExtractLicencePlates, ExtractLicencePlatesModel
from classify_image import ClassifyImage, ClassifyImageModel
from easyocr_languages import easyocr_languages
import pre_processor

app = FastAPI()

origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://0.0.0.0:8000",
    "http://0.0.0.0:3000",
]

app.add_middleware(    
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

keras_model = None
tesseract_model = None
easyocr_model = None
car_reg_model = None
classify_model = None
processor_model = None

# Pre-process image: includes adaptive thresholding, skew correction and noise removal
def pre_process(image_file_path, thresholding, skew_correction, noise_removal):
    tmp_path = check_if_tmp_file(image_file_path)

    image = pre_process_image(str(tmp_path), thresholding, skew_correction, noise_removal)
    tmp_path = '/tmp/processed_image.png'
    cv2.imwrite(tmp_path, image)

    return str(tmp_path)

# Extract text using Keras
@app.post("/keras")
def keras(image_file_path: UploadFile, thresholding: bool = False, skew_correction: bool = False, noise_removal: bool = False):
    global keras_model

    tmp_path = check_if_tmp_file(image_file_path)
    new_image = pre_process(tmp_path, thresholding, skew_correction, noise_removal)
    new_tmp_path = check_if_tmp_file(new_image)

    if keras_model == None:
        keras_model = KerasModel()

    return KerasModel.get_text(keras_model, str(new_tmp_path))

# Extract text using Tesseract
@app.post("/tesseract")
def tesseract(image_file_path: UploadFile, scale: int = 1, thresholding: bool = False, skew_correction: bool = False, noise_removal: bool = False):
    global tesseract_model

    tmp_path = check_if_tmp_file(image_file_path)
    new_image = pre_process(tmp_path, thresholding, skew_correction, noise_removal)
    new_tmp_path = check_if_tmp_file(new_image)

    if tesseract_model == None:
        tesseract_model = TesseractModel()

    return TesseractModel.get_text(model, str(new_tmp_path), scale)

# Extract text using EasyOCR
@app.post("/easyocr")
def easyocr(image_file_path: UploadFile, language: str, paragraph: bool = False, thresholding: bool = False, skew_correction: bool = False, noise_removal: bool = False):    
    tmp_path = check_if_tmp_file(image_file_path)
    new_image = pre_process(tmp_path, thresholding, skew_correction, noise_removal)
    new_tmp_path = check_if_tmp_file(new_image)    
    model = EasyOCRModel()
    return EasyOCRModel.get_text(model, str(new_tmp_path), language, paragraph)

# Extract car registration plates from images
@app.post("/get_car_reg")
def get_car_reg(image_file_path: UploadFile):
    global car_reg_model

    tmp_path = check_if_tmp_file(image_file_path)

    if car_reg_model == None:
        car_reg_model = ExtractLicencePlatesModel()
    
    return ExtractLicencePlates.get_text(car_reg_model, str(tmp_path))

# Categorises image before deciding which ocr model to use.
@app.post("/submit_image")
def submit_image(image_file_path: UploadFile):
    global classify_model, keras_model, car_reg_model
    tmp_path = check_if_tmp_file(image_file_path)

    print("start")
    # load models and save them so that they don't need to be loaded for every run
    if classify_model == None:
       classify_model = ClassifyImageModel(model=None, processor=None)
    
    category = ClassifyImageModel.classify_image(classify_model, str(tmp_path))
    print("classified")
    # remove image shadows
    processed_image = pre_processor.remove_shadows(str(tmp_path))
    processed_tmp_path = check_if_tmp_file(processed_image)
    processed_tmp_path = tmp_path
    if category == "car":
        print("car")
        if car_reg_model == None:
            car_reg_model = ExtractLicencePlatesModel()
        return ExtractLicencePlatesModel.get_text(car_reg_model, str(processed_tmp_path))
    
    if category == "document":
        print("document")
        if keras_model == None:
            keras_model = KerasModel()
        return KerasModel.get_text(keras_model, str(processed_tmp_path))

    if category == "message":
        print("message")
        if keras_model == None:
            keras_model = KerasModel()
        return KerasModel.get_text(keras_model, str(processed_tmp_path))

    else:
        print("other")
        if keras_model == None:
            keras_model = KerasModel()
        return KerasModel.get_text(keras_model, str(processed_tmp_path))


# Load all of the ocr and classification models - added to increase speed
@app.get("/load_all_models")
def load_all_models():
    global classify_model, keras_model, easyocr_model, tesseract_model, car_reg_model
    
    classify_model = ClassifyImageModel(model=None, processor=None)
    keras_model = KerasModel()
    easyocr_model = EasyOCRModel()
    tesseract_model = TesseractModel()
    car_reg_model = ExtractLicencePlatesModel()

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