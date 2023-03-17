from fastapi import UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi_offline import FastAPIOffline
from PIL import Image
import io

import plotly.express as px
from ocr.ocr_messages import ExtractMessagesModel
from ocr.ocr_keras import KerasModel
from ocr.ocr_tesseract import TesseractModel
from ocr.ocr_easyocr import EasyOCRModel
from ocr.ocr_cars import ExtractLicencePlatesModel
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
    image = pre_processor.pre_process_image(image_file_path, thresholding, skew_correction, noise_removal)

    return image

################################## Optical character recognition #######################################
# Extract text using Keras
@app.post("/keras")
async def keras(image_file_path: UploadFile=File(),
          thresholding: bool = False,
          skew_correction: bool = False,
          noise_removal: bool = False):

    global keras_model

    contents = await image_file_path.read()
    img = Image.open(io.BytesIO(contents))

    new_image = pre_process(img, thresholding, skew_correction, noise_removal)

    if keras_model is None:
        keras_model = KerasModel()

    results = KerasModel.get_text(keras_model, new_image)
    results["source_file"] = image_file_path.filename

    return results

# Extract text using Tesseract
@app.post("/tesseract")
async def tesseract(image_file_path: UploadFile,
              scale: int = 1,
              thresholding: bool = False,
              skew_correction: bool = False,
              noise_removal: bool = False):

    global tesseract_model

    contents = await image_file_path.read()
    img = Image.open(io.BytesIO(contents))

    new_image = pre_process(img, thresholding, skew_correction, noise_removal)

    if tesseract_model is None:
        tesseract_model = TesseractModel()

    results = TesseractModel.get_text(tesseract_model, new_image, scale)
    results["source_file"] = image_file_path.filename

    return results


# Extract text using EasyOCR
@app.post("/easyocr")
async def easyocr(image_file_path: UploadFile=File(),
            language: str="English",
            paragraph: bool = False,
            thresholding: bool = False,
            skew_correction: bool = False,
            noise_removal: bool = False):
    global easyocr_model

    contents = await image_file_path.read()
    img = Image.open(io.BytesIO(contents))

    new_image = pre_process(img, thresholding, skew_correction, noise_removal)

    if easyocr_model is None:
        easyocr_model = EasyOCRModel()

    results = EasyOCRModel.get_text(easyocr_model, new_image, language, paragraph)
    results["source_file"] = image_file_path.filename

    return results

# Extract car registration plates from images
@app.post("/get_car_reg")
async def get_car_reg(image_file_path: UploadFile=File()):
    global car_reg_model

    contents = await image_file_path.read()
    img = Image.open(io.BytesIO(contents))

    if car_reg_model is None:
        car_reg_model = ExtractLicencePlatesModel()

    results = ExtractLicencePlatesModel.get_text(car_reg_model, img)
    results["source_file"] = image_file_path.filename

    return results

# Extract messages from screenshots
@app.post("/get_messages")
async def get_messages(image_file_path: UploadFile=File()):
    global message_model

    contents = await image_file_path.read()
    img = Image.open(io.BytesIO(contents))

    if message_model is None:
        message_model = ExtractMessagesModel()

    results = ExtractMessagesModel.get_text(message_model, img)
    results["source_file"] = image_file_path.filename
    with open("text.txt", "w+") as f:
        f.write(results["text"])

    return results


# Categorises image before deciding which ocr model to use.
@app.post("/submit_image")
async def submit_image(image_file_path: UploadFile=File()):
    global classify_model, easyocr_model, keras_model, car_reg_model, message_model

    contents = await image_file_path.read()
    img = Image.open(io.BytesIO(contents))

    # load models and save them so that they don't need to be loaded for every run
    if classify_model is None:
        classify_model = ClassifyImageModel(model=None, processor=None)

    category = ClassifyImageModel.classify_image(classify_model, img)

    # remove image shadows
    processed_image = pre_processor.remove_shadows(img)
    with open("text.txt", "w+") as f:

        if category == "vehicle":
            if car_reg_model is None:
                car_reg_model = ExtractLicencePlatesModel()
            results = ExtractLicencePlatesModel.get_text(car_reg_model, processed_image)

        elif category == "document":
            if keras_model is None:
                keras_model = KerasModel()

            results = KerasModel.get_text(keras_model, processed_image)

        elif category == "sms":
            if message_model is None:
                message_model = ExtractMessagesModel()
            results = ExtractMessagesModel.get_text(message_model, img)
            text = "App: " + results["app"] + "\n\n"
            f.write(text)
        else:
            if easyocr_model is None:
                easyocr_model = EasyOCRModel()

            results = EasyOCRModel.get_text(easyocr_model, processed_image, "English", False)
        bytes_image = io.BytesIO()
        img.save(bytes_image, format='PNG')

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
