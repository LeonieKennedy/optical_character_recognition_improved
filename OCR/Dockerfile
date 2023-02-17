FROM python:latest
WORKDIR /app

RUN apt-get update

RUN apt install -y ffmpeg
RUN apt-get install libsm6 libxext6 libgl1  -y
RUN apt install tesseract-ocr -y
RUN apt-get install -y python3-opencv

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy easyocr models for each language into ~/.EasyOCR/model <- makes it work offline
COPY "models/easyocr_models" "~/.EasyOCR/model"

COPY . .

EXPOSE 8000
ENTRYPOINT ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
