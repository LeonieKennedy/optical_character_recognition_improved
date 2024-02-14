# Optical Character Recognition
Identifies text in images. Specialises in car license plates, documents and text message screenshots.

Custom dataset and trained YOLO models are used for license plates and text messages.

## How to run
To run API, use the following command inside the OCR directory

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

To run the UI with the API, enter the folloing command

```
docker-compose up
```

## Links

- FastAPI Link:

    - http://0.0.0.0:8000/docs#/


- UI Link

    - http://0.0.0.0:3000

