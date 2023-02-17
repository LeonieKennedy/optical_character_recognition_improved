from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel


class ClassifyImage(BaseModel):
    source_file: str
    plate_detected: bool
    text: str
    detection_time: float


class ClassifyImageModel:
    def __init__(self, model, processor):
        self.labels = ["car", "document", "texting"]
        self.threshold = 0.8

        if model is None:
            self.model = CLIPModel.from_pretrained("./models/image_classifier_model")
            self.processor = CLIPProcessor.from_pretrained("./models/image_classifier_processor")

        else:
            self.model = model
            self.processor = processor

    def classify_image(self, image_file_path):
        image = Image.open(image_file_path)
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True)
        output = self.model(**inputs)

        # get the confidence scores from the output
        logits_per_image = output.logits_per_image
        probs = logits_per_image.softmax(dim=-1).detach().numpy()[0]

        count = 0
        found = False
        for i in probs:
            if i > self.threshold:  # if the label confidence is above the threshold(80%), it is that label
                label = self.labels[count]
                print("File name:", image_file_path)
                print("Label:", self.labels[count])
                found = True
            count = count + 1

        if found is False:
            label = "other"
            print("File name:", image_file_path)
            print("Label: Other")

        return label
