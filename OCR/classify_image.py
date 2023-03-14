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
        self.labels = ["vehicle", "document", "texting"]  # 3 possible categories + other
        self.threshold = 0.6  # threshold = 75% -> any image with a confidence below this for all 3 categories is "other"

        # load pre-trained saved models
        if model is None:
            self.model = CLIPModel.from_pretrained("./models/image_classifier_model")
            self.processor = CLIPProcessor.from_pretrained("./models/image_classifier_processor")

        else:
            self.model = model
            self.processor = processor

    # work out which category the image fits into
    def classify_image(self, image):
        # image = Image.open(image_file_path)
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True)
        output = self.model(**inputs)

        # get the confidence scores from the output
        logits_per_image = output.logits_per_image
        probs = logits_per_image.softmax(dim=-1).detach().numpy()[0]
        print(probs)
        count = 0
        found = False
        for i in probs:
            if i > self.threshold:  # if the label confidence is above the threshold(80%), it is that label
                label = self.labels[count]
                print("File name:", "image_file_path")
                print("Label:", self.labels[count])
                found = True
            count = count + 1

        # if image doesn't fit into any category
        if found is False:
            label = "other"
            print("File name:", "image_file_path")
            print("Label: Other")

        return label
