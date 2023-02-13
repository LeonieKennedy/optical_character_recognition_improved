# file to be deleted once complete

from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import clip
import torch
import os
import shutil
from pydantic import BaseModel




class ClassifyImage(BaseModel):
    source_file: str
    plate_detected: bool
    text: str
    detection_time: float


class ClassifyImageModel:
    def __init__(self, model, processor):
        self.labels = ["car", "document", "texting"]
        self.threshold = 0.8

        if model == None:
            # self.model, self.processor = clip.load("ViT-B/32")
            # torch.save(self.model, "./models/image_classifier_model.pt")
            self.model = torch.load("./models/image_classifier_model.pt")

            # torch.save(self.processor, "./models/image_classifier_processor.pt")
            self.processor = torch.load("./models/image_classifier_processor.pt")

            # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            # torch.save(self.model, "./models/image_classifier_model.pt")

            # self.model.save_model("./models/image_classifier_model.h5")
            # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")            
            # self.processor.save_model("./models/image_classifier_processor.h5")
            # torch.save(self.processor, "./models/image_classifier_processor.pt")

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
            if i > self.threshold: # if the label confidence is above the threshold(80%), it is that label
                label = self.labels[count]
                print("File name:", image_file_path)
                print("Label:", self.labels[count])
                found = True
            count = count + 1

        if found == False:
            label = "other"
            print("File name:", image_file_path)
            print("Label: Other")
        
        # shutil.move(path_to_images +file_name, "../Task3_images/"+label+"/" + file_name)
        # print("Prob:", probs)

        return label

# ClassifyImageModel(model=None, processor=None)