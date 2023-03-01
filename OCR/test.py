import roboflow
import cv2
from keras.models import load_model
import tensorflow as tf
from skimage import io


# API_KEY = "0mhotLhKSdkllDDnaBCA"
# PROJECT_ID = "text-message-detector"
#
# rf = roboflow.Roboflow(api_key=API_KEY)
#
# workspace = rf.workspace()
# project = rf.project(PROJECT_ID)
# project.versions()
#
# model = project.version("2").model

model = cv2.dnn.readNetFromONNX('/home/iduadmin/Projects/train-model/runs/train/exp9/weights/epoch_696.onnx')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)





#
# results = model.predict("/home/iduadmin/Downloads/telegram_chat.png", confidence=40, overlap=30).json()
# img = cv2.imread("/home/iduadmin/Downloads/telegram_chat.png")
# for i in results["predictions"]:
#     x = i["x"]
#     y = i["y"]
#     width = i["width"]
#     height = i["height"]
#
#     print(f"""
#     x: {x}
#     y: {y}
#     width: {width}
#     height: {height}""")
#
#     x0 = x - width / 2
#     x1 = x + width / 2
#     y0 = y - height / 2
#     y1 = y + height / 2
#
#     start_point = (int(x0), int(y0))
#     end_point = (int(x1), int(y1))
#     cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)
#
#
#
# cv2.imwrite("img-with-bbox.jpg", img)
#
#
