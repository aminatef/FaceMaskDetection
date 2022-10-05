from cProfile import label
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot
import numpy as np
import cv2
import os
import argparse

def detect(frame,faceNet,maskNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs  = []
    preds = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > args["confidence"]:
            bbox = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY) = bbox.astype("int")
            (startX,startY) = (max(0,startX),max(0,startY))
            (endX,endY) = (min(w-1,endX),min(h-1,endY))
            face = frame[startY:endY,startX:endX]
            face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face = cv2.resize(face,(224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

            if(len(faces)>0):
                faces = np.array(faces, dtype="float32")
                preds = maskNet.predict(faces, batch_size=32)
            return locs,preds
def show_prediction(frame,locs,preds):
    for(box,pred) in zip(locs,preds):
        (startX,startY,endX,endY) = box
        (mask,without)  = pred
        label ="Mask" if mask > without else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, without) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imwrite("./test.jpg",frame)



ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True)
ap.add_argument("-f", "--face", type=str,
	default="faceDetector")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model")
ap.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"],"deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],"res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath,weightsPath)
maskNet = load_model(args["model"])
image = cv2.imread(args["image"])
originalImage = image.copy()
(locs, preds) = detect(image, faceNet, maskNet)
show_prediction(image,locs,preds)
