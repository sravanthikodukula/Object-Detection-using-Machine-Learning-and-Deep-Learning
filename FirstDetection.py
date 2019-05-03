# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:27:30 2019

@author: Kodukula Sravanthi

"""
 
from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "testimage.jpeg"), output_image_path=os.path.join(execution_path , "testimage.jpeg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

