# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:51:35 2018

@author: Mark Anthony R. Sulleza

"""

"""
Autonomous Indoor Navigation of Micro Aerial Vehicles
CMSC 190-2
University of the Philippines Los Baños

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
    The following code enables Micro Aerial Vehicles, a Parrot AR Drone 2.0 more specifically, to be
    able to navigate indoor environments using an InceptionV3 model and avoid detected objects uing a 
    Mobilenet SSD model.

"""

import numpy as np
import tensorflow as tf
import cv2 as cv
import time
import datetime
import threading
import ps_drone
import sys
from collections import defaultdict
from object_detection.utils import label_map_util


# import the necessary packages
import datetime
 

PATH_TO_OD_LABELS = 'mscoco_label_map.pbtxt'
PATH_TO_INCEPTION_LABELS = 'output_labels.txt'

# Maximum number of classes for the object detection
NUM_CLASSES = 80


font = cv.FONT_HERSHEY_SIMPLEX


# Loading of Object Detection Label Map

label_map = label_map_util.load_labelmap(PATH_TO_OD_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# function that will load the specified label file
def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()

    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

# Loading of Inception Label
labels = load_labels(PATH_TO_INCEPTION_LABELS)

# Loading of Inception Graph
inception_graph = tf.Graph()
with inception_graph.as_default():
    inception_graph_def = tf.GraphDef()
    with tf.gfile.GFile('output_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        inception_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(inception_graph_def, name='')

# Loading of Object Detection Graph
object_detection_graph = tf.Graph()
with object_detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

tensor_dict = {}
# Read and preprocess an image.
drone = ps_drone.Drone()                                     # Start using drone
drone.startup()                                              # Connects to drone and starts subprocesses

drone.reset()                                                # Sets drone's status to good (LEDs turn green when red)
while (drone.getBattery()[0] == -1):      time.sleep(0.1)    # Waits until drone has done its reset
print ("Battery: "+str(drone.getBattery()[0])+"%  "+str(drone.getBattery()[1]))	# Gives a battery-status
drone.useDemoMode(False)                                      # Set 15 basic dataset/sec (default anyway)
drone.getNDpackage(["demo"])                                 # Packets, which shall be decoded
time.sleep(0.5)                                              # Give it some time to awake fully

drone.trim()                                                 # Recalibrate sensors
print ("Auto-alt.:"+str(drone.selfRotation)+"dec/s")         # Showing value for auto-alteration


# ##### Mainprogram begin #####
drone.setConfigAllID()                                       # Go to multiconfiguration-mode
drone.sdVideo()                                              # Choose lower resolution
drone.frontCam()                                             # Choose front view
drone.fastVideo()
CDC = drone.ConfigDataCount
while CDC == drone.ConfigDataCount:       time.sleep(0.0001) # Wait until it is done (after resync is done)
drone.startVideo()                                           # Start video-function

IMC =  drone.VideoImageCount
while drone.VideoImageCount == IMC: time.sleep(0.01)     # Wait until the next video-frame
ok = True

start_time = time.time()
frames_count = 0

current_frames = []

# Creation of two separate sessions for the models
inception_session = tf.Session(graph = inception_graph)
object_detection_session = tf.Session(graph = object_detection_graph)


"""
===========================!!!WARNING!!!=============================
DO NOT FORGET TO COMMENT THE FOLLOWING SNIPPET IF YOU DON'T WANT YOUR
DRONE TO FLY AND GET REKT.
"""

""" ================= START OF CRUCIAL SNIPPET ==================== """

drone.takeoff()
while drone.NavData["demo"][0][2]: time.sleep(0.1) # Wait until drone is completely flying


""" =================== END OF CRUCIAL SNIPPET ==================== """

while (True):
    
    
    img = (drone.VideoImage)
    if (img is None): continue

    # image adjustmest to be able to feed to the inception model
    img2 =  cv.resize(img, dsize=(299, 299), interpolation = cv.INTER_CUBIC)
    np_img_data = np.asarray(img2)
    np_img_data = cv.normalize(np_img_data.astype('float'), None, -0.5, .5, cv.NORM_MINMAX)
    np_final = np.expand_dims(np_img_data, axis = 0)

    img = np.array(img)
    rows = img.shape[0]
    cols = img.shape[1]

    inp = cv.resize(img, (299, 299))
    
    # Prediction of current frame using the inception model
    softmax_tensor = inception_session.graph.get_tensor_by_name('final_result:0')
    results = inception_session.run(softmax_tensor, {'Placeholder:0' : np_final})
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    inception_decision = "None"
    
    for i in top_k:
        # Threshold for when the drone will make a decision
        if (results[i] >= 0.4):
            print(labels[i], results[i])
            inception_decision = labels[i]
        # breaks to get the first result
        break

    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Note: Running the two model resulted with a maximum framerate of 7 FPS.
    # MobileNet will cause the model to run at much a much faster rate.
    # Run the model
    out = object_detection_session.run([object_detection_graph.get_tensor_by_name('num_detections:0'),
                    object_detection_graph.get_tensor_by_name('detection_scores:0'),
                    object_detection_graph.get_tensor_by_name('detection_boxes:0'),
                    object_detection_graph.get_tensor_by_name('detection_classes:0')],
                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # # Visualize detected bounding boxes.
    num_detections = int(out[0][0])

    # ## out[3] index of the classifications of each detected objects per frame

    prox_warn = False
    num_of_prox_warn = 0
    evasive = "hover"
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        label = 'unknown'

        for i in category_index:
            if (category_index[i]["id"] == classId):
                label = category_index[i]["name"]

        # increase the score value for a more accurate result
        if score > 0.5 and classId == 1:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)


            # prints the current values of the points
            # print ("Bounding Box Values: ", bbox)
            # computing for the relative distance of the tracaked object:
            # round the value to one to get the percentage.
            # note: the values of the coordinates are relative to their distance from the origin.

            # parameters for the bbox [ymin, xmin, ymax, xmax]
            relative_distance = round ((1- (bbox[3] - bbox[1])), 1)
            print("distance: " + str(relative_distance))
            # decreasing value means that the object is "theoretically" getting closer to the screen

            # then get the midpoint of the detected objects
            object_mid_x = (bbox[3] + bbox[1]) / 2
            object_mid_y = (bbox[2] + bbox[0]) / 2

            
            # then display the d]]stance of the object from the camera
            # note: color format is BGR instead of RGB
            cv.putText(img, label, (int(object_mid_x * cols) , int(object_mid_y * rows + 50)), font, 0.7, (0, 255, 0), 2, cv.LINE_AA)

            if (relative_distance <= 0.70):
                if (object_mid_x > 0.3 and object_mid_x < 0.7):
                    cv.putText(img, "CAUTION!", (int(object_mid_x * cols) , int(object_mid_y * rows)), font, 0.7, (0, 0, 255), 2, cv.LINE_AA)
                    prox_warn = True
                    num_of_prox_warn += 1
            else:
                cv.putText(img, str(relative_distance), (int(object_mid_x * cols) , int(object_mid_y * rows)), font, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            print("mid point: ", object_mid_x)
            # Checking for the location of the object relative to the midpoint of the FOV.
            # 0.5 is the center width of the FOV
            
            # the object is one the right side
            if (object_mid_x > 0.4):
                # drone.moveLeft(0.25)
                evasive = "moveLeft"
            elif (object_mid_x < 0.7):
                # drone.moveRight(0.25)
                evasive = "moveRight"
            if (object_mid_x > 0.3 and object_mid_x < 0.6):
                # drone.stop()
                evasive = "stop"

    
    ## ========================= DRONE CONTROLS ==================== ##
    if (not prox_warn):
        if (inception_decision == "moveleft"):
            print("========== InceptionV3 Decision ==============")
            print("moving left...")
            drone.moveLeft(0.15)
        elif (inception_decision == "moveright"):
            print("========== InceptionV3 Decision ==============")
            print("moving right...")
            drone.moveRight(0.15)
        elif (inception_decision == "spinleft"):
            print("========== InceptionV3 Decision ==============")
            print("spinning left...")
            drone.turnLeft(0.15)
        elif (inception_decision == "spinright"):
            print("========== InceptionV3 Decision ==============")
            print("spinning right...")
            drone.turnRight(0.15)
        elif (inception_decision == "moveforward"):
            print("========== InceptionV3 Decision ==============")
            print("moving forward...")
            drone.moveForward(0.10)
    elif (prox_warn and num_of_prox_warn == 1):
        print("======= Mobilenet Decision =========")
        if (evasive == "moveLeft"):
            print("moving left...")
            drone.moveLeft(0.25)
        elif (evasive == "moveRight"):
            print("moving right...")
            drone.moveRight(0.25)
        elif (evasive == "stop"):
            print("stopping...")
            drone.stop()


    elif(prox_warn and num_of_prox_warn > 1):
        print("")
        # if the drone detects multiple incoming objects, hover in place
        drone.stop()

    
    # Prints the number of frames per second.
    if (start_time - time.time() <= -1):
        start_time = time.time()
        cv.putText(img, str(frames_count), (75, 402), font, 2, (255, 0, 0), 2, cv.LINE_AA)
        frames_count = 0
    else:
        frames_count += 1
    
    cv.imshow('Autonomous Indoor Navigation of Micro Aerial Vehicles using InceptionV3 and Mobilened SSD', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        # stops the program and lands the drone
        drone.land()
        drone.shutdown()
        break