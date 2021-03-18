#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import cv2
import json
import math
import depthai
import numpy as np
import pandas as pd
import skimage.measure as skim
import logo

sys.path.append('/home/pi/k9-chess-angular/python') 

device = depthai.Device('', False)

config={
    "streams": ["depth","metaout"],
    "ai": {
        "blob_file": "/home/pi/3dvision//mobilenet-ssd/mobilenet-ssd.blob",
        "blob_file_config": "/home/pi/3dvision/mobilenet-ssd/mobilenet-ssd.json",
        "calc_dist_to_bb": True,
        "camera_input": "right"
    },
    "camera": {
        "mono": {
            # 1280x720, 1280x800, 640x400 (binning enabled)
            # reducing resolution decreases min depth as
            # relative disparity is decreased
            'resolution_h': 400,
            'fps': 10   
        }
    }
}

body_cam = device.create_pipeline(config=config)

# Retrieve model class labels from model config file.
model_config_file = config["ai"]["blob_file_config"]
mcf = open(model_config_file)
model_config_dict = json.load(mcf)
labels = model_config_dict["mappings"]["labels"]
print(labels)

if body_cam is None:
    raise RuntimeError("Error initializing body camera")

nn2depth = device.get_nn_to_depth_bbox_mapping()

def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

detections = []

disparity_confidence_threshold = 130

def on_trackbar_change(value):
    device.send_disparity_confidence_threshold(value)
    return

cv2.namedWindow('depth')
trackbar_name = 'Disparity confidence'
conf_thr_slider_min = 0
conf_thr_slider_max = 255
cv2.createTrackbar(trackbar_name, 'depth', conf_thr_slider_min, conf_thr_slider_max, on_trackbar_change)
cv2.setTrackbarPos(trackbar_name, 'depth', disparity_confidence_threshold)

decimate =20
max_dist = 4000.0
height = 400.0
width = 640.0
cx = width/decimate/2
cy = height/decimate/2
fx = 1.4 # values found by measuring known sized objects at known distances
fy = 2.05

prev_frame = 0
now_frame = 0

x_bins = pd.interval_range(start = -2000, end = 2000, periods = 40)
y_bins = pd.interval_range(start= 0, end = 800, periods = 8)

while True: # main loop until 'q' is pressed

    nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())
        for detection in detections:
            if detection.label == 5: # we're looking for a bottle...
                pass
                #print('Bottle is ' + '{:.2f}'.format(detection.depth_z) + 'm away.')
                #angle = (math.pi / 2) - math.atan2(detection.depth_z, detection.depth_x)
                #print('Bottle is at ' + '{:.2f}'.format(angle) + ' radians.')

    for packet in data_packets:
        if packet.stream_name == 'depth':
            frame = packet.getData()
            # create a specific frame for display
            image_frame = np.copy(frame)
            cv2.putText(image_frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255))
            image_frame = (65535 // image_frame).astype(np.uint8)
            # colorize depth map
            image_frame = cv2.applyColorMap(image_frame, cv2.COLORMAP_HOT)
            if detections is not None:
                for detection in detections:
                    if (detection.label != 15 or detection.depth_z < 0.5 or detection.depth_z > 1.5):
                        detections.remove(detection)
            if detections is not None:
                x_min_sum = 0
                x_max_sum = 0
                y_min_sum = 0
                y_max_sum = 0
                z_sum = 0
                num_boxes = len(detections)
                for detection in detections:
                    x_min_sum = x_min_sum + detection.x_min
                    x_max_sum = x_max_sum + detection.x_max
                    y_min_sum = y_min_sum + detection.y_min
                    y_max_sum = y_max_sum + detection.y_max
                    z_sum = z_sum + detection.depth_z
                x_min_avg = x_min_sum / num_boxes
                x_max_avg = x_max_sum / num_boxes
                y_min_avg = y_min_sum / num_boxes
                y_max_avg = y_max_sum / num_boxes
                z_avg = z_sum / num_boxes
                # convert the resulting box into a bounding box
                pt1 = nn_to_depth_coord(x_min_avg, y_min_avg, nn2depth)
                pt2 = nn_to_depth_coord(x_max_avg, y_max_avg, nn2depth)
                color = (255, 255, 255) # bgr white
                label = labels[int(detection.label)]                 
                score = int(detection.confidence * 100)  
                cv2.rectangle(image_frame, pt1, pt2, color)
                cv2.putText(image_frame, str(score) + ' ' + label,(pt1[0] + 2, pt1[1] + 15),cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 2)  
                x_1, y_1 = pt1
                pt_t1 = x_1 + 5, y_1 + 60
                angle = ( math.pi / 2 ) - math.atan2(detection.depth_z, detection.depth_x)
                cv2.putText(image_frame, 'x:' '{:7.2f}'.format(detection.depth_x) + ' m', pt_t1, cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
                pt_t2 = x_1 + 5, y_1 + 80
                cv2.putText(image_frame, 'y:' '{:7.2f}'.format(detection.depth_y) + ' m', pt_t2, cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
                pt_t3 = x_1 + 5, y_1 + 100
                cv2.putText(image_frame, 'z:' '{:7.2f}'.format(detection.depth_z) + ' m', pt_t3, cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
                pt_t4 = x_1 + 5, y_1 + 120
                cv2.putText(image_frame, 'angle: ' '{:2.4f}'.format(angle) + ' radians', pt_t4, cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
                now_frame = time.time()
                fps = 1/(now_frame - prev_frame) 
                prev_frame = now_frame
                fps = str(int(fps))
                pt_t5 = x_1 + 5, y_1 + 140
                cv2.putText(image_frame, 'fps: ' + fps, pt_t5, cv2.FONT_HERSHEY_DUPLEX, 0.5, color)
                if (angle > 0.04) :
                    logo.right(abs(angle))
                if (angle < -0.04) :
                    logo.left(abs(angle))
        
            cv2.imshow('depth', image_frame)

            # Process depth map to communicate to robot
            frame = skim.block_reduce(frame,(decimate,decimate),np.min)
            height, width = frame.shape
            # Convert depth map to point cloud with valid depths
            column, row = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
            valid = (frame > 200) & (frame < max_dist)
            z = np.where(valid, frame, 0)
            x = np.where(valid, (z * (column - cx) /cx / fx) + 120 , max_dist)
            y = np.where(valid, 325 - (z * (row - cy) / cy / fy) , max_dist)
            # Flatten point cloud axes
            z2 = z.flatten()
            x2 = x.flatten()
            y2 = y.flatten()
            # Stack the x, y and z co-ordinates into a single 2D array
            cloud = np.column_stack((x2,y2,z2))
            # Filter the array by x and y co-ordinates
            in_scope = (cloud[:,1]<800) & (cloud[:,1] > 0) & (cloud[:,0]<2000) & (cloud[:,0] > -2000)
            in_scope = np.repeat(in_scope, 3)
            in_scope = in_scope.reshape(-1, 3)
            scope = np.where(in_scope, cloud, np.nan)
            # Remove invalid rows from array
            scope = scope[~np.isnan(scope).any(axis=1)]
            # Index each point into 10cm x and y bins (40 x 8)
            x_index = pd.cut(scope[:,0], x_bins)
            y_index = pd.cut(scope[:,1], y_bins)
            # Place the depth values into the corresponding bin
            binned_depths = pd.Series(scope[:,2])
            # Average the depth measures in each bin
            totals = binned_depths.groupby([y_index, x_index]).mean()
            # Reshape the bins into a 8 x 40 matrix
            totals = totals.values.reshape(8,40)
            # Determine the nearest segment for each of the 40
            # horizontal segments
            closest = np.amin(totals, axis = 0 )
            # Round the to the nearest 10cm
            closest = np.around(closest,-2)
            # Turn into a 1D array
            closest = closest.reshape(1,-1)
            # print(closest)
    
    if cv2.waitKey(1) == ord('q'):
        break

del body_cam
del device