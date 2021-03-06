#!/usr/bin/env python3
# coding: utf-8

import sys
import cv2
import json
import depthai
import numpy as np
import pandas as pd
import skimage.measure as skim
# import logo

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

decimate =20
max_dist = 4000.0
height = 400.0
width = 640.0
cx = width/decimate/2
cy = height/decimate/2
fx = 1.4 # values found by measuring known sized objects at known distances
fy = 2.05

x_bins = pd.interval_range(start = -2000, end = 2000, periods = 40)
y_bins = pd.interval_range(start= 0, end = 800, periods = 8)


while True: # main loop until 'q' is pressed

    nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())
        for detection in detections:
            if detection.label == 5: # we're looking for a bottle...
                print('Bottle is ' + '{:.2f}'.format(detection.depth_z) + 'm away.')

    for packet in data_packets:

        if packet.stream_name == 'depth':
            frame = packet.getData()
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
            print(closest)
    
    if cv2.waitKey(1) == ord('q'):
        break

del body_cam
del device