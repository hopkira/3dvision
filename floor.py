#!/usr/bin/env python3

import sys
import cv2
import depthai
import numpy as np
import numpy.ma as ma
import skimage.measure as skim

# Initialize body camera
device = depthai.Device('', False)
body_cam = device.create_pipeline(config={
    "streams": ["metaout", "depth"],
    "ai": {
        "blob_file": "/home/pi/depthai/resources/nn/face-detection-retail-0004/face-detection-retail-0004.blob.sh14cmx14NCE1",
        "blob_file_config": "/home/pi/depthai/resources/nn/face-detection-retail-0004/face-detection-retail-0004.json",
        'shaves' : 14,
        'cmx_slices' : 14,
        'NN_engines' : 1,
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
})

if body_cam is None:
    raise RuntimeError("Error initializing body camera")

detections = []

floor_slices = []

old_frame = np.full((20,32),1.0, dtype = np.uint16)

# Processing loop
while True:
    nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()

    # retrieve detections from neural net
    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:
        if packet.stream_name == 'depth':
            frame = packet.getData()
            #print(np.amax(frame))
            #print(np.amin(frame))
            #print(data.shape)
            #print(np.amax(data))
            #print(np.amin(data))
            #data0 = data[0, :, :]
            #data1 = data[1, :, :]
            #frame = cv2.merge([data0, data1])
            #floor_slice = frame[200:400,320:360]
            #view_top = np.full((200,640),65535, dtype=np.uint16)
            # print(view_top.shape)
            #floor = ma.masked_values(floor_slice, 65535)
            #mean = floor.mean(axis = 1)
            #mean = np.round(mean,0)
            #view_bottom = np.transpose([mean]*640)
            # print(view_bottom.shape)
            #view = np.concatenate((view_top, view_bottom))
            #frame = view

            #frame = frame.clip(min=1)

            decimate = 20
            frame = frame/65535

            frame = skim.block_reduce(frame,(decimate,decimate),np.min)

            frame = (old_frame + frame) / 2.0
            
            old_frame = frame

            frame = frame*65535

            frame = (65535 // frame).astype(np.uint8)

            img_h = frame.shape[0] # image height
            img_w = frame.shape[1] # image width

            # draw bounding boxes
            for detection in detections:
                pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)

                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            
            img_scaled = cv2.resize(frame, None, fx=decimate*2, fy=decimate*2, interpolation = cv2.INTER_NEAREST_EXACT)

            cv2.imshow('Depth', img_scaled)

    if cv2.waitKey(1) == ord('q'):
        break

del body_cam
del device
