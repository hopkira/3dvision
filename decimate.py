#!/usr/bin/env python3

import cv2
import depthai
import numpy as np
import numpy.ma as ma
import skimage.measure as skim

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi', -1, 20.0, (640,400))

# Initialize body camera
device = depthai.Device('', False)
body_cam = device.create_pipeline(config={
    "streams": ["depth"],
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

old_frame = np.full((20,32),1.0, dtype = np.uint16)

# Processing loop
while True:
    data_packets = body_cam.get_available_data_packets()

    for packet in data_packets:
        if packet.stream_name == 'depth':
            frame = packet.getData()
            decimate = 20
            frame = frame/65535
            frame = skim.block_reduce(frame,(decimate,decimate),np.min)
            frame = (old_frame + frame) / 2.0
            old_frame = ma.masked_values(frame, 65535)
            frame = frame*65535
            frame = (65535 // frame).astype(np.uint8)
            img_scaled = cv2.resize(frame, None, fx=decimate, fy=decimate, interpolation = cv2.INTER_NEAREST_EXACT)
            cv2.imshow('Depth', img_scaled)

    if cv2.waitKey(1) == ord('q'):
        break

del body_cam
del device