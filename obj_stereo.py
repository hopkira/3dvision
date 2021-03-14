from pathlib import Path
import numpy as np
import cv2      # opencv - display the video stream
import depthai  # access the camera and its data packets
import json

device = depthai.Device('', False)

config={'streams': ['previewout','metaout','depth'],
        'ai': {"blob_file":        str(Path('./mobilenet-ssd/mobilenet-ssd.blob').resolve().absolute()),
               "blob_file_config": str(Path('./mobilenet-ssd/mobilenet-ssd.json').resolve().absolute()),
               "calc_dist_to_bb": True,
               "camera_input": "right"}
        }
# Create the pipeline using the 'previewout, metaout & depth' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config=config)

# Retrieve model class labels from model config file.
model_config_file = config["ai"]["blob_file_config"]
mcf = open(model_config_file)
model_config_dict = json.load(mcf)
labels = model_config_dict["mappings"]["labels"]
print(labels)

if pipeline is None:   
    raise RuntimeError('Pipeline creation failed!')

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

stream_windows = ['depth']

for stream in stream_windows:
    if stream in ["disparity", "disparity_color", "depth"]:
        cv2.namedWindow(stream)
        trackbar_name = 'Disparity confidence'
        conf_thr_slider_min = 0
        conf_thr_slider_max = 255
        cv2.createTrackbar(trackbar_name, stream, conf_thr_slider_min, conf_thr_slider_max, on_trackbar_change)
        cv2.setTrackbarPos(trackbar_name, stream, disparity_confidence_threshold)
    
while True:    # Retrieve data packets from the device.   # A data packet contains the video frame data.    
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:

        if packet.stream_name == 'previewout':  
            meta = packet.getMetadata()
            camera = meta.getCameraName()
            window_name = 'previewout-' + camera
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)            
            data0 = data[0, :, :]            
            data1 = data[1, :, :]            
            data2 = data[2, :, :]           
            frame = cv2.merge([data0, data1, data2])            

            img_h = frame.shape[0]            
            img_w = frame.shape[1]            

            for detection in detections:                 
                pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)                              
                label = labels[int(detection.label)]                 
                score = int(detection.confidence * 100)   
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)    
                cv2.putText(frame, str(score) + ' ' + label,(pt1[0] + 2, pt1[1] + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
                color = (0, 0, 255)
                x1, y1 = pt1
                x2, y2 = pt2

            cv2.imshow(window_name, frame)      

        elif packet.stream_name == 'depth':  # Only process `depth`.
            window_name = packet.stream_name
            frame = packet.getData()  
            cv2.putText(frame, packet.stream_name, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            frame = (65535 // frame).astype(np.uint8)
            # colorize depth map
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

            if detections is not None:
                for detection in detections:  
                    pt1 = nn_to_depth_coord(detection.x_min, detection.y_min, nn2depth)
                    pt2 = nn_to_depth_coord(detection.x_max, detection.y_max, nn2depth)
                    color = (255, 255, 255) # bgr
                    label = labels[int(detection.label)]                 
                    score = int(detection.confidence * 100)  
                    cv2.rectangle(frame, pt1, pt2, color)
                    cv2.putText(frame, str(score) + ' ' + label,(pt1[0] + 2, pt1[1] + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
                    color1 = (0, 255, 0)
                    x1, y1 = pt1
                    #x2, y2 = pt2
                    pt_t3 = x1 + 5, y1 + 60
                    cv2.putText(frame, 'x:' '{:7.3f}'.format(detection.depth_x) + ' m', pt_t3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1)

                    pt_t4 = x1 + 5, y1 + 80
                    cv2.putText(frame, 'y:' '{:7.3f}'.format(detection.depth_y) + ' m', pt_t4, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1)

                    pt_t5 = x1 + 5, y1 + 100
                    cv2.putText(frame, 'z:' '{:7.3f}'.format(detection.depth_z) + ' m', pt_t5, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1)

            cv2.imshow(window_name, frame)            

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline 
del device            



