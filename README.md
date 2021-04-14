# 3dvision

## 3dvision.py
Simple program that uses the OAK-D stereoscopic camera and the mobile-net SSD model to detect a person's legs and follow them around the room. Simultaneously, it creates a point cloud that it simplifies into a one dimensional array that tells the robot how close obstacles are that are in front of it.  This is work on progress - do not use near small children or animals or in any place where a robot might do damage!

Currently runs at about 10 FPS on a Raspberry Pi 4.

Code will be fully described once complete, but partial descriptions are here:
* Cleaning up the raw depth map (https://k9-build.blogspot.com/2021/01/oak-d-point-cloud-on-pi-part-1.html)
* Generating a point cloud using numpy (https://k9-build.blogspot.com/2021/02/oak-d-point-cloud-on-pi-part-2.html)
* Enabling object detection (https://k9-build.blogspot.com/2021/03/3d-object-detection-with-point-cloud-at.html)

The program uses logo.py to make the robot move.

## logo.py
A simple program that emulates an old fashioned logo turtle and drives the robot using a differential drive (in this case via a RoboClaw motor controller with homemade encoders)
* Explanation of program (https://k9-build.blogspot.com/2018/02/when-is-dog-turtle.html)
* Printing your own encoder (https://k9-build.blogspot.com/2019/07/now-you-see-it-now-you-dont.html)
* System integration test (https://k9-build.blogspot.com/2020/01/by-my-dead-reckoning.html)
* Modifying robot logo routine to cope with on the fly changes (https://k9-build.blogspot.com/2021/04/calculating-optimal-speed-for-moving.html)

## requirement.txt
A list of the Python libraries you will need to get the 3d vision program to run. I'd advise creating a Python virtual environment and "running python3 -m pip install -r requirements.txt"
