#!/usr/bin/env python3
# coding: utf-8

#!/usr/bin/env python3
"""K9 Asssistant Following State Machine

Args:
    -a, --max (float): Maximum distance to follow
    -i, --min (float): Minimum distance to follow
    -s, --safe (float): Addiitional safety margin
    -c, --conf (float): Confidence level

Example:
    $ python3 3dvision.py -a 2.0 -i 0.2 -s 0.1 -c 0.75

Todo:
    * stuff

K9 word marks and logos are trade marks of the British Broadcasting Corporation and
are copyright BBC 1977-2021

K9 was created by Bob Baker and David Martin
"""
import argparse
import sys
import time
import os
import json
import math
import depthai
import numpy as np
import pandas as pd
import skimage.measure as skim
import paho.mqtt.client as mqtt
import logo # K9 movement library
from operator import attrgetter

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--max", type=float, default=2.0,
	help="Maximum distance")
ap.add_argument("-i", "--min", type=float, default=0.5,
	help="Minimium distance")
ap.add_argument("-s", "--safe", type=float, default=0.5,
	help="Safe distance")
ap.add_argument("-c", "--conf", type=float, default=0.85,
	help="Confidence")
args = vars(ap.parse_args())

print(args)

MAX_DIST = args['max']
MIN_DIST = args['min']
CONF = args['conf']
SAFETY_MARGIN = args['safe']
S_SPOT_BUFFR = (MAX_DIST - MIN_DIST - SAFETY_MARGIN) / 2.0
SWEET_SPOT = S_SPOT_BUFFR + MIN_DIST + SAFETY_MARGIN

print("Sweet spot is",SWEET_SPOT,"meters from robot")

# These values control K9s voice
SPEED_DEFAULT = 150
SPEED_DOWN = 125
AMP_UP = 200
AMP_DEFAULT = 190
AMP_DOWN = 180
PITCH_DEFAULT = 99
PITCH_DOWN = 89
SOX_VOL_UP = 25
SOX_VOL_DEFAULT = 20
SOX_VOL_DOWN = 15
SOX_PITCH_UP = 100
SOX_PITCH_DEFAULT = 0
SOX_PITCH_DOWN = -100

detections = []
angle = 0.0
last_seen = 0.05

disparity_confidence_threshold = 130

sys.path.append('/home/pi/k9-chess-angular/python') 

device = depthai.Device('', False)

config={
    "streams": ["depth","metaout"],
    "ai": {
        "blob_file": "/home/pi/3dvision/mobilenet-ssd/mobilenet-ssd.blob",
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
angle = 0.0
last_seen = 0.05
MIN_DIST = 0.5
MAX_DIST = 3.0
CONF = 0.85
SAFETY_MARGIN = 0.5

decimate = 20
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
y_bins = pd.interval_range(start = 0, end = 800, periods = 8)


class State(object):
    '''
    State parent class to support standard Python functions
    '''

    def __init__(self):
        print('Entering state:', str(self))

    def on_event(self, event):
        '''
        Incoming events processing is delegated to the child State
        to define and enable the valid state transitions.
        '''

    def run(self):
        '''
        Enable the state to do something - this is usually delegated
        to the child States)
        '''
        print('Run event for ' + str(self) + ' state not implemented')

    def __repr__(self):
        '''
        Leverages the __str__ method to describe the State.
        '''
        return self.__str__()

    def __str__(self):
        '''
        Returns the name of the State.
        '''
        return self.__class__.__name__

# Declare the basic K9 operational states
class Initializing(State):
    '''
    The child state where K9 is waiting and appears dormant
    '''

    def __init__(self):
        super(Initializing, self).__init__()

    def run(self):
        # Waits for a command from Espruino Watch
        k9.client.loop(0.1)

    def on_event(self, event):
        # Various events that can come from the watch...
        print("Event: " + event)
        if event == "k9mwakon":
            return Awake()
        return self


class Asleep(State):
    '''
    The child state where K9 is appears dormant
    '''

    def __init__(self):
        super(Asleep, self).__init__()
        # turn all lights off
        logo.stop()
        k9.speak("Conserving battery power")

    def run(self):
        # Waits for a command from Espruino Watch
        k9.client.loop(0.1)

    def on_event(self, event):
        if event == 'k9mwakon':
            return Awake()
        return self


class Awake(State):
    '''
    The child state where K9 is waiting and appears dormant
    '''

    def __init__(self):
        super(Awake, self).__init__()
        # turn on lights
        logo.stop()
        k9.speak("K9 operational")

    def run(self):
        k9.client.loop(0.1)

    def on_event(self, event):
        if event == 'chefolon':
            return Scanning()
        if event == 'k9mwakoff':
            return Asleep()
        return self


class Scanning(State):
    '''
    The child state where K9 is looking for the nearest person to follow
    '''

    def __init__(self):
        super(Scanning, self).__init__()
        print('Waiting for the closest person to be detected...')
        # k9.speak("Finding person to follow")
        k9.target = None

    def run(self):
        # Checks for the nearest person in the field of vision
        k9.client.loop(0.1)
        nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()
        for nnet_packet in nnet_packets:
            detections = list(nnet_packet.getDetectedObjects())
            if detections is not None:
                people = [detection for detection in detections
                            if detection.label == 15
                            if detection.depth_z > MIN_DIST
                            if detection.depth_z < MAX_DIST
                            if detection.confidence > CONF]
                if len(people) >= 1 :
                    k9.target = min(people, key=attrgetter('depth_z'))
                    k9.on_event('person_found')
        if k9.target is None and not logo.finished_move():
            logo.stop()

    def on_event(self, event):
        if event == 'person_found':
            return Turning()
        if event == 'chefoloff':
            return Awake()
        return self


class Turning(State):
    '''
    The child state where K9 is turning towards the target person
    '''

    def __init__(self):
        super(Turning, self).__init__()
        z = float(k9.target.depth_z)
        x = float(k9.target.depth_x)
        angle = ( math.pi / 2 ) - math.atan2(z, x)
        if abs(angle) > 0.2 :
            print("Moving ",angle," radians towards target")
            logo.right(angle)
        else:
            k9.on_event('turn_finished')

    def run(self):
        k9.client.loop(0.1)
        # Checks to see if motors have stopped
        if logo.finished_move():
            k9.on_event('turn_finished')

    def on_event(self, event):
        if event == 'chefoloff':
            return Awake()
        if event == 'turn_finished':
            return Moving_Forward()
        return self


class Moving_Forward(State):

    '''
    The child state where K9 is moving forwards to the target
    '''
    def __init__(self):
        super(Moving_Forward, self).__init__()
        z = float(k9.target.depth_z)
        distance = float(z - SWEET_SPOT)
        if distance > 0:
            print("Target is",z,"m away. Moving forward by",distance,"m")
            logo.forwards(distance)

    def run(self):
        k9.client.loop(0.1)
        # Wait until move finishes and return to target scanning
        # or detect that a collision is imminent and stop
        if logo.finished_move():
            k9.on_event('move_finished')
        # if check between the values of x and y is less than
        # SAFETY_MARGIN + MIN DIST, then stop
        check = k9.scan()
        try:
            min_dist = np.amin(check[17:25])
            if min_dist < SWEET_SPOT:
                k9.on_event('move_finished')
        except ValueError:
            pass

    def on_event(self, event):
        if event == 'chefoloff':
            return Awake()
        if event == 'move_finished':
            return Scanning()
        return self


class K9(object):
    '''
    A K9 finite state machine that starts in waiting state and
    will transition to a new state on when a transition event occurs.
    It also supports a run command to enable each state to have its
    own specific behaviours
    '''

    def __init__(self):
        ''' Initialise K9 in his waiting state. '''

        # Start with initializing actions
        self.state = Initializing()
        self.last_message = ""
        self.client = mqtt.Client("k9-python")
        self.client.connect("localhost")
        self.client.on_message = self.callback        # attach function to callback
        self.client.subscribe("/ble/advertise/watch/m")

    def run(self):
        ''' Run the behaviour of the current K9 state using its run function'''

        self.state.run()

    def on_event(self, event):
        '''
        Process the incoming event using the on_event function of the
        current K9 state.  This may result in a change of state.
        '''

        # The next state will be the result of the on_event function.
        print("State: " + str(self.state) + ", Event: " + event)
        self.state = self.state.on_event(event)

    def speak(self,speech):
        '''
        Break speech up into clauses and speak each one with
        various pitches, volumes and distortions
        to make the voice more John Leeson like
        '''
        
        print(speech)
        clauses = speech.split("|")
        for clause in clauses:
            if clause and not clause.isspace():
                if clause[:1] == ">":
                    clause = clause[1:]
                    pitch = PITCH_DEFAULT
                    speed = SPEED_DOWN
                    amplitude = AMP_UP
                    sox_vol = SOX_VOL_UP
                    sox_pitch = SOX_PITCH_UP
                elif clause[:1] == "<":
                    clause = clause[1:]
                    pitch = PITCH_DOWN
                    speed = SPEED_DOWN
                    amplitude = AMP_DOWN
                    sox_vol = SOX_VOL_DOWN
                    sox_pitch = SOX_PITCH_DOWN
                else:
                    pitch = PITCH_DEFAULT
                    speed = SPEED_DEFAULT
                    amplitude = AMP_DEFAULT
                    sox_vol = SOX_VOL_DEFAULT
                    sox_pitch = SOX_PITCH_DEFAULT
                cmd = "espeak -v en-rp '%s' -p %s -s %s -a %s -z" % (clause, pitch, speed, amplitude)
                os.system(cmd)

    def scan(self):
        '''
        Retrieve a 40 element array derived from the 3D camera
        '''
        nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()
        for packet in data_packets:
            if packet.stream_name == 'depth':
                frame = packet.getData()
                # create a specific frame for display
                image_frame = np.copy(frame)
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
                return closest

    def callback(self, client, userdata, message):
        """
        Enables K9 to receive a message from an Epruino Watch via
        MQTT over Bluetooth (BLE) to place it into active or inactive States
        """
        payload = str(message.payload.decode("utf-8"))
        if payload != self.last_message:
            self.last_message = payload
            event = payload[3:-1].lower()
            print("Event: ",str(event))
            self.on_event(event)

# Create the k9 finite state machine
k9 = K9()

try:
    while True:
        k9.run()
except KeyboardInterrupt:
    print("K9 halted by CTRL+C")
    del body_cam
    del device
    k9.client.loop_stop()
    sys.exit(0)
