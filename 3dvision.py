#!/usr/bin/env python3
# coding: utf-8

#!/usr/bin/env python3
"""K9 Asssistant Following State Machine
Args:
    -a, --max (float): Maximum distance to follow
    -i, --min (float): Minimum distance to follow
    -c, --conf (float): Confidence level
    --active :  Start in active mode
    --follow : Start in follow mode
Example:
    $ python3 3dvision.py -a 2.0 -i 0.2 -c 0.75 --active
Todo:
    * stuff
K9 word marks and logos are trade marks of the British Broadcasting Corporation and
are copyright BBC 1977-2021
K9 was created by Bob Baker and David Martin
"""
import argparse
import sys
import json
import math
import depthai
import numpy as np
import pandas as pd
import skimage.measure as skim
import paho.mqtt.client as mqtt
import logo # K9 movement library
from subprocess import Popen

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--max", type=float, default = 1.5,
	help="Maximum distance")
ap.add_argument("-i", "--min", type=float, default = 0.20,
	help="Minimium distance")
ap.add_argument("-c", "--conf", type=float, default = 0.70,
	help="Confidence")
ap.add_argument('--active', dest='active', action='store_true',
    help="Active mode")
ap.add_argument('--follow', dest='follow', action='store_true',
    help="Follow mode")
ap.set_defaults(active = False) 
ap.set_defaults(follow = False)     
args = vars(ap.parse_args())

MAX_DIST = args['max']
MIN_DIST = args['min']
CONF = args['conf']
SWEET_SPOT = MIN_DIST + (MAX_DIST - MIN_DIST) / 2.0

print("Sweet spot is",SWEET_SPOT,"meters from robot")

# These values control K9s voice
SPEED_DEFAULT = 150
SPEED_DOWN = 125
AMP_UP = 100
AMP_DEFAULT = 50
AMP_DOWN = 25
PITCH_DEFAULT = 99
PITCH_DOWN = 89
SOX_VOL_UP = 25
SOX_VOL_DEFAULT = 20
SOX_VOL_DOWN = 15
SOX_PITCH_UP = 100
SOX_PITCH_DEFAULT = 0
SOX_PITCH_DOWN = -100

JOY_SPEED = 0.03

detections = []
angle = 0.0

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

if body_cam is None:
    raise RuntimeError("Error initializing body camera")

nn2depth = device.get_nn_to_depth_bbox_mapping()

def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth

decimate = 20
MAX_RANGE = 4000.0
height = 400.0
width = 640.0
cx = width/decimate/2
cy = height/decimate/2
fx = 1.4 # values found by measuring known sized objects at known distances
fy = 2.05

prev_frame = 0
now_frame = 0

x_bins = pd.interval_range(start = -2000, end = 2000, periods = 40)
y_bins = pd.interval_range(start = 0, end = 1600, periods = 16)

# calculate the horizontal angle per bucket
h_bucket_fov = math.radians( 71.0 / 40.0)

class State(object):
    '''
    State parent class to support standard Python functions
    '''

    def __init__(self):
        print('entering',str(self).lower(),'state.')

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


# Declare the K9 operational states
class Initializing(State):
    '''
    The child state where K9 is waiting and appears dormant
    '''

    def __init__(self):
        super(Initializing, self).__init__()

    def run(self):
        # Waits for a command from Espruino Watch
        k9.client.loop(0.1)
        if args['active'] == True:
            k9.on_event('start_scan')
        if args['follow'] == True:
            k9.on_event('follow')

    def on_event(self, event):
        # Various events that can come from the watch...
        if event == "k9mwakon":
            return Awake()
        if event == "start_scan":
            return Scanning()
        if event == 'follow':
            return Following()
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
        if event == 'k9mrigsta':
            return Joystick()
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
        if event == 'k9mrigsta':
            return Joystick()
        return self


class Scanning(State):
    '''
    The child state where K9 is looking for the nearest person to follow
    '''

    def __init__(self):
        super(Scanning, self).__init__()
        k9.speak("Scanning")
        global started_scan
        #k9.started_scan = time.time()

    def run(self):
        k9.target = None
        k9.client.loop(0.1)
        k9.target = k9.person_scan()
        if k9.target is not None :
            k9.on_event('person_found')

    def on_event(self, event):
        if event == 'person_found':
            return Turning()
        if event == 'chefoloff':
            return Awake()
        if event == 'k9mrigsta':
            return Joystick()
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
            print("Turning: Moving ",angle," radians towards target")
            logo.right(angle)
        else:
            k9.on_event('turn_finished')

    def run(self):
        k9.client.loop(0.1)
        # Checks to see if motors have stopped
        #test = k9.person_scan()
        #if test is not None :
        #    k9.target = test
        #    k9.on_event('new_information')
        if logo.finished_move():
            k9.on_event('turn_finished')

    def on_event(self, event):
        #if event == 'new_information':
        #    return Turning()
        if event == 'chefoloff':
            return Awake()
        if event == 'turn_finished':
            return Moving_Forward()
        if event == 'k9mrigsta':
            return Joystick()
        return self


class Moving_Forward(State):

    '''
    The child state where K9 is moving forwards to the target
    '''
    def __init__(self):
        self.avg_dist = 4.0
        super(Moving_Forward, self).__init__()
        z = float(k9.target.depth_z)
        distance = float(z - SWEET_SPOT)
        if distance > 0:
            print("Moving Forward: target is",z,"m away. Moving",distance,"m")
            logo.forwards(distance)

    def run(self):
        k9.client.loop(0.1)
        # If robot is moving, then check for a 
        # potential collision (or a complete lack of
        # targets.  If nothiing to worry about then
        # check for a person in 
        # the field of view and adjust
        # if necessary
        if not logo.finished_move():
            pass
            # check for obstacles
            # DEBUG BELOW
            #depth_image = k9.scan()
            #print("Moving Forward: depth image:", depth_image[0].shape(), type(depth_image)) 
            #check = k9.point_cloud(depth_image[0])
            #print("Moving Forward: check:", check.shape()), type(check) 
            #if check is not None:
            #    min_dist = np.amin(check[17:25]) # narrow to robot width
            #    print("Min dist:", min_dist)
            #    # determine rolling average of distance to target
            #    self.avg_dist = (self.avg_dist + min_dist) / 2.0
            #    if self.avg_dist <= SWEET_SPOT:
            #        logo.stop()
            #        k9.on_event('target_reached')
            # DEBUG ABOVE
            #person_seen = k9.person_scan() # check for person
            #if person_seen is not None :
            #    k9.target = person_seen
            #    self.avg_dist = k9.target.depth_z
            #    z = float(k9.target.depth_z)
            #    x = float(k9.target.depth_x)
            #    angle = ( math.pi / 2 ) - math.atan2(z, x)
            #    if abs(angle) > 0.2 :
            #        k9.on_event('new_angle')
            #    else:
            #        k9.on_event('new_distance')
        else:
            k9.on_event('target_reached')

    def on_event(self, event):
        #if event == 'new_angle':
        #    return Turning()
        #if event == 'new_distance':
        #    return Moving_Forward()
        if event == 'chefoloff':
            return Awake()
        if event == 'target_reached':
            return Following()
        if event == 'k9mrigsta':
            return Joystick()
        if event == 'scan_again':
            return Scanning()
        return self


class Following(State):

    '''
    Having reached the target, now follow it blindly
    '''
    def __init__(self):
        super(Following, self).__init__()
        logo.stop()
        k9.speak("Mastah!")

    def run(self):
        # scan for things taller than 60 cm
        depth_image = k9.scan(min_range = 200.0, max_range = 1500.0,)
        if depth_image is not None:
            direction, distance = k9.follow_vector(depth_image, certainty=CONF)
            if distance is not None and direction is not None:
                distance = distance / 1000.0
                print("Following: direction:", direction, "distance:", distance)
                angle = direction * math.radians(77.0)
                move = (distance - SWEET_SPOT)
                print("Following: angle:", angle, "move:", move)
                damp_angle = 3.0
                damp_distance = 2.0
                if abs(angle) >= (0.1 * damp_angle) :
                    logo.rt(angle / damp_angle, fast = True)
                else:
                    if abs(move) >= (0.05 * damp_distance) :
                        logo.fd(move / damp_distance)
                        return

    def on_event(self, event):
        if event == 'chefoloff':
            return Awake()
        if event == 'k9mrigsta':
            return Joystick()
        return self


class Joystick(State):

    '''
    Receive manual movement commands from watch
    '''
    def __init__(self):
        super(Joystick, self).__init__()
        logo.stop()
        k9.speak("Under manual control")

    def run(self):
        k9.client.loop(0.1)

    def on_event(self, event):
        state = event[:3]
        direction = event[3:6]
        action = event[6:]
        if state != 'joy' or event == 'joyjoyoff':
            logo.stop()
            return Awake()
        if action != 'sta':
            logo.stop()
        else:
            if direction == 'top':
                logo.motor_speed(JOY_SPEED, JOY_SPEED)
            elif direction == 'mid':
                logo.motor_speed(-JOY_SPEED, -JOY_SPEED)
            elif direction == 'lef':
                logo.motor_speed(-JOY_SPEED / 4, JOY_SPEED / 4)
            elif direction == 'rig':
                logo.motor_speed(JOY_SPEED / 4, -JOY_SPEED / 4)
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
        self.client.on_message = self.mqtt_callback        # attach function to callback
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
        print(event, "raised in state", str(self.state).lower())
        self.state = self.state.on_event(event)

    def speak(self,speech):
        '''
        Break speech up into clauses and speak each one with
        various pitches, volumes and distortions
        to make the voice more John Leeson like
        '''
        
        print('speech:', speech)
        self.speaking = None
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
                #cmd = "espeak -v en-rp '%s' -p %s -s %s -a %s -z" % (clause, pitch, speed, amplitude)
                cmd = ['espeak','-v','en-rp',str(clause),'-p',str(pitch),'-s',str(speed),'-a',str(amplitude)]
                self.speaking = Popen(cmd)

    def person_scan(self):
        '''
        Returns detectd person nearest centre of field
        '''
        nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()
        for nnet_packet in nnet_packets:
            detections = list(nnet_packet.getDetectedObjects())
            if detections is not None :
                people = [detection for detection in detections
                            if detection.label == 15
                            if detection.confidence > CONF]
                if len(people) >= 1 :
                    min_angle = math.pi
                    for person in people:
                        z = float(person.depth_z)
                        x = float(person.depth_x)
                        angle = abs(( math.pi / 2 ) - math.atan2(z, x))
                        if angle < min_angle:
                            min_angle = angle
                            target = person
                    return target

    def scan(self, min_range = 500.0, max_range = 1200.0, decimate_level = 20, mean = True):
        '''
        Generate a simplified image of the depth image stream from the camera.  This image
        can be reduced in size by using the decimate_level parameter.  
        It also will remove invalid data from the image (too close or too near pixels)
        The mechanism to determine the returned value of each new pixel can be the mean or 
        minimum values across the area can also be specified.
        
        The image is returned as a 2D numpy array.
        '''
        func = np.mean if mean else np.min
        nnet_packets, data_packets = body_cam.get_available_nnet_and_data_packets()
        for packet in data_packets:
            if packet.stream_name == 'depth':
                frame = packet.getData()
                valid_frame = (frame >= min_range) & (frame <= max_range)
                valid_image = np.where(valid_frame, frame, max_range)
                decimated_valid_image = skim.block_reduce(valid_image,(decimate_level,decimate_level),func)
                return decimated_valid_image

    def point_cloud(self, frame, min_range = 200.0, max_range = 4000.0):
        '''
        Generates a point cloud based on the provided numpy 2D depth array.
        
        Returns a 16 x 40 numpy matrix describing the forward distance to
        the points within the field of view of the camera.
        
        Initial measures closer than the min_range are discarded.  Those outside of the
        max_range are set to the max_range.
        '''
        height, width = frame.shape
        # Convert depth map to point cloud with valid depths
        column, row = np.meshgrid(np.arange(width), np.arange(height), sparse=True)
        valid = (frame >= min_range) & (frame <= max_range)
        global test_image
        test_image = np.where(valid, frame, max_range)
        z = np.where(valid, frame, 0.0)
        x = np.where(valid, (z * (column - cx) /cx / fx) + 120.0 , max_range)
        y = np.where(valid, 325.0 - (z * (row - cy) / cy / fy) , max_range)
        # Flatten point cloud axes
        z2 = z.flatten()
        x2 = x.flatten()
        y2 = y.flatten()
        # Stack the x, y and z co-ordinates into a single 2D array
        cloud = np.column_stack((x2,y2,z2))
        # Filter the array by x and y co-ordinates
        in_scope = (cloud[:,1] < 1600) & (cloud[:,1] > 0) & (cloud[:,0] < 2000) & (cloud[:,0] > -2000)
        in_scope = np.repeat(in_scope, 3)
        in_scope = in_scope.reshape(-1, 3)
        scope = np.where(in_scope, cloud, np.nan)
        # Remove invalid rows from array
        scope = scope[~np.isnan(scope).any(axis=1)]
        # Index each point into 10cm x and y bins (40 x 16)
        x_index = pd.cut(scope[:,0], x_bins)
        y_index = pd.cut(scope[:,1], y_bins)
        # Place the depth values into the corresponding bin
        binned_depths = pd.Series(scope[:,2])
        # Average the depth measures in each bin
        totals = binned_depths.groupby([y_index, x_index]).mean()
        # Reshape the bins into a 16 x 40 matrix
        totals = totals.values.reshape(16,40)
        return totals

    def follow_vector(self, image, max_range = 1200.0, certainty = 0.75):
        final_distance = None
        direction = None
        # determine size of supplied image
        height, width = image.shape
        # just use the top half for analysis
        # as this will ignore low obstacles
        half_height = int(height/2)
        image = image[0:half_height,:]
        # find all the columns within the image where there are a
        # consistently significant number of valid depth measurements
        # this suggests a target in range that is reasonably tall
        # and vertical (hopefully a person's legs
        columns = np.sum(image < max_range, axis = 0) >= (certainty*half_height)
        # average the depth values of each column
        distance = np.average(image, axis = 0)
        # create an array with just the useful distances (by zeroing
        # out any columns with inconsistent data)
        useful_distances = distance * columns
        # average out all the useful distances
        # ignoring the zeros and the max_ranges
        subset = useful_distances[np.where((useful_distances < max_range) & (useful_distances > 0.0))]
        if len(subset) > 0:
            final_distance = np.average(subset)
        # determine the indices of the valid columns and average them
        # us the size of the image to determine a relative strength of
        # direction that can be converted into an angle once fov of
        # camera is known (range is theoretically -1 to +1 that
        # corresponds to the h_fov of the camera)
        mid_point = (width - 1.0) / 2.0
        indices = columns.nonzero()
        if len(indices[0]) > 0 :
            direction = (np.average(indices) - mid_point) / width
        return (direction, final_distance)

    def mqtt_callback(self, client, userdata, message):
        """
        Enables K9 to receive a message from an Epruino Watch via
        MQTT over Bluetooth (BLE) to place it into active or inactive States
        """
        payload = str(message.payload.decode("utf-8"))
        if payload != self.last_message:
            self.last_message = payload
            event = payload[3:-1].lower()
            # print("Event: ",str(event))
            self.on_event(event)

# Create the k9 finite state machine
k9 = K9()

try:
    while True:
        k9.run()
except KeyboardInterrupt:
    logo.stop()
    del body_cam
    del device
    k9.client.loop_stop()
    k9.speak("Inactive")
    print('exiting from', str(k9.state).lower(),'state.')
    sys.exit(0)