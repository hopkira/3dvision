#!/usr/bin/env python3
# coding: utf-8

import time
from transitions.extensions import GraphMachine as Machine

class K9Robot(object):

    states = ['observe', 'announce', 'identify', 'hotword', 'get_command', 'understand', 'respond', 'follow']

    def __init__(self, name)
        self.name = name
        self.machine = Machine(model=self, states=K9Robot.states, initial='observe')
        self.machine.add_transition(trigger='face_detected', source='observe', dest='identify')
        self.machine.add_transition(trigger='listen_on', source='observe', dest='hotword')
        self.machine.add_transition(trigger='name_heard', source='hotword', dest='get_command')
        self.machine.add_transition(trigger='command_received', source='get_command', dest='understand')
        self.machine.add_transition(trigger='respond', source='understand', dest='respond')
        self.running = True
        
    def run(self):
        while self.running:
            print('At state', self.state)
            # main event loop

k9 = K9Robot("K9")

k9.get_graph().draw('k9_state_diagram.png', prog = 'dot')

k9.run()

print('Done')

