################################################################################
# Copyright (C) 2012-2016 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################

import os, sys, inspect, thread, time
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = './lib/x64' if sys.maxsize > 2**32 else './lib/x86'
arch_dir1 = './lib'

sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
sys.path.insert(1, os.path.abspath(os.path.join(src_dir, arch_dir1)))

import Leap, train
import tensorflow as tf
import numpy as np
import keyboard_callback


checkpoint_dir = './checkpoint'


class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']

    recording = False
    frameCount=0
    gesture = []
    data = []
    minGestureFrames = 5

    def recordValue(self, val):
        self.gesture.append(val)
    def recordVector(self, v):
        self.recordValue(v[0])
        self.recordValue(v[1])
        self.recordValue(v[2])
    def recordDataValue(self, val):
        self.data.append(val)
    def recordData(self, v):
        self.recordDataValue(v[0])
        self.recordDataValue(v[1])
        self.recordDataValue(v[2])

    def on_init(self, controller):
        print "Initialized"

    def on_connect(self, controller):
        print "Connected"

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    def on_exit(self, controller):
        print "Exited"

    def on_frame(self, controller):

        # Get the most recent frame and report some basic information
        frame = controller.frame()


        #print "Frame id: %d, timestamp: %d, hands: %d, fingers: %d" % (frame.id, frame.timestamp, len(frame.hands), len(frame.fingers))

        #if (new Date().getTime() - this.lastHit < this.downtime) { return; }


        if(self.recordableFrame(frame)):
            if (self.recording == False):
                self.recording = True
                self.frameCount = 0
                self.gesture = []
                self.data = []
                print("started-recording")

            self.frameCount = self.frameCount+1
            self.recordFrame(frame)
        elif (self.recording == True):
            self.recording = False
            print("stopped-recording")
            if (self.frameCount >= self.minGestureFrames):
                print("gesture-detected")
                test_data = self.convert_to_test_data()
                test_lstm(test_data)

        #if not frame.hands.is_empty:
        #    print ""

    def recordableFrame(self, frame):
        min = 300
            
        for hand in frame.hands:
            palmVelocity = hand.palm_velocity
            palmVelocity = max(abs(palmVelocity[0]),abs(palmVelocity[1]), abs(palmVelocity[2]))

            if (palmVelocity >= min):
                return True;

    def recordFrame(self, frame):
        fingervalue = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        for hand in frame.hands:
            self.recordVector(hand.stabilized_palm_position)
            self.recordData(hand.stabilized_palm_position)

            for finger in hand.fingers:
                self.recordVector(finger.stabilized_tip_position)
                fingervalue[finger.id%10] = finger.stabilized_tip_position
            for a in range(5):
                self.recordData(fingervalue[a])

    def convert_to_test_data(self):
        num_frame = 0
        arr = np.array([], dtype=float)
        tests_tmp =[]
        tmp = []
        count = 0
        gesture = self.gesture
        for j in range (len(gesture)):
            if(gesture[j] == 0.0):
                tmp.append(gesture[j])
            elif(j%3 == 0):
                tmp.append(round((gesture[j] - gesture[0]),2))

            elif(j%3 == 1):
                tmp.append(round((gesture[j] - gesture[1]),2))

            elif(j%3 == 2):
                tmp.append(round((gesture[j] - gesture[2]),2))
            else:
                print("data error")
            count = count+1

            # 1 frame = 18 point(x,y,z)
            if(count%18==0):
                num_frame = num_frame+1
                out = np.array(tmp[0:18], dtype=float)
                arr = np.append(arr, out)
                tmp = []

                if(num_frame > train.get_max_frame()):
                    print("frame overflow")

        tests_tmp.append(tuple((arr, np.array([0,0,0,0,1]))))

        tests=[]
        for i in range(len(tests_tmp)):
            num_frame = int(len(tests_tmp[i][0])/18)
            zero_arr = np.zeros((train.get_max_frame()-num_frame)*18, dtype=float)
            tests.append(tuple((np.concatenate((tests_tmp[i][0], zero_arr)),  tests_tmp[i][1])))
        return tests


def test_lstm(test_data):
    sess = tf.InteractiveSession()
    new_saver = tf.train.import_meta_graph('checkpoint/model.ckpt-100.meta')
    new_saver.restore(sess, 'checkpoint/model.ckpt-100')

    tf.get_default_graph()
    X = sess.graph.get_tensor_by_name("input:0")
    Y = sess.graph.get_tensor_by_name("output:0")
    model = sess.graph.get_tensor_by_name("model:0")

    test_batch_size = len(test_data)
    test_batchx = []
    for i in range(0,test_batch_size):
        test_batchx.append(test_data[i][0])

    test_batchx = np.array(test_batchx)

    test_xs = test_batchx.reshape((test_batch_size, train.n_step, train.n_input))

    result_arr = (sess.run(model, feed_dict = {X: test_xs}))
    result = np.argmax(result_arr)      #up=0, down=1, right=2, left=3, pew=4

    print(result_arr)
    print(result)
    mapping_callback(result)

def mapping_callback(result):
    if(result == 0):
        keyboard_callback.up()
    elif(result == 1):
        keyboard_callback.down()
    elif(result == 2):
        keyboard_callback.right()
    elif(result == 3):
        keyboard_callback.left()
    else:
        #keyboard_callback.kill()
        sys.exit()

def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    # Keep this process running until Enter is pressed
    print "Press Enter to quit..."
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)
 

if __name__ == "__main__":
    main()

