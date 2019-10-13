import keyboard
import time

def up():
    sleep();
    print("UP Triggered! - zoom in")
    keyboard.send('windows,plus', True, False)

def down():
    #sleep();
    print("Down Triggered! - zoom out")
    keyboard.send('windows,-', True, False)


def right():
    #sleep();
    print("Right Triggered! - go right")
    keyboard.send('windows, right', True, False)


def left():
    #sleep();
    print("Left Triggered! go left")
    keyboard.send('windows, left', True, False)

def kill():
    print("keyboard interrupt == ctrl+c")
    keyboard.send('ctrl,c', True, False)


def sleep():
    time.sleep(2)