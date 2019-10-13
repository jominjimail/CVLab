import keyboard
import time

def up():
    sleep();
    print("UP Triggered! space == 57, down == 80")
    keyboard.send('windows + plus', True, False)

def down():
    #sleep();
    print("Down Triggered! up == 72")
    keyboard.send(72, True, False)
    keyboard.send('windows,-', True, False)


def right():
    #sleep();
    print("Right Triggered! left == 75")
    keyboard.send(75, True, False)


def left():
    #sleep();
    print("Left Triggered! right == 77")
    keyboard.send(77, True, False)

def kill():
    print("keyboard interrupt == ctrl+c")
    keyboard.send('ctrl,c', True, False)


def sleep():
    time.sleep(2)