import os
import numpy as np
import csv

gesture_names = ['up', 'down', 'right', 'left', 'pew']

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

max_num_frame = 0

num_examples = 50
# up:data[0]~[9], down:data[10]~[19], left:data[20]~[29], right[30]~[39], pew[40]~[49]
trains_tmp = []
trains = []
# data load + label.
def load_data():
	global max_num_frame
	global trains_tmp
	# number of gesture (5)
	for gesture_i in range(0, len(gesture_names)):
		data_dir = os.path.join('./data', gesture_names[gesture_i])
		data_list = os.listdir(data_dir)
		data_list.sort()
		gesture_data_list = [os.path.join(data_dir,x) for x in data_list]

		for i in range(0,len(gesture_data_list)):
			f = open(gesture_data_list[i], 'r')
			rdr = csv.reader(f)

			# 1 gesture
			for gesture in rdr:
				start_pos = 0
				end_pos = len(gesture)-1
				num_frame = (int)((len(gesture)-1)/18)

				if(num_frame>max_num_frame):
					max_num_frame = num_frame

				# gesture to numpy. (row = frame)
				div = 18 	# 1frame ( (hand x, y, z) + 5x(finger x, y, z))
				arr = np.array(gesture[start_pos : start_pos+div], dtype=float)
				start_pos = start_pos+div
				for idx in range(start_pos, end_pos, div):
					out = gesture[start_pos : start_pos+div]
					start_pos = start_pos+div
					arr = np.append(arr,out)

				arr = np.array(arr, dtype=float)

				# data[i] = tuple(x_i, y_i)
				# x_i : a gesture data in array format with size (num_frame x 18)
				# y_i : label data indicates actual gesture name
				trains_tmp.append(tuple((arr,gesture_names[gesture_i])))
			f.close()

def zero_padding():
	global trains_tmp
	global trains
	for i in range(num_examples):
		num_frame = int(len(trains_tmp[i][0])/18)
		zero_arr = np.zeros((max_num_frame-num_frame)*18, dtype=float)
		trains.append(tuple((np.concatenate((trains_tmp[i][0], zero_arr)),  trains_tmp[i][1])))

load_data()
zero_padding()

print(trains[11])