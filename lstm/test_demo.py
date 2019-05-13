import tensorflow as tf
import numpy as np
import os
import csv
import random

num_examples = 50
learning_rate = 0.001
total_epoch = 100
batch_size = 16

# next_batch
epochs_completed = 0
index_in_epoch = 0

n_input = 18
n_hidden = 128
n_class = 5

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

gesture_names = ['up', 'down', 'right', 'left', 'pew']

# up:data[0]~[9], down:data[10]~[19], left:data[20]~[29], right[30]~[39], pew[40]~[49]
trains_tmp = []
trains = []

#for step
max_num_frame = 0

# data load + label.
def load_data():
    global max_num_frame
    # number of gesture (5)
    for gesture_i in range(0, len(gesture_names)):
        data_dir = os.path.join('./data', gesture_names[gesture_i])
        data_list = os.listdir(data_dir)
        data_list.sort()
        gesture_data_list = [os.path.join(data_dir,x) for x in data_list]

        # onehot label data
        label_onehot = np.zeros(len(gesture_names), dtype=int)
        label_onehot[gesture_i] = 1

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
                div = 18    # 1frame ( (hand x, y, z) + 5x(finger x, y, z))
                arr = np.array(gesture[start_pos : start_pos+div], dtype=float)
                start_pos = start_pos+div for idx in range(start_pos, end_pos, div):
                    out = gesture[start_pos : start_pos+div]
                    start_pos = start_pos+div
                    arr = np.append(arr,out)

                arr = np.array(arr, dtype=float)

                # data[i] = tuple(x_i, y_i)
                # x_i : a gesture data in array format with size (num_frame x 18)
                # y_i : label data indicates actual gesture name 
                trains_tmp.append(tuple((arr,label_onehot)))
            f.close()

def zero_padding():
    global trains_tmp
    global trains
    for i in range(num_examples):
        num_frame = int(len(trains_tmp[i][0])/18)
        zero_arr = np.zeros((max_num_frame-num_frame)*18, dtype=float)
        trains.append(tuple((np.concatenate((trains_tmp[i][0], zero_arr)),  trains_tmp[i][1])))

def next_batch(batch_size,shuffle=True):
    global index_in_epoch
    global epochs_completed
    global num_examples
    global trains
    start = index_in_epoch
    # Shuffle for the first epoch

    if epochs_completed == 0 and start == 0 and shuffle:
        #perm0 = np.arange(num_examples)
        #np.random.shuffle(perm0)
        #trains = trains[perm0]
        random.shuffle(trains)
    # Go to the next epoch
    if start + batch_size > num_examples:
        # Finished epoch
        epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = num_examples - start

        batchx = []
        batchy = []
        rest_trains = trains[start:num_examples]

        for i in range(start, num_examples):
            batchx.append(trains[i][0])
            batchy.append(trains[i][1])

        # Shuffle the data 
        if shuffle:
            random.shuffle(trains)
        # Start next epoch
        start = 0
        index_in_epoch = batch_size - rest_num_examples
        end = index_in_epoch
        new_trains= trains[start:end]

        for i in range(start, end):
            batchx.append(trains[i][0])
            batchy.append(trains[i][1])

        batchx = np.array(batchx)
        batchy = np.array(batchy)
        
        #return np.concatenate((rest_trains,new_trains), axis=0)[0], np.concatenate((rest_trains,new_trains), axis=0)[1]
        return batchx, batchy

    else:     
        index_in_epoch += batch_size
        end = index_in_epoch
        batchx = []
        batchy = []
        for i in range(start,end):
            batchx.append(trains[i][0])
            batchy.append(trains[i][1])
        
        batchx = np.array(batchx)
        batchy = np.array(batchy)
        #return trains[start:end][0], trains[start:end][1]
        return batchx, batchy


load_data()
zero_padding()
n_step = max_num_frame

X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))
 
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
 
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
 
model = tf.matmul(outputs, W) + b
 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model, labels=Y
))
 
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    total_batch = int(num_examples / batch_size)
 
    for epoch in range(total_epoch):
        total_cost = 0
 
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size)
            batch_xs = batch_xs.reshape((batch_size, n_step, n_input))
 
            _, cost_val = sess.run([optimizer, cost],
                feed_dict={X: batch_xs, Y: batch_ys})
            total_cost += cost_val
 
        print('Epoch:', '%04d' % (epoch + 1),
            'Avg. cost: {:.4}'.format(total_cost / total_batch))
 
    print("fin") 
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
 

    test_batch_size =len(tests) 
    test_batchx = []
    test_batchy = []
    for i in range(0,test_batch_size):
        test_batchx.append(tests[i][0])
        test_batchy.append(tests[i][1])

    test_batchx = np.array(test_batchx)
    test_ys= np.array(test_batchy)
    test_xs = test_batchx(test_batch_size, n_step, n_input)

 
    print('accuracy : ', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))

