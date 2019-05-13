import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
 
index_in_epoch = 0
epochs_completed = 0
num_examples = 10

learning_rate = 0.001
total_epoch = 30
batch_size = 128
 
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10
 
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
 

def next_batch(batch_size,shuffle=True):
    start = index_in_epoch
    # Shuffle for the first epoch
    if epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(num_examples)
      numpy.random.shuffle(perm0)
      trains = trains[perm0]
    # Go to the next epoch
    if start + batch_size > num_examples:
      # Finished epoch
      epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = num_examples - start
      rest_trains = trins[start:num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(num_examples)
        numpy.random.shuffle(perm)
        trains = trains[perm]
      # Start next epoch
      start = 0
      index_in_epoch = batch_size - rest_num_examples
      end = index_in_epoch
      new_trains= trains[start:end]
      return numpy.concatenate(
          (rest_trains,new_trains), axis=0)
    else:
      index_in_epoch += batch_size
      end = index_in_epoch
      return trains[start:end]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
 
    total_batch = int(mnist.train.num_examples / batch_size)
 
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
 
    test_batch_size = len(mnist.test.images)
    test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
    test_ys = mnist.test.labels
 
    print('accuracy : ', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))


