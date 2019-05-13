import tensorflow as tf
import numpy as np

index_in_epoch = 0
epochs_completed = 0
num_examples = 30
batch_size =4 

total_epoch = 3
trains = np.arange(1,num_examples+1)


def next_batch(batch_size,shuffle=True):
    global index_in_epoch
    global epochs_completed
    global num_examples
    global trains
    start = index_in_epoch
    # Shuffle for the first epoch
    if epochs_completed == 0 and start == 0 and shuffle:
        
        perm0 = np.arange(num_examples)
        np.random.shuffle(perm0)
        trains = trains[perm0]
        print ("first shuffles : ")
        print (trains)
    # Go to the next epoch
    if start + batch_size > num_examples:
        # Finished epoch
        epochs_completed += 1
        # Get the rest examples in this epoch
        rest_num_examples = num_examples - start
        rest_trains = trains[start:num_examples]
        # Shuffle the data 
        if shuffle:
            perm = np.arange(num_examples)
            np.random.shuffle(perm)
            trains = trains[perm]
            print("next epoch shuffle :")
            print(trains)
        # Start next epoch
        start = 0
        index_in_epoch = batch_size - rest_num_examples
        end = index_in_epoch
        new_trains= trains[start:end]
        print("rest + new :")
        
        return np.concatenate((rest_trains,new_trains), axis=0)
    else:     
        index_in_epoch += batch_size
        end = index_in_epoch
        print ("batch start ~ end : ",start, end)
        return trains[start:end]

print( "before trains : ",trains)
total_batch = int(num_examples / batch_size)

for epoch in range(total_epoch):
    for i in range(total_batch):
        print("epoch : ",epoch)
        print("batch : ",i )
        output_trains = next_batch(batch_size)
        
        print(output_trains)
        print("\n")
