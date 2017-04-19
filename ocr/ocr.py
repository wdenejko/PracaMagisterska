import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import time

start = time.time()
LEARNING_RATE = 1e-4

TRAINING_ITERATIONS = 20000
ITERATIONS_PER_RUN = 5000

DROPOUT = 0.5
BATCH_SIZE = 200

VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 263

def stratified_shuffle(labels, num_classes):
    ix = np.argsort(labels).reshape((num_classes,-1))
    for i in range(len(ix)):
        np.random.shuffle(ix[i])
    return ix.T.reshape((-1))

dataset = pd.read_csv("output.csv")

images = dataset.iloc[:,1:].values
images = images.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)
print('data size: (%g, %g)' % images.shape)

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width,image_height))


def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)


display(images[IMAGE_TO_DISPLAY])

labels_flat = dataset[[0]].values.ravel()
print('length of one image ({0})'.format(len(labels_flat)))
print ('label of image [{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels_flat[IMAGE_TO_DISPLAY]))

labels_count = np.unique(labels_flat).shape[0]
print('number of labes => {0}'.format(labels_count))

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

print('labels({0[0]},{0[1]})'.format(labels.shape))
print ('labels vector for image [{0}] => {1}'.format(IMAGE_TO_DISPLAY,labels[IMAGE_TO_DISPLAY]))

# split data into training & validation sets
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
train_labels_flat = labels_flat[VALIDATION_SIZE:]

print('train data size({0[0]},{0[1]})'.format(train_images.shape))
print('validation data size({0[0]},{0[1]})'.format(validation_images.shape))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder('float', shape=[None, image_size])
y_ = tf.placeholder('float', shape=[None, labels_count])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

image = tf.reshape(x, [-1,image_width , image_height,1])

h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

layer1 = tf.reshape(h_conv1, (-1, image_height, image_width, 4 ,8))
layer1 = tf.transpose(layer1, (0, 3, 1, 4,2))
layer1 = tf.reshape(layer1, (-1, image_height*4, image_width*8))

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

layer2 = tf.reshape(h_conv2, (-1, 14, 14, 4 ,16))
layer2 = tf.transpose(layer2, (0, 3, 1, 4,2))
layer2 = tf.reshape(layer2, (-1, 14*4, 14*16))

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, labels_count])
b_fc2 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step=global_step)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

predict = tf.argmax(y,1)
print(predict)

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

train_accuracies = []
validation_accuracies = []
x_range = []

display_step = 100

def next_batch(batch_size):
    global train_images
    global train_labels
    global train_labels_flat
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    if index_in_epoch > num_examples:
        epochs_completed += 1

        perm = stratified_shuffle(train_labels_flat, 10)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        train_labels_flat = train_labels_flat[perm]
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

def check_progress(sess, accuracy, batch_xs, batch_ys, i):
    import datetime
    train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
    if(VALIDATION_SIZE):
        validation_accuracy = accuracy.eval(session=sess, feed_dict={x: validation_images,
                                                                     y_: validation_labels,
                                                                     keep_prob: 1.0})
        print('%s: trn / val => %.4f / %.4f for step %d'
              % (str(datetime.datetime.now()), train_accuracy, validation_accuracy, i+1))
    else:
        print('training_accuracy => %.4f for step %d' % (train_accuracy, i+1))
        validation_accuracy = []
    return (train_accuracy, validation_accuracy)

saver = tf.train.Saver(max_to_keep=5)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

ckpt = tf.train.get_checkpoint_state('.')
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restored session from: %s" % ckpt.model_checkpoint_path)
else:
    print("No checkpoint found.")

start_step = global_step.eval(sess)
for i in range(start_step, min(start_step + ITERATIONS_PER_RUN, TRAINING_ITERATIONS)):
    batch_xs, batch_ys = next_batch(BATCH_SIZE)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

    if (i+1) % display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        train_accuracy, validation_accuracy = check_progress(sess, accuracy, batch_xs, batch_ys, i)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        x_range.append(i)

saver.save(sess, os.path.join(os.path.dirname(__file__), "convolutional.ckpt"))
print(os.path.join(os.path.dirname(__file__), "convolutional.ckpt"))

if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(session=sess, feed_dict={x: validation_images,
                                                   y_: validation_labels,
                                                   keep_prob: 1.0})
    print('validation_accuracy => %.4f'%validation_accuracy)

end = time.time()
print(end - start)


test_images = pd.read_csv('test.csv').values
test_images = test_images.astype(np.float)

test_images = np.multiply(test_images, 1.0 / 255.0)
print('test_images({0[0]},{0[1]})'.format(test_images.shape))

predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(
        session=sess, feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE],
                                 keep_prob: 1.0})

print('predicted_lables({0})'.format(len(predicted_lables)))

display(test_images[IMAGE_TO_DISPLAY])
print ('predicted_lables[{0}] => {1}'.format(IMAGE_TO_DISPLAY,predicted_lables[IMAGE_TO_DISPLAY]))

# save results
file_name = 'submission_softmax_%s.csv' % global_step.eval(sess)
np.savetxt(file_name,
           np.c_[range(1,len(test_images)+1),predicted_lables],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')
print('saved: %s' % file_name)

layer1_grid = layer1.eval(session=sess, feed_dict={x: test_images[IMAGE_TO_DISPLAY:IMAGE_TO_DISPLAY+1],
                                                   keep_prob: 1.0})
plt.axis('off')
plt.imshow(layer1_grid[0], cmap=cm.seismic )
plt.show()