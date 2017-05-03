import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import model

TRAINING_ITERATIONS = 10000
ITERATIONS_PER_RUN = 10000

DROPOUT = 0.5
BATCH_SIZE = 25

VALIDATION_SIZE = 2000

IMAGE_TO_DISPLAY = 263

dataset = pd.read_csv('test7500.csv')
images = dataset.iloc[:, 1:785].values.astype(np.float)
labels_flat = dataset[[0]].values.ravel()

images = np.multiply(images, 1.0 / 255.0)
print('data size: (%g, %g)' % images.shape)

image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
print ('image_width => {0}\nimage_height => {1}'.format(image_width, image_height))


def display(img):
    one_image = img.reshape(image_width, image_height)
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

#display(images[IMAGE_TO_DISPLAY])

print('length of one image ({0})'.format(len(labels_flat)))
print ('label of image [{0}] => {1}'.format(IMAGE_TO_DISPLAY, labels_flat[IMAGE_TO_DISPLAY]))

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

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]
train_labels_flat = labels_flat[VALIDATION_SIZE:]

print('train data size({0[0]},{0[1]})'.format(train_images.shape))
print('validation data size({0[0]},{0[1]})'.format(validation_images.shape))

# model
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

num_examples = train_images.shape[0]

def stratified_shuffle(labels, num_classes):
    ix = np.argsort(labels).reshape((num_classes,-1), len(ix))
    for i in range(len(ix)):
        np.random.shuffle(ix[i])
    return ix.T.reshape((-1), len(ix))

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

train_accuracies = []
validation_accuracies = []
x_range = []

display_step = 100

# serve data by batches
def next_batch(batch_size):
    global train_images
    global train_labels
    global train_labels_flat
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch += batch_size

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        #perm = stratified_shuffle(train_labels_flat, 35)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        train_labels_flat = train_labels_flat[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]

global_step = tf.Variable(0, name='global_step', trainable=False)
y_ = tf.placeholder("float", [None, 26])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_step = global_step.eval(sess)
    for _ in range(10000):
        batch_xs, batch_ys = next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    print(sess.run(accuracy, feed_dict={x:validation_images,y_: validation_labels}))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)