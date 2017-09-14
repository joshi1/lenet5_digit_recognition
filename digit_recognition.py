# Udactiy project for neural network implementation in tensorflow
# Uses the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten


# Parameters
EPOCHS = 10
BATCH_SIZE = 50
learning_rate = 0.001
n_classes = 10 


layer_width = {
    'layer_1': 6,
    'layer_2': 16,
    'fully_connected': 120
}

# Store layers weight & bias
#  Note: weights dimensions are
#   filter_height x filter_width x input_depth x output_depth
#
#   filter_height/filter_width: 
#     For Layer 1, the requirement is to have a 28x28x6 inline with the
#     LeNet architecture. Therefore, assuming a stride of 1, we plug
#     in numbers to get the filter_height and filter_width.
#     out_height = ceil(float(in_height - filter_height + 1)/float(strides[1]))
#     28 = (32 - filter_height + 1) / 1. Therefore filter_height = 5
#     Same for filter_width.
#   input_depth: in this grayscale image it is 1 for layer 1.
#                (could be 3 channels in color)
#   output_depth: The output depth is given at each layer.

weights = {
    'layer_1': tf.Variable(tf.truncated_normal(
        [5, 5, 1, layer_width['layer_1']],
        mean = 0, stddev = 0.1)),
    'layer_2': tf.Variable(tf.truncated_normal(
        [5, 5, layer_width['layer_1'], layer_width['layer_2']],
        mean = 0, stddev = 0.1)),
    'fully_connected': tf.Variable(tf.truncated_normal(
        [5*5*16, layer_width['fully_connected']],
        mean = 0, stddev = 0.1)),
    'out': tf.Variable(tf.truncated_normal(
        [layer_width['fully_connected'], n_classes],
        mean = 0, stddev = 0.1))
}

biases = {
    'layer_1': tf.Variable(tf.zeros(layer_width['layer_1'])),
    'layer_2': tf.Variable(tf.zeros(layer_width['layer_2'])),
    'fully_connected': tf.Variable(tf.zeros(layer_width['fully_connected'])),
    'out': tf.Variable(tf.zeros(n_classes))
}

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='VALID')

def LeNet(x):
    # Reshape from 2D to 4D. This prepares the data for
    # convolutional and pooling layers.
    x = tf.reshape(x, (-1, 28, 28, 1))
    
    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")
    
    # Convolution Layer 1. Input = 32x32x1. Output = 28x28x6.
    # Activation 1.
    
    conv1 = conv2d(x, weights['layer_1'], biases['layer_1'])

    # Pooling Layer 1. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1)

    # Convolution Layer 2. Output = 10x10x16.    
    # Activation 2.
    conv2 = conv2d(conv1, weights['layer_2'], biases['layer_2'])

    # Pooling Layer 2. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2)

    # Flatten Layer.
    fc1 = tf.reshape(
        conv2,
        [-1, weights['fully_connected'].get_shape().as_list()[0]])
    
    # Fully Connected Layer 1. Input = 5x5x16. Output = 120.
    fc1 = tf.add(
        tf.matmul(fc1, weights['fully_connected']),
        biases['fully_connected'])

    # Activation 3.
    fc1 = tf.nn.relu(fc1)

    # Fully Connected Layer 2. Input = 120. Output = 10.
    logits = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return logits

class tf_context:
    def __init__(self):
        self.x = tf.placeholder(tf.float32, (None, 784))
        self.y = tf.placeholder(tf.int32, (None))
        self.one_hot_y = tf.one_hot(self.y, 10)
        self.logits = LeNet(self.x)
        self.loss_operation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.one_hot_y))
        self.optimizer = tf.train.AdamOptimizer()
        self.training_operation = self.optimizer.minimize(self.loss_operation)
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1),
                                           tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

#model evaluation
def evaluate(tf_c,
             X_data, y_data):
    num_examples = len(X_data)
    total_accuracy, total_loss = 0, 0
    sess = tf.get_default_session()
    #print ("X_validation shape {}".format(X_data.shape))
    
    for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy =  sess.run([tf_c.loss_operation,
                                    tf_c.accuracy_operation],
                                   feed_dict={tf_c.x: batch_x,
                                              tf_c.y: batch_y})
        total_accuracy += (accuracy * batch_x.shape[0])
        total_loss     += (loss * batch_x.shape[0])
        
    return total_loss / num_examples, total_accuracy / num_examples


#train model
def train(tf_c, sess, X_train, y_train, X_validation, y_validation):
    
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        #print ("{}. X_train shape {}".format(i, X_train.shape))
        print("EPOCH {} ...".format(i+1))
        print(" Training...");
        for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _, loss = sess.run([tf_c.training_operation,
                             tf_c.loss_operation],
                            feed_dict={tf_c.x: batch_x,
                                       tf_c.y: batch_y})
        print(" Validation...");
        validation_loss, validation_accuracy = evaluate(tf_c,
                                                        X_validation,
                                                        y_validation)
        

        print("Validation Loss     = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
            
    try:
        saver
    except NameError:
        saver = tf.train.Saver()
        saver.save(sess, 'lenet')
        print("Model saved")

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/")
    X_train, y_train = mnist.train.images, mnist.train.labels
    #X_validation, y_validation = mnist.train.images, mnist.train.labels
    X_test, y_test   = mnist.test.images, mnist.test.labels
    
    
    index = random.randint(0, len(X_train))
    image = X_train[index]
    
    # Reshape MNIST image from vector to matrix
    image = np.reshape(image, (28, 28))
    
    plt.figure(figsize=(1,1))
    plt.imshow(image, cmap="gray")
    #plt.show()
    #print(y_train[index])

    X_train, X_validation, y_train, y_validation = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.2,
                                                                    random_state=1)
    
    # shuffle the data
    X_train, y_train = shuffle(X_train, y_train)

    #Create tensorflow context
    tf_c = tf_context()
    
    #setup tensorflow
    with tf.Session() as sess:
        train(tf_c, sess, X_train, y_train, X_validation, y_validation)
        
        #evaluate performance on test set
        loader = tf.train.import_meta_graph('lenet.meta')
        loader.restore(sess, tf.train.latest_checkpoint('./'))
        
        test_loss, test_accuracy = evaluate(tf_c, X_test, y_test)

        print("Test Loss     = {:.3f}".format(test_loss))
        print("Test Accuracy = {:.3f}".format(test_accuracy))


