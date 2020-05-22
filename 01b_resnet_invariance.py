# -----------------------------------------------------------------------------------------------------
#
# Copyright 2020 Amos Tsai amosdst@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# -----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# TensorFlow 2 Practices
#
#   https://www.tensorflow.org/
#
# refs ...
#   /home/amos/ws/refs/neural/tools/Learning TensorFlow.pdf
#
# tensorflow references
#   api ...
#     https://www.tensorflow.org/versions
#     https://www.tensorflow.org/api_docs/python/tf
#   tutorials ...
#     https://www.tensorflow.org/tutorials
#
# A tensorflow core program consists of two discrete sections
#
#   tf.Graph -- the graph is composed of two types of objects
#               (https://www.tensorflow.org/api_docs/python/tf/Graph#class_graph)
#
#     tf.Operation (https://www.tensorflow.org/api_docs/python/tf/Operation#class_operation)
#       - An Operation is a node in a TensorFlow Graph that takes zero or more Tensor objects as input, and produces zero or
#         more Tensor objects as output.
#       - Objects of type Operation are created by calling a Python op constructor (such as tf.matmul) or tf.Graph.create_op.
#
#     tf.Tensor -- represents one of the input/output of an operation
#                  (https://www.tensorflow.org/api_docs/python/tf/Tensor#class_tensor)
#       - A Tensor builds a dataflow connection between operations
#       - A Tensor can be computed by passing it to tf.Session.run()
#       - tf.Tensor.eval() is a shortcut for calling tf.get_default_session().run(tf.Tensor&)
#
#  tf.Session -- encapsulates the environment where operations are executed, and tensors are evaluated
#                (https://www.tensorflow.org/api_docs/python/tf/Session#class_session)
#
#     - A session own resources, such as tf.Variable, tf.Queue, and tf.ReaderBase
#     - Session configuration options is composed in a tf.ConfigProto object
#
# ----------------------------------------------------------------------------

##

import sys
import os
import numpy
import time
import enum
from enum import IntEnum
# from enum import unique

import cv2
import imutils
# from sympy import Matrix

# import matplotlib.dates
# import matplotlib.ticker
# import mpl_toolkits.mplot3d.axes3d as axes3d
# import matplotlib.pyplot as pyplot
# import mpl_finance as finance

# import pylab
# from scipy.io import wavfile

# disable warning message "RuntimeWarning: numpy.dtype size changed..." from importing tensorflow
# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# enable the following line to clear residual info messages in the environment
#os.system('clear')

##
# ----------------------------------------------------------------------------

#  Level | Level for Humans | Level Description
#  -------|------------------|------------------------------------
#   0     | DEBUG            | [Default] Print all messages
#   1     | INFO             | Filter out INFO messages
#   2     | WARNING          | Filter out INFO & WARNING messages
#   3     | ERROR            | Filter out all messages

#log_level = tf.get_logger().getEffectiveLevel()
#tf.get_logger().setLevel('WARNING')
#log_level = tf.get_logger().getEffectiveLevel()

#log_level = tf.compat.v1.logging.get_verbosity()
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

##
# ----------------------------------------------------------------------------

print('\ncuda home : {}\n'.format(os.environ['CUDA_HOME']))
print('path      : {}\n'.format(os.environ['PATH']))
#print('ld_path   : {}\n'.format(os.environ['LD_LIBRARY_PATH']))

print('tensor-flow version : %s\n' % tf.__version__)

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

##
# ----------------------------------------------------------------------------
# Load and prepare the MNIST dataset.
#
# refs ...
#   https://keras.io/api/datasets/
#   https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data
#   http://yann.lecun.com/exdb/mnist/
#

# tf.keras.datasets.mnist.load_data()
#
# Arguments
#
#   path 	path where to cache the dataset locally (relative to ~/.keras/datasets).
#
# Returns
#
#   Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
#
#   x_train, x_test: uint8 arrays of grayscale image data with shapes (num_samples, 28, 28).
#
#   y_train, y_test: uint8 arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
#

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = (x_train / 255.0).astype(numpy.float32), (x_test / 255.0).astype(numpy.float32)

print('x_train shape = {}'.format(x_train.shape))
print('y_train shape = {}'.format(y_train.shape))

print('x_test shape  = {}'.format(x_test.shape))
print('y_test shape  = {}'.format(y_test.shape))

'''
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes = 10)

print('y_train one-hot shape = {}'.format(y_train.shape))
print('y_test  one-hot shape = {}'.format(y_test.shape))
'''

##
# ----------------------------------------------------------------------------
# A Typical CNN Network
#
# keras documentation
#
#  https://keras.io
#
#  API (models)
#    https://keras.io/api/models/
#  API (layers)
#    https://keras.io/api/layers/
#  API (training)
#    https://keras.io/api/models/model_training_apis/
#  API (model saving)
#    https://keras.io/api/models/model_saving_apis/
#
# callable classes
#
#  keras layer objects are callables and are used to bind to the layer given in its argument
#  (https://treyhunner.com/2019/04/is-it-a-class-or-a-function-its-a-callable/)
#
# shapes
#
#  'Input'
#    shape : a shape tuple without the batch size
#
#    # keras.Input(shape = (x_train.shape[1] * x_train.shape[2]), dtype = tf.float32)
#    #  => Out[14]: <tf.Tensor 'input_7:0' shape=(None, 784) dtype=float32>
#
#    # keras.Input(shape = (x_train.shape[1], x_train.shape[2]), dtype = tf.float32)
#    #  => Out[13]: <tf.Tensor 'input_6:0' shape=(None, 28, 28) dtype=float32>
#
#  'Dense'
#    input shape  : (batch_size, ..., input_dim) e.g., (batch_size, input_dim)
#    output shape : (batch_size, ..., units)     e.g., (batch_size, units)
#    activation   : https://keras.io/api/layers/activations/
#                   relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential(...)
#
# activation
#
#  - activation is implemented as layer classes in keras, and is ...
#      instantiated via activation functions, or
#      passed as a literal string asking the layer object to instantiate it for the client
#
#  - activation (argument) is None by default for certain amount of layer class
#      care should be taken when composing model layers
#
# Q1:
#
# Q2:
#

import tensorflow.keras as keras

model_name = '01b_resnet_invariance'
batch_size = 16
nr_epochs = 16
nr_filters = 32

# ----------------------------------------------------------------------------
# The Microsoft ResNet type of CNN Model for Classifier
#

def conv_block(nr_filters, x_in) :

    x_out = keras.layers.Conv2D(filters = nr_filters, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(x_in)
    x_out = keras.layers.Conv2D(filters = nr_filters, kernel_size = (3, 3), strides = (2, 2), padding = 'same', activation = 'relu')(x_out)

    return x_out

def residual_block(nr_filters, x_in) :

    x_out = x_in
    x_out = keras.layers.Conv2D(filters = nr_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')(x_out)
    x_out = keras.layers.Conv2D(filters = nr_filters, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation = 'relu')(x_out)
    x_out = keras.layers.add([x_in, x_out])

    return x_out

# the input
x_in = keras.Input(shape = (x_train.shape[1], x_train.shape[2], 1), batch_size = batch_size)

# (front-end) the first residual block
x = residual_block(nr_filters, x_in)
x = conv_block(nr_filters * 2, x)

# (front-end) the second residual block
x = residual_block(nr_filters * 2, x)
x = conv_block(nr_filters * 4, x)

# (front-end)
# ......
x = residual_block(nr_filters * 4, x)
#x = conv_block(nr_filters * 8, x)

#x = residual_block(nr_filters * 8, x)
#x = conv_block(nr_filters * 16, x)

# (front-end) the final pooling layer
x = keras.layers.GlobalAveragePooling2D()(x)

# (back-end) the input layer
#x = keras.layers.Flatten(data_format = 'channels_last')(x)
#x = keras.layers.Dense(100)(x)
#x = keras.activations.relu(x)

# (back-end) the hidden layers
#x = keras.layers.Dense(200, activation = 'sigmoid')(x)

# (back-end) the output layer
#x = keras.layers.Dropout(rate = 0.2)(x)
y = keras.layers.Dense(10, activation = 'softmax')(x)

# the model
model = keras.Model(x_in, y)

# Model: "model_3"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_5 (InputLayer)            [(5, 28, 28, 1)]     0
# __________________________________________________________________________________________________
# conv2d_16 (Conv2D)              (5, 28, 28, 16)      160         input_5[0][0]
# __________________________________________________________________________________________________
# conv2d_17 (Conv2D)              (5, 28, 28, 16)      2320        conv2d_16[0][0]
# __________________________________________________________________________________________________
# add_5 (Add)                     (5, 28, 28, 16)      0           input_5[0][0]
#                                                                  conv2d_17[0][0]
# __________________________________________________________________________________________________
# conv2d_18 (Conv2D)              (5, 14, 14, 32)      4640        add_5[0][0]
# __________________________________________________________________________________________________
# conv2d_19 (Conv2D)              (5, 7, 7, 32)        9248        conv2d_18[0][0]
# __________________________________________________________________________________________________
# conv2d_20 (Conv2D)              (5, 7, 7, 32)        9248        conv2d_19[0][0]
# __________________________________________________________________________________________________
# conv2d_21 (Conv2D)              (5, 7, 7, 32)        9248        conv2d_20[0][0]
# __________________________________________________________________________________________________
# add_6 (Add)                     (5, 7, 7, 32)        0           conv2d_19[0][0]
#                                                                  conv2d_21[0][0]
# __________________________________________________________________________________________________
# global_average_pooling2d_3 (Glo (5, 32)              0           add_6[0][0]
# __________________________________________________________________________________________________
# dense_3 (Dense)                 (5, 10)              330         global_average_pooling2d_3[0][0]
# ==================================================================================================
# Total params: 35,194
# Trainable params: 35,194
# Non-trainable params: 0
# __________________________________________________________________________________________________
#
model.summary()

##
# ----------------------------------------------------------------------------
# Add the Optimizer
#
# available losses
#
#   https://keras.io/api/losses/
#
#   probabilistic losses
#    'binary_crossentropy'                          for two-classes (0 or 1) classification
#    'categorical_crossentropy'                     for multi-class classification
#    'sparse_categorical_crossentropy'              categorical_crossentropy() with build-in one-hot conversion
#    'poisson'
#    'binary_crossentropy'
#    'binary_crossentropy'
#
#   regression losses
#    'mean_squared_error'
#    'mean_absolute_error'
#    'mean_absolute_percentage_error'
#    'mean_squared_logarithmic_error'
#    'cosine_similarity'
#    'huber_loss'
#    'log_cosh'
#
#   hinge losses for maximum-margin classification
#    'hinge'
#    'squared_hinge'
#    'categorial_hinge'
#
# available optimizers
#
#   https://keras.io/api/optimizers/
#
#   id          class       description
#   ----------  ----------  ----------------------------------------------------------------------------------
#   'sgd'       SGD         stochastic gradient descent
#   'rmsprop'   RMSprop     root-mean-square (moving average square root) gradient descent
#   'adam'      Adam        a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
#   'adadelta'  Adadelta    a stochastic gradient descent method that is based on adaptive learning rate per dimension to address two drawbacks: the continual decay of learning rates, and the need for a manually selected global learning rate
#   'adagrad'   Adagrad     an optimizer with parameter-specific learning rates
#   'adamax'    Adamax      a variant of Adam based on the infinity norm
#   'nadam'     Nadam       Adam is essentially RMSprop with momentum, Nadam is Adam with Nesterov momentum
#   'ftrl'      Ftrl        https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
#
# useful tools to deal with one-hot conversion
#
#   keras.utils.to_categorical(y, num_classes)      scaler to one-hot binary vector/matrix
#
#   tf.argmax(y, axis)                              one-hot vector to scaler
#                                                   (not compatible with keras.utils.to_categorical() -- unable to do the inverse conversion directly)
#
#   numpy.argsort(a)[i]                             sort the array-like 'a' in ascending order, replace the item
#                                                   with its index, and then take the i'th index
#

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##
# ----------------------------------------------------------------------------
# data invariance via keras.preprocessing.image.ImageDataGenerator
#
#  https://keras.io/api/preprocessing/image/#imagedatagenerator-class
#

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

data_generator = keras.preprocessing.image.ImageDataGenerator(
    rotation_range = 16,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 16,
    zoom_range = 0.1,
    horizontal_flip = False,
    vertical_flip = False,
    data_format = 'channels_last',
    validation_split = 0.1,
    #dtype = numpy.float32,
)

data_generator.fit(x_train)

##
# ----------------------------------------------------------------------------
# model training
#
#  steps_per_epoch * batch_size = number_of_rows_in_train_data
#
#    When you provide 's' steps per epoch , Each 's' step will have 'x' batches each consisting 'n' samples
#    are sent to fit_generator.  So, if you specify 5 steps per epoch, each epoch computes 'x' batches each
#    consisting of 'n' samples 5 times, then the next epoch is started!
#
#  How keras behave when step_per_epoch != NUM_OF_SAMPLES / batch_size ...
#
#    https://github.com/keras-team/keras/issues/10164
#
#    We launch a process to creates the samples.
#    1. The process knows that you have 10 batches per epoch from the Sequence itself (len is defined)
#       Keras starts a loop asking for 5 batches, because you asked for an epoch of 5 steps_per_epoch.
#    2. It computes the loss on those 5 batches, calls you callbacks etc.
#       Now Keras starts its second epoch and it asked for 5 batches.
#    3. The process is still creating the samples for batches 6..10 so it return those. Then it restarts
#       to the sample 0.
#

#x_train = x_train.astype(numpy.float32)

callbacks = [
    keras.callbacks.TensorBoard(log_dir = './tb/%s' % (model_name)),
    keras.callbacks.ModelCheckpoint(filepath = './chkp/%s_{epoch:03d}' % (model_name), save_freq = 'epoch')
]

#history = model.fit(data_generator.flow(x_train, y_train, batch_size = batch_size), epochs = nr_epochs, verbose = 1, callbacks = callbacks)
history = model.fit(data_generator.flow(x_train, y_train, batch_size = batch_size), epochs = nr_epochs, steps_per_epoch = len(x_train) / batch_size, verbose = 1, callbacks = callbacks)

print('\nTraining History ...\n\n{}\n'.format(history.history))

##
# ----------------------------------------------------------------------------
# evaluate model-training performance

print('\nPerformance Evaluation ...\n')

# returns loss and metrics
eval_report = model.evaluate(x_test, y_test, batch_size = batch_size)

print('\nEvaluation report for metrics ({}) ...'.format(model.metrics_names))

print(' {}'.format(eval_report))

##
# ----------------------------------------------------------------------------
# generate per-sample predictions
#
#  - For small amount of inputs that fit in one batch, directly using __call__ is recommended for faster
#    execution, e.g., model(x), or model(x, training=False) if you have layers such as
#    tf.keras.layers.BatchNormalization that behaves differently during inference.
#

print('\nInference Check ...\n')

# returns loss and metrics
predictions = model.predict(x_test)#, batch_size = batch_size)

print('predictions shape = {}\n'.format(predictions.shape))

print('prediction of   x_test[:20] = y_pred[:20] = {}\n'.format(tf.argmax(predictions[0:20, :], 1)))
print('ground truth of x_test[:20] = y_test[:20] = {}\n'.format(y_test[:20]))

# expected output ...
#
# Epoch 1/5
# 10800/10800 [==============================] - 23s 2ms/step - loss: 0.1650 - accuracy: 0.9490 - val_loss: 0.0564 - val_accuracy: 0.9820
# Epoch 2/5
# 10800/10800 [==============================] - 24s 2ms/step - loss: 0.0565 - accuracy: 0.9827 - val_loss: 0.0472 - val_accuracy: 0.9863
# Epoch 3/5
# 10800/10800 [==============================] - 25s 2ms/step - loss: 0.0360 - accuracy: 0.9888 - val_loss: 0.0600 - val_accuracy: 0.9843
# Epoch 4/5
# 10800/10800 [==============================] - 22s 2ms/step - loss: 0.0237 - accuracy: 0.9922 - val_loss: 0.0599 - val_accuracy: 0.9857
# Epoch 5/5
# 10800/10800 [==============================] - 22s 2ms/step - loss: 0.0182 - accuracy: 0.9940 - val_loss: 0.0549 - val_accuracy: 0.9878
#
# Training History ...
#
# {'loss': [0.16497066617012024, 0.05647317320108414, 0.036025892943143845, 0.023727182298898697, 0.018161695450544357], 'accuracy': [0.9489629864692688, 0.9826666712760925, 0.9888148307800293, 0.9921666383743286, 0.9940370321273804], 'val_loss': [0.05641787871718407, 0.047156356275081635, 0.05997416004538536, 0.059918370097875595, 0.05494614318013191], 'val_accuracy': [0.9819999933242798, 0.9863333106040955, 0.984333336353302, 0.9856666922569275, 0.9878333210945129]}
#
#
# Performance Evaluation ...
#
# 2000/2000 [==============================] - 2s 1ms/step - loss: 0.0555 - accuracy: 0.9862
#
# Evaluation report for metrics (['loss', 'accuracy']) ...
#  [0.055539220571517944, 0.9861999750137329]
#
# Inference Check ...
#
# predictions shape = (20, 10)
#
# prediction of   x_test[:20] = y_pred[:20] = [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
#
# ground truth of x_test[:20] = y_test[:20] = [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]

##
# ----------------------------------------------------------------------------
# manually inspect mismatching items ..........................|
#                                                              v
# prediction of   x_test[:20] = y_pred[:20] = [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
# ground truth of x_test[:20] = y_test[:20] = [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
#

cv2.namedWindow('mnist_image', flags = cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('expected',    flags = cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('best_guess',  flags = cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('second_best', flags = cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('dummy',       flags = cv2.WINDOW_AUTOSIZE)

img_expected = numpy.zeros((x_test.shape[1], x_test.shape[2]), dtype = numpy.uint8)
img_label1   = numpy.zeros((x_test.shape[1], x_test.shape[2]), dtype = numpy.uint8)
img_label2   = numpy.zeros((x_test.shape[1], x_test.shape[2]), dtype = numpy.uint8)
img_dummy    = numpy.zeros((x_test.shape[1], x_test.shape[2]), dtype = numpy.uint8)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

predicted = tf.argmax(predictions, 1)

for i in range(len(y_test)) :

    label_predicted = predicted[i]
    label_second_best = numpy.argsort(predictions[i])[-2]
    label = y_test[i]

    if (label_predicted != label) :

        print('[{}] label){} predicted){} second_best){} logit = {}'.format(i, label, label_predicted, label_second_best, predictions[i]))

        # ---------

        img_digit = (x_test[i, :, :] * 255.0).astype(numpy.uint8)

        cv2.imshow('mnist_image', img_digit)
        cv2.moveWindow('mnist_image', 10, 10)

        # ---------

        img_expected.fill(120)
        cv2.putText(img_expected, text = '%u' % label,
                    org = (5, 22), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8,
                    color = 255, lineType = cv2.LINE_AA)

        cv2.imshow('expected', img_expected)
        cv2.moveWindow('expected', 400, 10)

        img_label1.fill(120)
        cv2.putText(img_label1, text = '%u' % label_predicted,
                    org = (5, 22), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8,
                    color = 255, lineType = cv2.LINE_AA)

        cv2.imshow('best_guess', img_label1)
        cv2.moveWindow('best_guess', 10, 200)

        img_label2.fill(120)
        cv2.putText(img_label2, text = '%u' % label_second_best,
                    org = (5, 22), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8,
                    color = 255, lineType = cv2.LINE_AA)

        cv2.imshow('second_best', img_label2)
        cv2.moveWindow('second_best', 400, 200)

        # ---------
        # refersh a dummy window temporarily fixes issue of delayed update of the label window
        #img_dummy.fill(0)
        cv2.imshow('dummy', img_dummy)
        cv2.moveWindow('dummy', 10, 400)

        # ---------

        quit = False

        while True :
            key = cv2.waitKey(1)
            if (key & 0xff) == ord('q') :
                quit = True
                break
            elif (key & 0xff) == ord(' ') :
                break

        if (quit) :
            break

##
# ----------------------------------------------------------------------------

#tf.logging.set_verbosity(log_level)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
