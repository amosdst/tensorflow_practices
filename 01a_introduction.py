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
# A Typical Feed-Forward Network
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
#  - The number of nodes in the input layer does not need to match the shape of the input vector.
#  - For a 10 nodes inputs layer and a 13 element input vector, there will be 130 connections between the
#    input vector and the input layer.
#  - It looks there are no weighting and bias values within the connections between the input vector and
#    the input layer, since there is no associated trainable parameters in it's model summary.
#    => confirmed in the graph diagram './logs/01a_introduction/input_vector_to_input_layer.png'
#
# Q2:
#  - The accuracy value of the training report differs greatly in between the two cases that the 'Dropout'
#    layer was placed before and after the final 'Dense' layer.  It looks the placement of the dropout layer
#    do affect the final training accuracy, and maybe the training time.  Refer to the 'hint[1]' section
#    bellow.
#  - Is there any generic way to state or any quantitative measurement schemes to tell 'what is a good
#    training result or a good performance' ?
#
# Q3:
#  - Is there good rule of thub way about determining the number of neurons of each layer to start with ?
#    (such as from hints of targeting number of classifications ?)
#

import tensorflow.keras as keras

class LOCAL_MODELING_SCHEME(IntEnum) :
    SEQUENTIAL = 1
    FUNCTIONAL = 2

modeling_scheme = LOCAL_MODELING_SCHEME.SEQUENTIAL
#modeling_scheme = LOCAL_MODELING_SCHEME.FUNCTIONAL

model_name = '01a_introduction'
batch_size = 16
nr_epochs = 16

if (modeling_scheme == LOCAL_MODELING_SCHEME.SEQUENTIAL) :

    # ----------------------------------------------------------------------------
    # The Sequential API Approach

    print('\nSequential Modeling Scheme\n')

    # the model
    model = keras.Sequential()

    # the input layers
    model.add(keras.Input(shape = (x_train.shape[1], x_train.shape[2]), batch_size = batch_size))
    model.add(keras.layers.Flatten(data_format = 'channels_last'))
    #model.add(keras.layers.Dense(100, input_shape = (batch_size, x_train.shape[1] * x_train.shape[2]), activation = 'relu'))
    #model.add(keras.layers.Dense(100, activation = 'relu'))
    model.add(keras.layers.Dense(100, activation = 'relu'))

    # the hidden layers
    model.add(keras.layers.Dense(200, activation = 'sigmoid'))

    # the output layer
    #  hint: accuracy was increased when the 'Dropout' layer was placed before the final 'Dense' layer
    #        see 'hint[1]' comments at the training section bellow
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10))
    model.add(keras.layers.Softmax())

    # Model: "sequential"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # flatten (Flatten)            (5, 784)                  0
    # _________________________________________________________________
    # dense (Dense)                (5, 100)                  78500
    # _________________________________________________________________
    # dense_1 (Dense)              (5, 200)                  20200
    # _________________________________________________________________
    # dense_2 (Dense)              (5, 10)                   2010
    # _________________________________________________________________
    # dropout (Dropout)            (5, 10)                   0
    # _________________________________________________________________
    # softmax (Softmax)            (5, 10)                   0
    # =================================================================
    # Total params: 100,710
    # Trainable params: 100,710
    # Non-trainable params: 0
    #
    model.summary()

elif (modeling_scheme == LOCAL_MODELING_SCHEME.FUNCTIONAL) :

    # ----------------------------------------------------------------------------
    # The Functional API Approach

    print('\nFunctional Modeling Scheme\n')

    # the input layers
    in_data    = keras.Input(shape = (x_train.shape[1], x_train.shape[2]), batch_size = batch_size)
    in_vector  = keras.layers.Flatten(data_format = 'channels_last')(in_data)
    in_layer   = keras.layers.Dense(100)(in_vector)
    in_layer_f = keras.activations.relu(in_layer)

    # the hidden layers
    h_layer    = keras.layers.Dense(200, activation = 'sigmoid')(in_layer_f)

    # the output layers
    out_drop   = keras.layers.Dropout(rate = 0.2)(h_layer)
    out_layer  = keras.layers.Dense(10, activation = 'softmax')(out_drop)

    # the model
    model = keras.Model(in_data, out_layer)

    # Model: "model"
    # _________________________________________________________________
    # Layer (type)                 Output Shape              Param #
    # =================================================================
    # input_1 (InputLayer)         [(5, 28, 28)]             0
    # _________________________________________________________________
    # flatten (Flatten)            (5, 784)                  0
    # _________________________________________________________________
    # dense (Dense)                (5, 100)                  78500
    # _________________________________________________________________
    # tf_op_layer_Relu (TensorFlow [(5, 100)]                0
    # _________________________________________________________________
    # dense_1 (Dense)              (5, 200)                  20200
    # _________________________________________________________________
    # dropout (Dropout)            (5, 200)                  0
    # _________________________________________________________________
    # dense_2 (Dense)              (5, 10)                   2010
    # =================================================================
    # Total params: 100,710
    # Trainable params: 100,710
    # Non-trainable params: 0
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
#    'binary_crossentropy'
#    'categorical_crossentropy'
#    'sparse_categorical_crossentropy'
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

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

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
#  example output (sequential)
#
#    Epoch 1/5
#    10800/10800 [==============================] - 16s 1ms/step - loss: 0.4634 - accuracy: 0.8178 - val_loss: 0.1070 - val_accuracy: 0.9665
#    Epoch 2/5
#    10800/10800 [==============================] - 15s 1ms/step - loss: 0.3130 - accuracy: 0.8579 - val_loss: 0.0828 - val_accuracy: 0.9743
#    E poch 3/5
#    10800/10800 [==============================] - 14s 1ms/step - loss: 0.2780 - accuracy: 0.8689 - val_loss: 0.0774 - val_accuracy: 0.9777
#    Epoch 4/5
#    10800/10800 [==============================] - 14s 1ms/step - loss: 0.2593 - accuracy: 0.8724 - val_loss: 0.0902 - val_accuracy: 0.9747
#    Epoch 5/5
#    10800/10800 [==============================] - 14s 1ms/step - loss: 0.2440 - accuracy: 0.8781 - val_loss: 0.0863 - val_accuracy: 0.9770
#    Out[12]: <tensorflow.python.keras.callbacks.History at 0x7f339c0df080>
#
#    Process finished with exit code 0
#
#  example output (functional)
#
#    Epoch 1/5
#    10800/10800 [==============================] - 15s 1ms/step - loss: 0.2507 - accuracy: 0.9229 - val_loss: 0.1062 - val_accuracy: 0.9685
#    Epoch 2/5
#    10800/10800 [==============================] - 15s 1ms/step - loss: 0.1115 - accuracy: 0.9662 - val_loss: 0.0891 - val_accuracy: 0.9723
#    Epoch 3/5
#    10800/10800 [==============================] - 15s 1ms/step - loss: 0.0795 - accuracy: 0.9752 - val_loss: 0.0913 - val_accuracy: 0.9718
#    Epoch 4/5
#    10800/10800 [==============================] - 15s 1ms/step - loss: 0.0635 - accuracy: 0.9803 - val_loss: 0.0756 - val_accuracy: 0.9780
#    Epoch 5/5
#    10800/10800 [==============================] - 15s 1ms/step - loss: 0.0511 - accuracy: 0.9835 - val_loss: 0.0724 - val_accuracy: 0.9817
#
#    Process finished with exit code 0
#

#x_train = x_train.astype(numpy.float32)

callbacks = [
    keras.callbacks.TensorBoard(log_dir = './tb/%s' % (model_name))
    #keras.callbacks.ModelCheckpoint(filepath = './chkp/%s_{epoch:03d}' % (model_name), period = 1, save_freq = 'epoch')
]

history = model.fit(x_train, y_train, epochs = nr_epochs, batch_size = batch_size, validation_split = 0.1, verbose = 1, callbacks = callbacks)

print('\nTraining History ...\n\n{}\n'.format(history.history))

# hint[1] accuracy was increased when the 'Dropout' layer was placed before the 'final 'Dense' output layer
#
# case 1 (lower accuracy value)
#
#  - the 'Dropout' layer was placed after the final 'Dense' output layer
#
#      model.add(keras.layers.Dense(10))
#      model.add(keras.layers.Dropout(0.2))
#      model.add(keras.layers.Softmax())
#
#  - model summary and training briefing
#
#      Model: "sequential"
#      _________________________________________________________________
#      Layer (type)                 Output Shape              Param #
#      =================================================================
#      flatten (Flatten)            (5, 784)                  0
#      _________________________________________________________________
#      dense (Dense)                (5, 100)                  78500
#      _________________________________________________________________
#      dense_1 (Dense)              (5, 200)                  20200
#      _________________________________________________________________
#      dense_2 (Dense)              (5, 10)                   2010
#      _________________________________________________________________
#      dropout (Dropout)            (5, 10)                   0
#      _________________________________________________________________
#      softmax (Softmax)            (5, 10)                   0
#      =================================================================
#      Total params: 100,710
#      Trainable params: 100,710
#      Non-trainable params: 0
#
#      Epoch 1/5
#      10800/10800 [==============================] - 16s 1ms/step - loss: 0.4634 - accuracy: 0.8178 - val_loss: 0.1070 - val_accuracy: 0.9665
#      Epoch 2/5
#      10800/10800 [==============================] - 15s 1ms/step - loss: 0.3130 - accuracy: 0.8579 - val_loss: 0.0828 - val_accuracy: 0.9743
#      Epoch 3/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.2780 - accuracy: 0.8689 - val_loss: 0.0774 - val_accuracy: 0.9777
#      Epoch 4/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.2593 - accuracy: 0.8724 - val_loss: 0.0902 - val_accuracy: 0.9747
#      Epoch 5/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.2440 - accuracy: 0.8781 - val_loss: 0.0863 - val_accuracy: 0.9770
#      Out[12]: <tensorflow.python.keras.callbacks.History at 0x7f339c0df080>
#
#      Performance Evaluation ...
#
#      2000/2000 [==============================] - 2s 1ms/step - loss: 0.0860 - accuracy: 0.9767
#
#      Evaluation report for metrics (['loss', 'accuracy']) ...
#       [0.08598091453313828, 0.9767000079154968]
#
#      Inference Check ...
#
#      predictions shape = (20, 10)
#
#      prediction of   x_test[:20] = y_pred[:20] = [7 2 1 0 4 1 4 9 6 9 0 6 9 0 1 5 9 7 3 4]
#
#      ground truth of x_test[:20] = y_test[:20] = [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
#
# case 2 (higher accuracy value)
#
#  - the 'Dropout' layer was placed before the final 'Dense' output layer
#
#      model.add(keras.layers.Dropout(0.2))
#      model.add(keras.layers.Dense(10))
#      model.add(keras.layers.Softmax())
#
#  - model summary and training briefing
#
#      Model: "sequential"
#      _________________________________________________________________
#      Layer (type)                 Output Shape              Param #
#      =================================================================
#      flatten (Flatten)            (5, 784)                  0
#      _________________________________________________________________
#      dense (Dense)                (5, 100)                  78500
#      _________________________________________________________________
#      dense_1 (Dense)              (5, 200)                  20200
#      _________________________________________________________________
#      dropout (Dropout)            (5, 200)                  0
#      _________________________________________________________________
#      dense_2 (Dense)              (5, 10)                   2010
#      _________________________________________________________________
#      softmax (Softmax)            (5, 10)                   0
#      =================================================================
#      Total params: 100,710
#      Trainable params: 100,710
#      Non-trainable params: 0
#
#      Epoch 1/5
#      10800/10800 [==============================] - 15s 1ms/step - loss: 0.2472 - accuracy: 0.9251 - val_loss: 0.1285 - val_accuracy: 0.9620
#      Epoch 2/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.1096 - accuracy: 0.9664 - val_loss: 0.0842 - val_accuracy: 0.9730
#      Epoch 3/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.0789 - accuracy: 0.9763 - val_loss: 0.0779 - val_accuracy: 0.9772
#      Epoch 4/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.0625 - accuracy: 0.9803 - val_loss: 0.1003 - val_accuracy: 0.9713
#      Epoch 5/5
#      10800/10800 [==============================] - 15s 1ms/step - loss: 0.0499 - accuracy: 0.9844 - val_loss: 0.0752 - val_accuracy: 0.9785
#
#      Performance Evaluation ...
#
#      2000/2000 [==============================] - 2s 1ms/step - loss: 0.0860 - accuracy: 0.9767
#
#      Evaluation report for metrics (['loss', 'accuracy']) ...
#       [0.08598091453313828, 0.9767000079154968]
#
#      Inference Check ...
#
#      predictions shape = (20, 10)
#
#      prediction of   x_test[:20] = y_pred[:20] = [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
#
#      ground truth of x_test[:20] = y_test[:20] = [7 2 1 0 4 1 4 9 5 9 0 6 9 0 1 5 9 7 3 4]
#
# case 3 (also higher accuracy value)
#
#  - no 'Dropout' layer
#
#      model.add(keras.layers.Dense(10))
#      model.add(keras.layers.Softmax())
#
#  - model summary and training briefing
#
#      Model: "sequential"
#      _________________________________________________________________
#      Layer (type)                 Output Shape              Param #
#      =================================================================
#      flatten (Flatten)            (5, 784)                  0
#      _________________________________________________________________
#      dense (Dense)                (5, 100)                  78500
#      _________________________________________________________________
#      dense_1 (Dense)              (5, 200)                  20200
#      _________________________________________________________________
#      dense_2 (Dense)              (5, 10)                   2010
#      _________________________________________________________________
#      softmax (Softmax)            (5, 10)                   0
#      =================================================================
#      Total params: 100,710
#      Trainable params: 100,710
#      Non-trainable params: 0
#
#      Epoch 1/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.2307 - accuracy: 0.9301 - val_loss: 0.1162 - val_accuracy: 0.9650
#      Epoch 2/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.1017 - accuracy: 0.9684 - val_loss: 0.0912 - val_accuracy: 0.9697
#      Epoch 3/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.0728 - accuracy: 0.9766 - val_loss: 0.0988 - val_accuracy: 0.9700
#      Epoch 4/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.0548 - accuracy: 0.9823 - val_loss: 0.0756 - val_accuracy: 0.9787
#      Epoch 5/5
#      10800/10800 [==============================] - 14s 1ms/step - loss: 0.0438 - accuracy: 0.9860 - val_loss: 0.0762 - val_accuracy: 0.9795
#
#      Process finished with exit code 0
#

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
