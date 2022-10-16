# -*- coding: utf-8 -*-
# Custom Module 3: Network Architecture
'''
This module is used to build the network architecture
of the combined (MNIST + Summed Output) classification model
'''

# Necessary Libraries
from pathlib import Path
import os
import os.path
import numpy as np
import pandas as pd
import cv2
from google.colab.patches import cv2_imshow
import time
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation, concatenate
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu
from keras.layers.core import Reshape


# For Reproducible Results: reset seeds and reset global tensorflow sessions
# Set seed_value
seed_value = 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)  

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
if tf.__version__[0] == '2':
    tf.compat.v1.set_random_seed(seed_value)
    tf.random.set_seed(seed_value)
else:
    tf.set_random_seed(seed_value)

# 5. For working on GPUs from "TensorFlow Determinism"
os.environ["TF_DETERMINISTIC_OPS"] = "1" 

# 6. Configure a new global `tensorflow` session
if tf.__version__[0] == '2': 
  session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
  tf.compat.v1.keras.backend.set_session(sess)
else:
  session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
  tf.compat.v1.keras.backend.set_session(sess)

  
  


# Class to store model evaluation results
class network_architecture:

  # init method or constructor
  def __init__(self, INPUT_IMG_SIZE, NUM_OF_IMG_CLASSES, INPUT_RANDNUM_SIZE, NUM_OF_SUM_CLASSES):
    self.INPUT_IMG_SIZE = INPUT_IMG_SIZE
    self.NUM_OF_IMG_CLASSES = NUM_OF_IMG_CLASSES
    self.INPUT_RANDNUM_SIZE = INPUT_RANDNUM_SIZE
    self.NUM_OF_SUM_CLASSES = NUM_OF_SUM_CLASSES

  # Method 1: To build an image classification network that predicts numbers 0-9 from input images
  def build_img_classification_model(self):
    # Model Inputs_A (Initialize a tensor for 'inputs_a')
    inputs_a = Input(shape = self.INPUT_IMG_SIZE, name="inputs_a")

    '''
    Building a CNN model for classifying MNIST Dataset Images
    '''
    # Conv2D-> Activation(ReLU)-> Conv2D-> BN-> Activation(ReLU) -> MaxPooling2D-> Dropout
    x = Conv2D(32,(5,5), activation='relu')(inputs_a)
    x = Conv2D(32,(5,5))(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    # Conv2D-> Activation(ReLU)-> Conv2D-> BN-> Activation(ReLU) -> MaxPooling2D-> Dropout
    x = Conv2D(64,(3,3), activation='relu')(x)
    x = Conv2D(64,(3,3))(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    # Flattens the input
    x = Flatten()(x)

    # Dense/FC Layer -> BN -> Activation(ReLU) -> Dropout
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Dropout(0.25)(x)

    # Dense/FC Layer -> BN -> Activation(ReLU) -> Dropout
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)

    # Dense/FC Layer -> BN -> Activation(ReLU) -> Dropout
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Dropout(0.25)(x)
    x = Model(inputs= inputs_a, outputs= x)

    return x

  # Method 2: Building a complete network (MNIST Image Classification + Summed Output)
  def build_img_class_n_summed_output_model(self):

    '''
    Build a complete model that classifies the input image to return the predicted label
    as well as returns sum of the predicted (label) number and a random number input.
    This sum is addressed as summed_output.
    
    (Inputs):
    self.INPUT_IMG_SIZE: size of input image to be fed to the network
    self.NUM_OF_IMG_CLASSES: number of image class labels (10)
    self.INPUT_RANDNUM_SIZE: random number input (one-hot encoded 0-9)
    self.NUM_OF_SUM_CLASSES: number of summed output class labels (19)

    (Output): Combined MNIST classification + Summed output prediction model
    '''

    # Image classifation branch 1 (before classifying images)
    img_classification_branch = self.build_img_classification_model()

    # Classify MNIST Images
    class_img_outputs = Dense(self.NUM_OF_IMG_CLASSES, activation= 'softmax',                        # self.NUM_OF_IMG_CLASSES: 10 categories 0-9
                              name='class_img_outputs')(img_classification_branch.output)  
  
    # Predicted Label - tensor of indices (for feeding to the network)
    label_predicted_idx = tf.math.argmax(class_img_outputs, 1)                    
    # Output from model to be added with the encoded random number (generates one-hot encoded tensor of image classifier output)
    class_img_outputs_encoded = tf.one_hot(label_predicted_idx, self.NUM_OF_IMG_CLASSES)


    # Model Inputs_B: For Random Number Inputs (Initialize a tensor for 'inputs_b')
    inputs_b = Input(shape = self.INPUT_RANDNUM_SIZE, name="inputs_b")


    # Combine the output of the two branches (Concatenate outputs of both the branches)
    # Note: 'concatenate': used in Functional API, 'Concatenate': used in Sequential
    concat_inputs = concatenate([inputs_b, class_img_outputs_encoded], axis=-1)    

    z = Flatten()(concat_inputs)

    # Creating a list for number of neuron units in same number of repetitive Dense/FC Layers
    neuron_units_list = [64, 128, 256, 64, 38]

    # Looping and connecting Dense/FC Layers
    for n_unit in neuron_units_list: 
      z = Dense(n_unit)(z)
      z = BatchNormalization()(z)
      z = Activation(relu)(z)

    # Classify Summed Labels
    class_summed_outputs = Dense(self.NUM_OF_SUM_CLASSES, activation= 'softmax',           # self.NUM_OF_SUM_CLASSES:19 categories 0-18
                                name= 'class_summed_outputs')(z) 

    # Build Final Combined Model
    model = Model(inputs=[img_classification_branch.input, inputs_b],
                  outputs=[class_img_outputs, class_summed_outputs],
                  name='Combined_MNIST_Image_Classification_and_Summed_Output_Model')
    
    return model
    
  














  

