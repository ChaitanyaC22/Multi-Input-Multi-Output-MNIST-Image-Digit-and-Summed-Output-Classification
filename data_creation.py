# -*- coding: utf-8 -*-
# Custom Module 2: Data Creation
'''
This module is used for pre-processing the data inputs and outputs in the
desired format. (The random number inputs (0-9) along with the true image
labels (0-9) and summed output true labels (0-18) are encoded in the proper
format. Additionally, the model/network retrieves the MNIST image data
via Image Data Generators, which are defined in this module.)
'''

# Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from google.colab.patches import cv2_imshow
import tensorflow
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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




# Class for creating data
class data_creation:

  def __init__(self, training_imgs, training_imgs_labels, 
                validation_imgs, validation_imgs_labels, 
                test_imgs, test_imgs_labels):

    self.training_imgs = training_imgs
    self.training_imgs_labels = training_imgs_labels
    self.validation_imgs = validation_imgs
    self.validation_imgs_labels = validation_imgs_labels
    self.test_imgs = test_imgs
    self.test_imgs_labels = test_imgs_labels

    # Creating variables to avoid generating different random numbers on every run
    self.training_randnum_list = to_categorical(np.random.randint(0, 10, 
        size=(training_imgs_labels.shape[0],1), dtype= int), num_classes=10)
    self.validation_randnum_list = to_categorical(np.random.randint(0, 10, 
        size=(validation_imgs_labels.shape[0],1), dtype= int), num_classes=10)
    self.test_randnum_list = to_categorical(np.random.randint(0, 10, 
        size=(test_imgs_labels.shape[0],1), dtype= int), num_classes=10)      


  def reshape_imgs_data(self):
    ''' 
    This function takes all the training, validation and test
    image datasets and returns the reshaped dataset. This helps
    in avoiding dimension error during model building phase.
    '''
    reshaped_training_imgs = self.training_imgs.reshape(self.training_imgs.shape[0], 
                                                  self.training_imgs.shape[1], 
                                                  self.training_imgs.shape[2], 1)
    reshaped_validation_imgs = self.validation_imgs.reshape(self.validation_imgs.shape[0], 
                                                  self.validation_imgs.shape[1], 
                                                  self.validation_imgs.shape[2], 1)
    reshaped_test_imgs = self.test_imgs.reshape(self.test_imgs.shape[0], 
                                                  self.test_imgs.shape[1], 
                                                  self.test_imgs.shape[2], 1)

    print();print('-'*100)
    print("Reshaped Training Image Dataset Dimensions   : ", reshaped_training_imgs.shape)
    print("Reshaped Validation Image Dataset Dimensions : ", reshaped_validation_imgs.shape)
    print("Reshaped Test Image Dataset Dimensions       : ", reshaped_test_imgs.shape)
    print('-'*100)

    return reshaped_training_imgs, reshaped_validation_imgs, reshaped_test_imgs


  def reshape_imgs_labels(self, labels_print=True):
    '''
    This function reshapes the image labels to avoid dimension error
    '''
    r_training_imgs_labels, r_validation_imgs_labels, r_test_imgs_labels = self.training_imgs_labels.reshape(self.training_imgs_labels.shape[0],-1), \
                self.validation_imgs_labels.reshape(self.validation_imgs_labels.shape[0],-1), \
                self.test_imgs_labels.reshape(self.test_imgs_labels.shape[0],-1)

    if labels_print==False:
      return r_training_imgs_labels, r_validation_imgs_labels, r_test_imgs_labels
    if labels_print==True:   
      print();print("-"*100)
      print("Reshaped Training Image Labels      : ", r_training_imgs_labels.shape)
      print("Reshaped Validation Image Labels    : ", r_validation_imgs_labels.shape)
      print("Reshaped Test Image Labels          : ", r_test_imgs_labels.shape);print()
      print("-"*100)

    

  def categorical_encoded_imgs_labels(self):
    '''
    This function reshapes the image labels and returns one-hot encoded
    labels for image classes (0-9)
    '''
    r_training_imgs_labels, r_validation_imgs_labels, r_test_imgs_labels = self.reshape_imgs_labels(labels_print=False)
    encoded_training_imgs_labels = to_categorical(r_training_imgs_labels, num_classes=10)
    encoded_validation_imgs_labels = to_categorical(r_validation_imgs_labels, num_classes=10)
    encoded_test_imgs_labels = to_categorical(r_test_imgs_labels, num_classes=10)

    print();print("-"*100)
    print("Training Image Labels - One-hot encoded Matrix Size   : ", encoded_training_imgs_labels.shape)
    print("Validation Image Labels - One-hot encoded Matrix Size : ", encoded_validation_imgs_labels.shape)
    print("Test Image Labels - One-hot encoded Matrix Size       : ", encoded_test_imgs_labels.shape)  
    print("-"*100)

    return encoded_training_imgs_labels, encoded_validation_imgs_labels, encoded_test_imgs_labels


  def randnum_encoded_list(self, randnum_labels_print=True):
    '''
    This function transforms the random number list of integers between 0-9
    into one-hot encoded labels (for creating random number inputs to the model)
    '''
    training_randnum_list = self.training_randnum_list.copy()
    validation_randnum_list = self.validation_randnum_list.copy()
    test_randnum_list = self.test_randnum_list.copy()

    if randnum_labels_print==True:
      print();print("-"*100)
      print("(One-Hot Encoded) Training Random Number List Dimensions   : ", training_randnum_list.shape)
      print("(One-Hot Encoded) Training Random Number List (5 samples)  :\n", training_randnum_list[:5]);print()

      print("(One-Hot Encoded) Validation Random Number List Dimensions : ", validation_randnum_list.shape)
      print("(One-Hot Encoded) Validation Random Number List (5 samples):\n", validation_randnum_list[:5]);print()

      print("(One-Hot Encoded) Test Random Number List Dimensions       : ", test_randnum_list.shape)
      print("(One-Hot Encoded) Test Random Number List (5 samples)      :\n", test_randnum_list[:5])
      print("-"*100)
    if randnum_labels_print==False:
      pass

    return training_randnum_list, validation_randnum_list, test_randnum_list


  def summed_output_encoded_true_labels(self, summed_encoded_print = True):
    '''
    This function takes in one-hot encoded random number list
    along with true image int labels and returns true summed
    output labels ranging from numbers (0-18) - 19 classes
    '''
    # Random number list (one-hot encoded)
    training_randnum_list, validation_randnum_list, test_randnum_list = self.randnum_encoded_list(randnum_labels_print=False)
  
    # Convert one-hot encoded random number list to integer list
    training_randnum_int_list = np.array([np.argmax(item) for item in training_randnum_list])
    validation_randnum_int_list = np.array([np.argmax(item) for item in validation_randnum_list])
    test_randnum_int_list = np.array([np.argmax(item) for item in test_randnum_list])

    # Adding random number int list with true image int labels -> to get true summed output (0-18) 19 labels
    training_sum_labels = training_randnum_int_list + self.training_imgs_labels
    validation_sum_labels = validation_randnum_int_list + self.validation_imgs_labels
    test_sum_labels = test_randnum_int_list + self.test_imgs_labels  
    
    # One-hot encoded summed output labels (0-18) 19 classes
    training_sum_labels_encoded = to_categorical(training_sum_labels, num_classes = 19)
    validation_sum_labels_encoded = to_categorical(validation_sum_labels, num_classes = 19)
    test_sum_labels_encoded = to_categorical(test_sum_labels, num_classes = 19)

    if summed_encoded_print == True:
      print();print("-"*100)
      print("Training Random Number (5 samples)                 : ", 
                                                  training_randnum_int_list[:5])
      print("Training Image Label Number (5 samples)            : ", 
                                                  self.training_imgs_labels[:5])
      print("Training Sum Labels (5 samples)                    : \n", 
                                        training_sum_labels[:5]);print();print()
      print("(One-Hot Encoded) Training Sum Labels (5 samples)  : \n", 
                                training_sum_labels_encoded[:5])
      print("Training Sum Labels (One-Hot Encoded)  Dimensions  : ",  
                              training_sum_labels_encoded.shape);print();print()

      print("Validation Random Number (5 samples)               : ", 
                                                validation_randnum_int_list[:5])
      print("Validation Image Label Number (5 samples)          : ", 
                                                self.validation_imgs_labels[:5])
      print("Validation Sum Labels (5 samples)                  : \n", 
                                      validation_sum_labels[:5]);print();print()      
      print("(One-Hot Encoded) Validation Sum Labels (5 samples): \n", 
                                validation_sum_labels_encoded[:5])
      print("Validation Sum Labels (One-Hot Encoded)  Dimensions: ", 
                            validation_sum_labels_encoded.shape);print();print()
      

      print("Test Random Number (5 samples)                     : ", 
                                                      test_randnum_int_list[:5])
      print("Test Image Label Number (5 samples)                : ", 
                                                      self.test_imgs_labels[:5])
      print("Test Sum Labels (5 samples)                        : \n", 
                                            test_sum_labels[:5]);print();print()
      print("(One-Hot Encoded) Test Sum Labels (5 samples)      : \n", 
                                test_sum_labels_encoded[:5])
      print("Test Sum Labels (One-Hot Encoded)  Dimensions      : ", 
                                test_sum_labels_encoded.shape);print();print()

      print("-"*100)
    
    if summed_encoded_print == False:
      pass

    return training_sum_labels_encoded, validation_sum_labels_encoded, test_sum_labels_encoded






class mnist_image_data_generator:
  '''
  Image Data Generators: Three data generators are created for respective image 
  datasets and the images are rescaled/normalized.
  '''

  def __init__(self, training_imgs, training_imgs_labels,
                validation_imgs, validation_imgs_labels,
                test_imgs, test_imgs_labels,
                batch_size):

    self.training_imgs =  training_imgs
    self.training_imgs_labels = training_imgs_labels
    self.validation_imgs = validation_imgs
    self.validation_imgs_labels = validation_imgs_labels
    self.test_imgs = test_imgs
    self.test_imgs_labels = test_imgs_labels
    self.BATCH_SIZE = batch_size

  def training_imgs_generator(self):
    '''
    This function creates an image data generator for training image dataset with
    data augmentation and returns preprocessed training images for the input feed 
    to the model in batches

    (Data Augmentation: Augmentation is conducted on the training image data samples. 
    Random training samples are automatically subjected to transformations like:
    rotation range (degree range: 5) along with height and width shift range of 5% 
    and zoom_range of 5%, to introduce variation in the training image dataset.)
    '''
    training_imgs_datagen = ImageDataGenerator(
                    rescale=1.0/255,                                  # Normalizing Training Image Data                    
                    rotation_range= 5,                               # Rotation range (5 degrees)-> Keeping it minimal else there will be confusion identifying 6 and 9 numbers
                    fill_mode="nearest",           
                    width_shift_range= 0.05,                           # Horizontal shift (5% of width)
                    height_shift_range= 0.05,                          # Vertical shift (5% of height)
                    zoom_range = 0.05)
    
    # Generate batches using Image Data Generator
    training_images_generator = training_imgs_datagen.flow(
                                      x = self.training_imgs,                     # Inputs
                                      y = self.training_imgs_labels,              # Outputs
                                      batch_size = self.BATCH_SIZE,
                                      shuffle = True,                                                      
                                      seed=42)

    return training_images_generator



  def validation_imgs_generator(self):
    '''
    This function creates a data generator for validation image dataset
    and normalizes/rescales the input image data to return scaled images
    '''
    validation_imgs_datagen = ImageDataGenerator(rescale=1.0/255) 

    # Generate batches using Image Data Generator
    validation_images_generator = validation_imgs_datagen.flow(
                                      x = self.validation_imgs,                  # Inputs
                                      y = self.validation_imgs_labels,           # Outputs
                                      batch_size = self.BATCH_SIZE,
                                      shuffle = False,                                                       
                                      seed=42) 


    return validation_images_generator



  def test_imgs_generator(self):
    '''
    This function creates a data generator for test image dataset
    and normalizes/rescales the input image data to return scaled images
    '''
    test_imgs_datagen = ImageDataGenerator(rescale=1.0/255) 

    # Generate batches using Image Data Generator
    test_images_generator = test_imgs_datagen.flow(
                                      x = self.test_imgs,                  # Inputs
                                      y = self.test_imgs_labels,           # Outputs
                                      batch_size = self.BATCH_SIZE,
                                      shuffle = False,                                                       
                                      seed=42) 

    return test_images_generator





  








    


  