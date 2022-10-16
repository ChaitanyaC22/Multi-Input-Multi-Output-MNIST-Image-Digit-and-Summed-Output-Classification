# -*- coding: utf-8 -*-
# Custom Module 4: 'evaluate_func.py'
'''
This module comprises functions to pre-process test input, display
model training and evaluation results
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
class eval_func:
    
  @staticmethod
  def evaluate_model_test_set(model, batch_size, 
                              inputs_a, inputs_b, 
                              class_img_outputs, class_summed_outputs):
    ''' This function takes the trained model as input and returns model metric 
    scores after evaluation on test set
    (inputs) 
    model -> trained model
    batch_size: number of datapoints in a batch
    inputs_a: test image data preprocessed inputs, 
    inputs_b= test random number list (one-hot encoded 0-9)
    class_img_outputs = test image predicted labels (one-hot encoded 0-9)
    class_summed_outputs = summed output predicted labels (one-hot encoded 0-18)
    '''

    scores = model.evaluate(
                            x = {'inputs_a': inputs_a, 'inputs_b':inputs_b},
                            y = {'class_img_outputs': class_img_outputs, 
                                'class_summed_outputs':class_summed_outputs}, 
                            batch_size = batch_size, verbose=0)

    # Evaluation (Test) Results
    print('\033[1m'+f"Test Accuracy (Image Labels)               : {scores[3]*100:.4f} %")
    print(f"Test Accuracy (Summed Output Labels)       : {scores[4]*100:.4f} %")
    print(f"Overall Test Loss                          : {scores[0]:.4f}", '\033[0m')
    print(f"Test Loss (Image Labels)                   : {scores[1]:.4f}")
    print(f"Test Loss (Summed Output Labels)           : {scores[2]:.4f}")


  @staticmethod
  def confusion_mtx_and_class_report(test_gen, predictions, 
                                      NUM_OF_CLASSES, CLASSIFICATION_TYPE):  # Corresponding int labels
    '''
    This function takes the following inputs and returns a plot of confusion matrix
    and classification report

    (Inputs):
    test_gen: True test labels (Corresponding true int labels)
    predictions: Model generated predictions on test data (Corresponding predicted int labels)
    NUM_OF_CLASSES: number of class labels
    
    (Outputs): Confusion Matrix and Classification Report with metric scores

    '''
    print("\033[1m")
    print(f"========= CONFUSION MATRIX AND CLASSIFICATION REPORT - ({CLASSIFICATION_TYPE}) TEST SET ==========");print()
    from sklearn.metrics import confusion_matrix, classification_report
    # Confusion Matrix
    cm = confusion_matrix(test_gen, predictions)   

    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False, annot_kws={"size": 13})

    # Class indices (actual labels)
    test_generator_class_indices = {num:str(num) for num in range(0,NUM_OF_CLASSES,1)}
    # 'class_indices' returns the corresponding class labels
    plt.xticks(ticks=np.arange(NUM_OF_CLASSES) + 0.5, labels=test_generator_class_indices, fontsize=11)
    plt.yticks(ticks=np.arange(NUM_OF_CLASSES) + 0.5, labels=test_generator_class_indices, rotation=90, fontsize=11)
    plt.xlabel("Predicted",fontsize=14)
    plt.ylabel("Actual",fontsize=14)
    plt.title("Confusion Matrix",fontsize=14)
    plt.show(); print()
    # Classification Report
    print("*"*15+"   Classification Report   "+"*"*15)
    target_list = [str(i) for i in list(test_generator_class_indices.keys())]
    clr = classification_report(test_gen, predictions, target_names=target_list, digits=4) 
    print(clr)


  @staticmethod
  def preprocess_test_input_img(test_image_path):
    '''
    This function takes the path to the input test image
    and returns a preprocessed image (which can be used as a input to the model)
    (Input): Single test image path
    (Output): Preprocessed image 
    '''

    # Read the original test image
    orig_sample_test_img = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)

    # Convert image to gray scale
    gray_sample_test_img = cv2.cvtColor(orig_sample_test_img, cv2.COLOR_RGB2GRAY)

    # Resizing image to desired input size
    gray_resized_test_img = cv2.resize(gray_sample_test_img, (28, 28),
                        interpolation = cv2.INTER_AREA)   # To shrink an image

    # Remove blemishes from image (if any)
    (thresh, black_n_white_sample_img) = cv2.threshold(gray_resized_test_img, 127,255, cv2.THRESH_BINARY_INV)

    # Display Images	: Plot Sample Input and Preprocessed Test Image

    f = plt.figure(figsize=(10,5))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)

    ax1.imshow(np.squeeze(orig_sample_test_img), cmap='gray')
    ax1.set_title("Original Test Input Image", pad=15, fontsize=13, fontweight='bold')
    ax2.imshow(np.squeeze(black_n_white_sample_img), cmap='gray')
    ax2.set_title("Preprocessed Test Input Image", pad=15, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return orig_sample_test_img, black_n_white_sample_img


  @staticmethod
  def predict_mnist_img_n_randnum_summed_op(model, test_img_path, randnum):
    ''' A function that takes models inputs and returns predicted outputs.
    1. Inputs - 
      i. Input a: Sample Test Image
      ii. Input b: Any Random Number
    2. Output: Displays both original and preprocessed image and returns model
      predicted results
    '''

    # Image: input_a
    # Preprocessing test input image
    orig_test_img, preprocessed_test_img = eval_func.preprocess_test_input_img(test_img_path)

    # Reshaping input_a
    sample_img_input_a_dims = np.expand_dims(preprocessed_test_img,0)

    # Random Number: input_b
    if randnum == None:
      randnum = np.random.randint(0,10)
    
    # Encode Random Number input
    enc_randnum_input_b = to_categorical(np.array([randnum]), num_classes = 10)   # Random number from 0-9

    # Model Predictions
    predictions = model.predict({'inputs_a':sample_img_input_a_dims, 'inputs_b':enc_randnum_input_b}, verbose=0)

    # 1. Image Classification Prediction
    img_class_pred = np.argmax(predictions[0], axis=1)[0]

    # 2. Summed Output Classification Prediction
    summed_output_class_pred = np.argmax(predictions[1], axis=1)[0]

    # Display inputs and predictions
    print("\n"+'\033[1m'+"Random Number Input                     :", randnum)
    print("Sample Image Output (Predicted Label 1) :", img_class_pred)
    print("Summed Output (Predicted Label 2)       :", summed_output_class_pred, '\033[0m'+'\n')
    print('*'*100)

class model_training_results:

  @staticmethod
  def retrieve_best_model_training_scores(best_model_name):
    # Retrieving all the results obtained after training and testing the model
    training_results = best_model_name.rstrip(".hdf5").split('-')[2:]

    # Training Results
    print('\033[1m'+f"Training Accuracy (Image Labels)           : {100*float(training_results[3]):.4f} %")
    print(f"Training Accuracy (Summed Output Labels)   : {100*float(training_results[4]):.4f} %")
    print(f"Overall Training Loss                      : {float(training_results[0]):.4f}", '\033[0m')
    print(f"Training Loss (Image Labels)               : {float(training_results[1]):.4f}")
    print(f"Training Loss (Summed Output Labels)       : {float(training_results[2])}");print()

    # Validation Results
    print('\033[1m'+f"Validation Accuracy (Image Labels)         : {100*float(training_results[8]):.4f} %")
    print(f"Validation Accuracy (Summed Output Labels) : {100*float(training_results[9]):.4f} %")
    print(f"Overall Validation Loss                    : {float(training_results[5]):.4f}", '\033[0m')
    print(f"Validation Loss (Image Labels)             : {float(training_results[6]):.4f}")
    print(f"Validation Loss (Summed Output Labels)     : {float(training_results[7])}");print()
