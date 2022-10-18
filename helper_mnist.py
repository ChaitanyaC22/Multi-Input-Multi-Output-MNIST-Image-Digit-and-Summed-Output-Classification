# -*- coding: utf-8 -*-
# Custom Module 1: 'helper_mnist.py'
'''
Contains all the necessary helper functions or methods that are needed
 to plot graphs, vizualize images and optimize tensorflow sessions
'''

# Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from collections import Counter
import cv2
from google.colab.patches import cv2_imshow

# For Reproducible Results: reset seeds and reset global tensorflow sessions
# Set seed_value
seed_value = 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)

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


# Class for helper functions
class magic_helper:

    @staticmethod
    def freq_plot_image_unique_num(train_imgs_labels):
        '''
        Plots a frequency plot of unique image numbers in the
        train dataset.
        (train dataset: The MNIST image dataset before being split
        into training and validation sets)
        '''
        fig, ax = plt.subplots(figsize=(12, 8))

        num_range = np.arange(10)
        idx_count_pairs = sorted(Counter(train_imgs_labels).items())
        num_indices = [i for (i, j) in idx_count_pairs]
        val_counts = [j for (i, j) in idx_count_pairs]
        plt.title("Analyzing frequency of numbers in the MNIST image train dataset",
                  fontsize=15, pad=12, fontweight='bold')
        plt.xticks(num_range)
        plt.xlabel("Unique numbers in the train dataset", fontsize=13,
                   fontstyle='italic')
        plt.ylabel("Frequency of unique numbers (train dataset)", fontsize=13,
                   fontstyle='italic')
        # Custom colors palette
        my_palette = ["#0000FF", "#800080", "#FFFF00", "#FF00FF", "#FFC0CB",
                      "#808080", "#FFA500", "#FFA500", "#008000", "#808000"]
        # Set your custom color palette
        customPalette = sns.set_palette(sns.color_palette(my_palette))
        # Plot barplot
        ax = sns.barplot(x=num_indices, y=val_counts, palette=customPalette)

        # Display values on bars
        for p in ax.patches:
            ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., \
                                                        p.get_height()), ha='center', \
                        va='center', xytext=(0, 10), textcoords='offset points', \
                        fontsize=13)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_rand_training_imgs_20_samples(training_imgs,
                                           training_imgs_labels,
                                           my_color='gray'):
        '''
        Plots 20 random unique samples of training images for visualization
        '''
        fig = plt.figure(figsize=(15, 15))
        print('\033[1m' + "Random instances of MNIST images from training dataset:" +
              '\033[0m');
        print()
        for idx in range(20):
            plt.subplot(5, 4, idx + 1)
            plt.imshow(training_imgs[idx], cmap=my_color)
            plt.title("True Label {}".format(training_imgs_labels[idx]))

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_scaling_effect_training_img(training_imgs, training_imgs_labels, sample_idx=0):
        '''
        This function take the training image data and its labels as input
        and compares the effect of normalization on a sample training image.
        It returns a visual display of actual and normalized image
        '''
        # Original image
        og_img = training_imgs[sample_idx]  # Load a sample training image
        og_img_label = training_imgs_labels[sample_idx]

        # Normalized img
        normalized_img = training_imgs[sample_idx] / 255.0

        # Comparing original and normalized MNIST image
        print("\033[1m" + "\nComparing original and normalized MNIST image sample:\n" + "\033[0m")
        f, axes = plt.subplots(1, 2, sharey=True)
        f.set_figwidth(12)

        n_line = '\n'
        axes[0].imshow(og_img, cmap='gray')
        axes[0].set_title(f"Actual Image {n_line}True Label: {og_img_label}", pad=10, fontsize=12)

        axes[1].imshow(normalized_img, cmap='gray')
        axes[1].set_title(f"Normalized Image {n_line}True Label: {og_img_label}", pad=10, fontsize=12)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def optimize_session():
        '''
        Function for optimizing session, memory storage and tensorflow session
        related components
        '''
        # Optimize memory usage
        tf.keras.backend.clear_session()
        # Reset default graph
        tf.compat.v1.reset_default_graph()

    @staticmethod
    def plot_model_training_performance(history):
        '''
        (input) history -> trained model
        (outputs) -> model training performance plots
        The function takes the trained model in 'history' as input and
        returns multiple model training performances plots namely
        1. Accuracy vs Epoch (Image Classification)
        2. Accuracy vs Epoch (Summed Output Classification)
        3. Loss vs Epoch (Overall)
        4. Loss vs Epoch (Image Classification)
        5. Loss vs Epoch (Summed Output Classification)
        '''

        print("\n" + "\033[1m" + "*" * 100);
        print(" " * 38 + f"Model Training Plots");
        print("*" * 100 + '\033[0m');
        print()
        # summarize history for  Image Classification Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['class_img_outputs_accuracy'])
        plt.plot(history.history['val_class_img_outputs_accuracy'])
        plt.title('Accuracy vs Epoch (Image Classification)', fontsize=16)
        plt.ylabel('Accuracy (Image Classification)', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylim(bottom=0.7)  # lower max_lim (based on generic evaluation)
        plt.ylim(top=1.05)  # max accuracy limit
        x_ticks_acc = [_ for _ in range(0, len(history.history['class_img_outputs_accuracy']))]
        x_ticks_acc_labels = [_ for _ in range(1, len(history.history['class_img_outputs_accuracy']) + 1)]
        plt.xticks(x_ticks_acc, x_ticks_acc_labels, fontsize=10)
        plt.legend(['Training Accuracy (Image Classification)', 'Validation Accuracy (Image Classification)'],
                   fontsize=12, loc='lower right')
        plt.show();
        print()

        # summarize history for  Summed Output Classification Accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['class_summed_outputs_accuracy'])
        plt.plot(history.history['val_class_summed_outputs_accuracy'])
        plt.title('Accuracy vs Epoch (Summed Output Classification)', fontsize=16)
        plt.ylabel('Accuracy (Summed Output Classification)', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylim(bottom=0.7)  # lower max_lim (based on generic evaluation)
        plt.ylim(top=1.05)  # max accuracy limit
        x_ticks_acc = [_ for _ in range(0, len(history.history['class_summed_outputs_accuracy']))]
        x_ticks_acc_labels = [_ for _ in range(1, len(history.history['class_summed_outputs_accuracy']) + 1)]
        plt.xticks(x_ticks_acc, x_ticks_acc_labels, fontsize=10)
        plt.legend(
            ['Training Accuracy (Summed Output Classification)', 'Validation Accuracy (Summed Output Classification)'],
            fontsize=12, loc='lower right')
        plt.show();
        print("\n\n")
        print('\033[1m' + "*" * 100 + '\033[0m');
        print()

        # Summarize history for Overall Loss
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss vs Epoch (Overall)', fontsize=16)
        plt.ylabel('Overall Loss', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylim(bottom=0.0)  # lower max_lim (based on generic evaluation)
        plt.ylim(top=0.5)  # max limit for loss (based on generic evaluation)
        x_ticks_loss = [_ for _ in range(0, len(history.history['loss']))]
        x_ticks_loss_labels = [_ for _ in range(1, len(history.history['loss']) + 1)]
        plt.xticks(x_ticks_loss, x_ticks_loss_labels, fontsize=10)
        plt.legend(['Training Loss - Overall', 'Validation Loss - Overall'], fontsize=12, loc='upper right')
        plt.show()

        # Summarize history for Image Classification Loss
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['class_img_outputs_loss'])
        plt.plot(history.history['val_class_img_outputs_loss'])
        plt.title('Loss vs Epoch (Image Classification)', fontsize=16)
        plt.ylabel('Loss (Image Classification)', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylim(bottom=0.0)  # lower max_lim (based on generic evaluation)
        plt.ylim(top=0.5)  # max limit for loss (based on generic evaluation)
        x_ticks_loss = [_ for _ in range(0, len(history.history['class_img_outputs_loss']))]
        x_ticks_loss_labels = [_ for _ in range(1, len(history.history['class_img_outputs_loss']) + 1)]
        plt.xticks(x_ticks_loss, x_ticks_loss_labels, fontsize=10)
        plt.legend(['Training Loss (Image Classification)', 'Validation Loss (Image Classification)'], fontsize=12,
                   loc='upper right')
        plt.show()

        # Summarize history for Summed Output Classification loss
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['class_summed_outputs_loss'])
        plt.plot(history.history['val_class_summed_outputs_loss'])
        plt.title('Loss vs Epoch (Summed Output Classification)', fontsize=16)
        plt.ylabel('Loss (Summed Output Classification)', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylim(bottom=0.0)  # lower max_lim (based on generic evaluation)
        plt.ylim(top=0.5)  # max limit for loss (based on generic evaluation)
        x_ticks_loss = [_ for _ in range(0, len(history.history['class_summed_outputs_loss']))]
        x_ticks_loss_labels = [_ for _ in range(1, len(history.history['class_summed_outputs_loss']) + 1)]
        plt.xticks(x_ticks_loss, x_ticks_loss_labels, fontsize=10)
        plt.legend(['Training Loss (Summed Output Classification)', 'Validation Loss (Summed Output Classification)'],
                   fontsize=12, loc='upper right')
        plt.show()

        print();
        print('\033[1m' + "*" * 100 + '\033[0m')

    @staticmethod
    def visualize_ten_misclassified_samples(test_image_generator_x,
                                            misclass_imgs_indices,
                                            misclass_imgs_true,
                                            misclass_imgs_pred):
        '''
        This function takes the following inputs:
        (Inputs):
        test_image_generator_x: image data via image data generator
        misclass_imgs_indices: indices of the first ten misclassified test images
        misclass_imgs_true: true labels of the first ten misclassified test images
        misclass_imgs_pred: predicted labels of the first ten misclassified test images
        and returns (Output): visual display of those ten misclassifed test images
        '''
        print('\033[1m' +"Misclassified MNIST Digit Image Samples from Test Dataset:"+ '\033[0m');print()
        fig = plt.figure(figsize=(14, 8))
        for idx in range(10):
          plt.subplot(2, 5, idx + 1)
          plt.imshow(np.squeeze(test_image_generator_x[misclass_imgs_indices[idx]]), cmap='gray')
          plt.title(f"True Label: {misclass_imgs_true[idx]}\nPredicted Label: {misclass_imgs_pred[idx]}",
                                                                         pad=20)
          # turn off the axes
          plt.axis('off')
        plt.tight_layout()
        plt.show()
