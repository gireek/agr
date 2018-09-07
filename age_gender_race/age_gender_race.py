import cv2
import os
import numpy as np
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
import tensorflow as tf
from random import randint
import json
from time import sleep


class AgeGenderRace:
    TRAINING_WEIGHTS = os.path.join('pretrained_models', 'weights.18-4.06.hdf5')
    CASE_PATH = os.path.join('pretrained_models', 'haarcascade_frontalface_alt.xml')
    WRN_WEIGHTS_PATH = os.path.join('pretrained_models', 'weights.18-4.06.hdf5')
    CSV_PATH = os.path.join('dataset', 'age_gender_race.csv')

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64, detect_size=25, train = False):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AgeDetection, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64, detect_size=25, train = False):
        self.face_size = face_size
        self.detect_size = detect_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        self.image = None
        self.image_bounding_boxes = None
        self.face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
        self.save_image_path = 'Extracted'
        self.people_dict = {'total_people': [], 'people_under_age': [], 'people_of_age': []}
        if self.save_image_path not in os.listdir("."):
            os.mkdir(self.save_image_path)
        model_dir = os.path.join(os.getcwd(), "pretrained_models").replace("//", "\\")
        if not train:
            fpath = get_file('weights.18-4.06.hdf5',
                             self.WRN_WEIGHTS_PATH,
                             cache_subdir=model_dir)
            self.model.load_weights(fpath)
        else:
            fpath = get_file('weights.18-4.06.hdf5',
                             self.TRAINING_WEIGHTS,
                             cache_subdir=model_dir)
            self.model.load_weights(fpath)
        self.graph = tf.get_default_graph()