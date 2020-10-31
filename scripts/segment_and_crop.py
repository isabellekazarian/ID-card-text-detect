import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.optimizers import Adam
from model.iou_loss import IoU
from imutils.object_detection import non_max_suppression

from utils import image
from utils import metrics



INPUT_FILE = './images/CA01_11.jpg'
OUTPUT_FILE = "./images/id2.jpg"
MODEL_FILE = "unet_model_whole_100epochs.h5"


def load_input_image():
    print('Loading image...')
    img = cv2.imread(INPUT_FILE)
    height,width = img.shape[:2]
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img.reshape(1,256,256,3)
    return img, height, width

def load_model_file():
    print('Loading model...')
    model = load_model(MODEL_FILE, compile=False)
    model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])
    return model

def predict_image(model, img):
    print('Predicting model...')
    predict = model.predict(img.reshape(1,256,256,3))
    return predict[0]



def main():

    ###
    #filename = input("Enter file name:  ")
    # INPUT_FILE = './dataset/' + filename + '.jpg'
    ###

    model = load_model_file()
    img, height, width = load_input_image()

    print('Segmenting...')
    segmented_image = predict_image(model,img)
    segmented_image = cv2.resize(segmented_image, (width, height))

    print('Cropping...')
    segmented_image = cv2.convertScaleAbs(segmented_image, alpha = 255)
    warped = image.convert_object(segmented_image, cv2.imread(INPUT_FILE))

    print('Saving output file...', OUTPUT_FILE)
    plt.imsave(OUTPUT_FILE, warped)

    print('Done.')



if __name__ == '__main__':
    main()