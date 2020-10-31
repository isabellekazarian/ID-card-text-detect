import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

from model.iou_loss import IoU

from utils import image
from utils import metrics

#INPUT_FILE = "./dataset/png/KS01_30.png"
OUTPUT_MASK = "./dataset/png/output_mask.png"
#OUTPUT_FILE = "./dataset/png/output_pred.png"
MODEL_FILE = "unet_model_whole_100epochs.h5"


def load_image(file):
    img = cv2.imread(file)
    height,width = img.shape[:2]
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img.reshape(1,256,256,3)
    return img, height, width


def predict_image(model, image):
    predict = model.predict(image.reshape(1,256,256,3))
    return predict[0]


def main():

    filename = input("Enter file name:  ")
    INPUT_FILE = "./dataset/png/" + filename + ".png"
    OUTPUT_FILE = "./dataset/png/" + filename + "_output.png"

    if not os.path.isfile(INPUT_FILE):
        print('Input image not found ', INPUT_FILE)

    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)

        else:
            print('Load model... ', MODEL_FILE)
            model = load_model(MODEL_FILE, compile=False)
            model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

            print('Load image... ', INPUT_FILE)
            img, h, w = load_image(INPUT_FILE)

            print('Prediction...')
            output_image = predict_image(model, img)

            print('Cut it out...')
            mask_image = cv2.resize(output_image, (w, h))
            mask_image = cv2.convertScaleAbs(mask_image, alpha = 255)
            warped = image.convert_object(mask_image, cv2.imread(INPUT_FILE))

            print('Save output files...', OUTPUT_FILE)
            plt.imsave(OUTPUT_MASK, mask_image, cmap='gray')
            plt.imsave(OUTPUT_FILE, warped)

            print('Done.')


if __name__ == '__main__':
    main()
