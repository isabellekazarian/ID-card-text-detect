import cv2
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from model.iou_loss import IoU
import os
from keras.models import load_model

from utils import image
from utils import metrics

INPUT_FILE = "test.jpg"
OUTPUT_MASK = "output.jpg"
OUTPUT_FILE = "id.jpg"
MODEL_FILE = "unet_model_whole_100epochs.h5"


def load_image():
    img = cv2.imread(INPUT_FILE)
    height,width = img.shape[:2]
    img = cv2.resize(img, (256,256))
    img = img / 255.0
    img.reshape(1,256,256,3)
    return img, height, width


def predict_image(model, image):
    predict = model.predict(image)
    return predict[0]


def main():
            print('Cut it out...')
            output_image = cv2.imread(OUTPUT_MASK)
            #mask_image = cv2.resize(output_image, (w, h))
            warped = image.convert_object(output_image, cv2.imread(INPUT_FILE))

            print('Save output files...', OUTPUT_FILE)
            ##plt.imsave(OUTPUT_MASK, mask_image, cmap='gray')
            plt.imsave(OUTPUT_FILE, warped)

            print('Done.')


if __name__ == '__main__':
    main()