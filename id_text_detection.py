import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

from model.iou_loss import IoU

from utils import image
from utils import metrics


INPUT_FILE = "./dataset/img-test/KS01_30.png"
OUTPUT_FILE = INPUT_FILE[:-4] + "_detection.png"
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

    # INPUT_FILE = "./dataset/img-test/" + input("\n\nEnter file name:  ") + ".png"
    # print(INPUT_FILE)
    # OUTPUT_FILE = INPUT_FILE[:-4] + "_detection.png"

    if not os.path.isfile(INPUT_FILE):
        print('Input image not found ', INPUT_FILE)

    else:
        if not os.path.isfile(MODEL_FILE):
            print('Model not found ', MODEL_FILE)
            exit(1)

        else:
            print('Loading model... ', MODEL_FILE)
            model = load_model(MODEL_FILE, compile=False)
            model.compile(optimizer=Adam(1e-4), loss=IoU, metrics=['binary_accuracy'])

            print('Loading image... ', INPUT_FILE)
            img, height, width = load_image(INPUT_FILE)

            print('Predicting...')
            mask_image = predict_image(model, img)

            print('Cropping...')
            mask_image = cv2.resize(mask_image, (width, height))
            mask_image = cv2.convertScaleAbs(mask_image, alpha = 255)
            cropped_img = image.convert_object(mask_image, cv2.imread(INPUT_FILE))


            cv2.imshow('Cropped', cropped_img)
            cv2.waitKey(0)

            # ------ resize -------
            NEW_SIZE = 1000
            print('Resizing image...')
            height, width = cropped_img.shape[:2]
            longest_side = width
            if (height > width): longest_side = height

            ratio = NEW_SIZE / longest_side
            resized_img = cv2.resize(cropped_img, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_CUBIC)

            cv2.imshow('Resized', resized_img)
            cv2.waitKey(0)

            # ----- create threshold -----
            print('Thresholding...')
            thresh = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(thresh, 170, 255, cv2.THRESH_BINARY_INV)
            thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 5)

            cv2.imshow('Binarized', thresh)
            cv2.waitKey(0)

            # ----- morphs -----
            print('Morphing...')
            closing_kernel = np.ones((4, 4), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, closing_kernel)

            dilate_kernel = np.ones((5, 8), np.uint8)
            dilation = cv2.dilate(opening, dilate_kernel, iterations=1)

            cv2.imshow('Dilate', dilation)
            cv2.waitKey(0)


            # ----- find contours -----
            print('Finding contours...')
            contours, hierarchy = cv2.findContours(dilation,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for c in contours:
                rect = cv2.boundingRect(c)
                x, y, width, height = rect
                cv2.rectangle(resized_img, (x, y), (x + width, y + height), (0, 255, 0), 2)

            cv2.imshow('Text detection', resized_img)
            cv2.waitKey(0)

            # ----- save -----
            print('Saving output file...', OUTPUT_FILE)
            plt.imsave(OUTPUT_FILE, resized_img)

            print('Done.')


if __name__ == '__main__':
    main()
