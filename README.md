# ID Card Crop & Text Detection
*This project is a modification of [this repository](https://github.com/Red-Eyed/ID-Card-Segmentation)*

This script takes an input image of an ID card, crops the card to size, and detects text on the card.


## Prerequisites
* Python 3.7
* Keras
* OpenCV
* numpy


## Getting Started

Open `id-text-detection.py`.
Check the paths of the two input files (namely `INPUT_FILE` and `MODEL_FILE`) and choose a location for `OUTPUT_FILE`. 
Note: the input and output files MUST be either png or jpg formats.

## Using the Script
When run, the script will run operations on the `INPUT_FILE` image.
No additional input from the user is necessary.
The final cropped image with text detection will be output to the `OUTPUT_FILE` location.

*Note: The script will run several operations on the input image.
For this script to work properly, the input image must be a photo of a standard size id card with no obstructions.*

## References
* [ID-Card-Segmentation by Red-Eyed](https://github.com/Red-Eyed/ID-Card-Segmentation)
* [OpenCV Text Detection by Adrian Rosebrock](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)
* [Text Skew Correction by Adrian Rosebrock](https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/)
