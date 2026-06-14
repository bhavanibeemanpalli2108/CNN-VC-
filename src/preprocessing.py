import cv2
import numpy as np


def resize_image(image_path, size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    return image


def normalize(image):
    return image / 255.0