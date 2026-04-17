import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2

CLASS_NAMES = np.array(['Cr', 'In', 'Pa', 'PS', 'Rs', 'Sc'])

IMG_HEIGHT = 32
IM_WIDTH = 32

model = load_model('model.h5')

src = cv2.imread('data/val/Pa/Pa_2.bmp')
src = cv2.resize(src,(32,32))
src = src.astype('int32')
src = src / 255

test_img = tf.expand_dims(src,0)

preds = model.predict(test_img)

score = preds[0]
# print(score)

print('模型预测的结果为{}， 概率为{}'.format(CLASS_NAMES[np.argmax(score)], np.max(score)))