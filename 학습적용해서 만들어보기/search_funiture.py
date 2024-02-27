# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 19:28:01 2024

@author: sungd
"""

import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np

# 미리 학습된 VGG16 모델 로드
model = VGG16(weights='imagenet')

# 이미지 불러오기 및 전처리
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 무드(가구) 예측
def predict_mood(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=7)[0]

    # 상위 3개 예측 결과 출력
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {'search furniture >>> '} {label} ({score:.2f})")

    # 가장 높은 확률의 감정(label) 반환
    return decoded_predictions[0][1]

# 방사진에서 가구 예측
img_path = "./testcolor/test_21.jpg"
predicted_mood = predict_mood(img_path)
#print(f'Predicted Funiture: {predicted_mood}')