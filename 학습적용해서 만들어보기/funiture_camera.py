# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:12:45 2024

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
def predict_mood(img_array):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=7)[0]

    # 상위 3개 예측 결과 출력
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        print(f"{i + 1}: {'search furniture >>> '} {label} ({score:.2f})")

    # 가장 높은 확률의 가구(label) 반환
    return decoded_predictions[0][1]

# 카메라로 실시간 사진 촬영 및 예측
cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera', frame)
            if cv2.waitKey(1) != -1:
                # 실시간으로 찍은 사진을 모델에 입력
                img_array = cv2.resize(frame, (224, 224))
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                predicted_mood = predict_mood(img_array)
                #print(f'Predicted Furniture: {predicted_mood}')
                break
        else:
            print('no frame!')
            break
else:
    print('no camera!')
cap.release()
cv2.destroyAllWindows()
