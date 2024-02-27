# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:52:36 2023

@author: sungd
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# 간단한 CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3개의 클래스에 대한 출력
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 데이터 전처리 및 학습
# 여기에서는 데이터가 없으므로 실제 데이터로 변경해야 합니다.
# 데이터는 색상에 따라 클래스가 분류되어야 합니다.

# 예제 이미지 로드 및 전처리
img_path = 'C:\00.kosmo137\photo\197633.png'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# 레이블(색상)에 따라 클래스 할당 (예: 빨간색: 0, 초록색: 1, 파란색: 2)
# 실제 데이터에 맞게 변경해야 합니다.
class_label = 0  # 빨간색 예시

# 모델 학습
model.fit(img_array, np.array([class_label]), epochs=10)
