# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:06:35 2023

@author: sungd
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:52:36 2023

@author: sungd
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import keras 
from keras.datasets import cifar10


# 데이터 준비
data_path = r"C:\00.kosmo137\photo"  # 실제 데이터셋 경로로 변경 필요

def load_data(data_path):
    images = []
    labels = []
    color_labels = os.listdir(data_path)

    for color_label in color_labels:
        color_path = os.path.join(data_path, color_label)
        for img_name in os.listdir(color_path):
            img_path = os.path.join(color_path, img_name)
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(color_label)

    return np.array(images), np.array(labels)

images, labels = load_data(data_path)

# 레이블 인코딩
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

# 간단한 CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(labels_encoded)), activation='softmax')  # 클래스 수에 대한 출력
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
