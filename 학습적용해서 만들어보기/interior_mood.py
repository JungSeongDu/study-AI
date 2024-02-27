# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:32:00 2024

@author: sungd
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 데이터셋을 저장할 폴더 경로
dataset_folder = "mood"

# 클래스 레이블 정의
class_labels = sorted(os.listdir(dataset_folder))

# 데이터 수집 및 전처리
data = []
labels = []

for label_id, class_label in enumerate(class_labels):
    class_folder = os.path.join(dataset_folder, class_label)
    for file_name in os.listdir(class_folder):
        image_path = os.path.join(class_folder, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB
        image = cv2.resize(image, (100, 100))  # 이미지 크기 조정
        data.append(image)
        labels.append(label_id)

data = np.array(data) / 255.0  # 이미지 정규화
labels = to_categorical(labels)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 모델 정의
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(class_labels), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 새로운 이미지 분류 예측
new_image = cv2.imread("./testcolor/p1065582637908554_587_thum.jpg")
new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
new_image = cv2.resize(new_image, (100, 100))
new_image = np.expand_dims(new_image, axis=0)  # 배치 차원 추가
new_image = new_image / 255.0  # 이미지 정규화
prediction = model.predict(new_image)
predicted_class = class_labels[np.argmax(prediction)]
print("Predicted Atmosphere:", predicted_class)
