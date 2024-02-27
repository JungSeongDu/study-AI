# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:19:52 2024

@author: sungd
"""

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#설명
#VGG-16은 16개 계층으로 구성된 컨벌루션 신경망입니다. 
#ImageNet 데이터베이스의 1백만 개가 넘는 영상에 대해 훈련된 신경망의 사전 훈련된 버전을 불러올 수 있습니다 
#사전 훈련된 신경망은 영상을 키보드, 마우스, 연필, 각종 동물 등 1,000가지 사물 범주로 분류할 수 있습니다.

# 미리 학습된 VGG16 모델 로드
model_vgg = VGG16(weights='imagenet')

# 데이터셋을 저장할 폴더 경로
dataset_folder = "mood_1"

# 클래스 레이블 정의
class_labels = sorted(os.listdir(dataset_folder))

# 데이터셋 불러오기 및 전처리
datagen = ImageDataGenerator(rescale=1./255)

# 이미지 저장을 위한 변수
image_number = 0

# 데이터셋을 훈련 및 검증용으로 나누기
train_generator = datagen.flow_from_directory(
    dataset_folder,
    target_size=(100, 100),
    batch_size=100,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_folder,
    target_size=(100, 100),
    batch_size=100,
    class_mode='categorical',
    subset='validation'
)

# CNN 모델 정의
model_cnn = Sequential([
    Conv2D(100, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(class_labels), activation='softmax')
])

model_cnn.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# 모델 학습
history_cnn = model_cnn.fit(train_generator, epochs=15, validation_data=val_generator)

# 분위기 및 물체 예측 함수 정의
def predict_mood(img_array, model):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=7)[0]

    # 상위 3개 예측 결과 출력
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        #print(f"{i + 1}: {label} ({score:.2f})")
        print(f"{i + 1}: {label}")

    # 가장 높은 확률의 레이블 반환
    return decoded_predictions[0][1]

# 카메라로 실시간 사진 촬영 및 분위기 및 물체 예측
cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera', frame)

            # 키보드 's'를 눌러서 이미지 저장
            if cv2.waitKey(1) & 0xFF == ord(' '):
                image_number += 1
                image_path = f"testcolor/captured_image_{image_number}.jpg"
                cv2.imwrite(image_path, frame)
                print(f"!!!!! Image saved as {image_path} !!!!!!")

                # 저장된 이미지를 읽어와 분석
                test_image = cv2.imread(image_path)
                test_image_cnn = cv2.resize(test_image, (100, 100))
                test_image_cnn = np.expand_dims(test_image_cnn, axis=0)
                test_image_cnn = test_image_cnn / 255.0

                # 분위기 예측
                prediction_cnn = model_cnn.predict(test_image_cnn)
                predicted_class_cnn = class_labels[np.argmax(prediction_cnn)]
                print("Predicted Mood >>>>>>> :", predicted_class_cnn, '!!!!!!!')

                # 물체 예측
                img_array_vgg = cv2.resize(test_image, (224, 224))
                img_array_vgg = np.expand_dims(img_array_vgg, axis=0)
                img_array_vgg = preprocess_input(img_array_vgg)
                predicted_object = predict_mood(img_array_vgg, model_vgg)
                #print(f'Predicted Object: {predicted_object}')

                # 이미지를 한 장만 찍고 종료
                break

        else:
            print('no frame!')
            break
else:
    print('no camera!')
cap.release()
cv2.destroyAllWindows()

