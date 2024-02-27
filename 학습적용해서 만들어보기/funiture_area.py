import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

# 미리 학습된 VGG16 모델 로드
model = VGG16(weights='imagenet')

# 이미지 불러오기 및 전처리
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 방사진에서 가구 예측 및 표시
def predict_and_display(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=7)[0]

    # OpenCV를 사용하여 이미지 읽기
    img = cv2.imread(img_path)

    # 예측 결과에 대한 바운딩 박스 표시
    for i, (_, label, _) in enumerate(decoded_predictions):
        x, y, w, h = 10, 40 + i * 30, 200, 20  # 바운딩 박스 좌표 및 크기 설정 (크기를 작게 조절)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 바운딩 박스 그리기
        cv2.putText(img, label, (x, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # 레이블 표시

    # 이미지를 화면에 표시
    cv2.imshow("Predictions", img)
    cv2.waitKey(0)

# 방사진에서 가구 예측 및 표시
img_path = "./testcolor/test_15.jpg"
predict_and_display(img_path)
