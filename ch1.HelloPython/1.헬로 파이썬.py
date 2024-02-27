# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 14:10:24 2023

@author: sungd
"""


1.1 파이썬이란?
간단하고 배우기 쉬운 프로그래밍 언어.

1.2 파이썬 설치하기
파이썬 버전
현재 파이썬은 2.x 버전과 3.x 버전이 공전한다. 둘은 하위 호환성이 없다. 여기에서는 파이썬 3를 사용한다.

사용하는 외부 라이브러리
numpy: 수치계산용 라이브러리. 행렬 연산 및 수학 알고리즘을 위해 사용.
matplotlib: 그래프를 그려주는 라이브러리
아나콘다 배포판
배포판이란 한번에 설치할 수 있도록 필요한 라이브러리 등을 하나로 정리해 둔 것이다. 그 중 아나콘다는 데이터 분석에 중점을 둔 파이썬 배포판이다. 상기한 넘파이와 matplotlib가 포함되어 있다.

https://www.continuum.io/downloads
파이썬 인터프리터
python --version
산술 연산
print(1 - 2)
print(4 * 5)
print(7 / 5)
print(3 ** 2)
-1
20
1.4
9
파이썬3에서 정수와 정수를 나눈 결과는 실수가 된다.

자료형
type()함수로 데이터의 자료형을 알아볼 수 있다.

print(type(10))
print(type(2.718))
print(type("hello"))
<class 'int'>
<class 'float'>
<class 'str'>
변수
파이썬은 동적 언어로 분류되며, 자동 형변환이 일어난다.

리스트
여러 데이터를 리스트로 정리할 수 있다. 원소에 접근할 때는 []를 이용한다. 슬라이싱을 이용해 부분 리스트를 얻을 수 있다.

a = [1, 2, 3, 4, 5]
print(a)
print(len(a))
print(a[0])
print(a[4])
a[4] = 99
print(a)
print(a[0:2])
print(a[1:])
print(a[:3])
print(a[:-1])
print(a[:-2])
[1, 2, 3, 4, 5]
5
1
5
[1, 2, 3, 4, 99]
[1, 2]
[2, 3, 4, 99]
[1, 2, 3]
[1, 2, 3, 4]
[1, 2, 3]
딕셔너리
키와 값을 한 쌍으롤 저장한다.

me = {'height': 100}
print(me['height'])
me['weight'] = 70
print(me)
100
{'height': 100, 'weight': 70}
bool
참True과 거짓False 두 값 중 하나를 취한다. and, or, not 연산자를 사용할 수 있다.

if 문
for 문
함수
1.4 파이썬 스크립트 파일
파일로 저장하기
클래스
클래스를 정의해 독자적인 자료형을 만들 수 있다. 클래스에는 그 클래스만의 전용 함수(메서드)와 속성을 정의할 수 있다. __init__()이라는 특별한 메서드는 클래스를 초기화하는 생성자로 쓰인다. 파이썬에서는 메서드의 첫 번째 인수로 자신의 인스턴스를 나타내는 self를 명시적으로 사용한다.

1.5 넘파이
딥러닝을 구현하는데는 배열, 행렬 연산이 많이 등장한다 넘파이의 배열 클래스인 numpy.array에는 이런 메서드가 많이 준비되어 있다.

넘파이 배열 생성하기
np.array() 메서드를 사용한다. 파이썬의 list를 인수로 받아 numpy.ndarray를 반환한다.

import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))
[ 1.  2.  3.]
<class 'numpy.ndarray'>
넘파이의 산술연산
배열 x와 y의 원소 수가 같다면 산술 연산은 각 원소에 대해서 행해진다(element-wise). 배열과 스칼라값의 조합의 연산은 스칼라값과의 계산이 배열의 원소별로 한 번씩 수행된다(브로드캐스트).

x = np.array([1, 2, 3])
y = np.array([2, 4, 6])

print(x+y)
print(x-y)
print(x*y)
print(x/y)

print(x / 2.0)
[3 6 9]
[-1 -2 -3]
[ 2  8 18]
[ 0.5  0.5  0.5]
[ 0.5  1.   1.5]
넘파이의 N차원 배열
1차원 배열 뿐 아니라 다차원 배열도 작성할 수 있다. 행렬의 형상은 shape로 원소의 자료형은 dtype으로 알 수 있다.

NOTE: 넘파이 배열은 N차원 배열을 작성할 수 있다. 수학에서는 1차원 배열은 벡터, 2차원 배열은 행렬이라고 부른다. 이를 일반화 한 것을 텐서라 한다.

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A+B)
print(A*B)
print(A*10)
[[1 2]
 [3 4]]
(2, 2)
int32
[[ 4  2]
 [ 3 10]]
[[ 3  0]
 [ 0 24]]
[[10 20]
 [30 40]]
브로드캐스트
넘파이에서는 형상이 다른 배열끼리도 계산할 수 있다.

원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0]) # 0행
print(X[0][1]) #(0, 1)

X = X.flatten() # X를 1차원 배열로 평탄화
print(X)
print(X[np.array([0, 2, 4])]) # 인덱스가 0, 2, 4인 원소 얻기

print(X > 15)
print(X[X>15])
[[51 55]
 [14 19]
 [ 0  4]]
[51 55]
55
[51 55 14 19  0  4]
[51 14  0]
[ True  True False  True False False]
[51 55 19]
인덱스를 배열로 지정해 한 번에 여러 원소에 접근할 수 있다. 넘파이 배열에 부등호 연산자를 사용한 결과는 bool 배열이다.

1.6 matplotlib
matplotlib는 그래프를 그려주는 데이터 시각화 라이브러리이다.

단순한 그래프 그리기
그래프를 그리려면 pyplot모듈을 이용한다.

import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1) # 0~6 0.1 간격으로 생성
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y)
plt.show()

pyplot의 기능
cos 함수를 추가로 그리고 제목, 축의 레이블 등의 기능을 사용해본다.

import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1) # 0~6 0.1 간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, label="cos", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

이미지 표시하기
pyplot에는 이미지를 표시하는 메서드인 imshow()도 있다. 이미지를 읽을 땐 matplotlib.image 모듈의 imread()를 사용한다.

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png')

plt.imshow(img)
plt.show()

1.7 정리
파이썬과 넘파이에 대한 기본을 중심으로 살펴보았다.