# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 14:21:31 2023

@author: sungd
"""

# 5.2.1 계산 그래프의 역전파
"""
y = f(x)
x       → f → y
E*∂y/∂x ← f ← E
"""


# 5.2.2 연쇄법칙이란?
"""
합성 함수 : 여러 함수로 구성된 함수
z = (x + y)²
또는
z = t²
t = x + y

연쇄법칙 : 합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.
∂z/∂x = ∂z/∂t*∂t/∂x
∂z/∂t = 2t
∂t/∂x = 1
따라서
∂z/∂x = ∂z/∂t*∂t/∂x = 2t*1 = 2(x+y)
"""

# 5.2.3 연쇄법칙과 계산 그래프
"""
x ↘   t
    + → **2 → z
y ↗

∂z/∂z*∂z/∂t*∂t/∂x
  ↖    ∂z/∂z*∂z/∂t
    +      ←      **2 ← ∂z/∂z
y ↗
"""

# 5.3 역전파
# 5.3.1 덧셈 노드의 역전파
"""
z = x + y
∂z/∂x = 1
∂z/∂y = 1
따라서 상류에서 온 값(∂L/∂z)을 그대로 하류로 전달한다.
"""

# 5.3.2 곱셈 노드의 역전파
"""
z = xy
∂z/∂x = y
∂z/∂y = x
따라서 상류에서 온 값에 순전파 때의 입력 신호를 서로 바꾼 값을 곱해 하류로 전달한다.
순방향 입력 신호의 값을 저장해 둘 필요가 있다.
"""