from functions import *
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import pandas as pd
import numpy as np

# 3 년간 주가 데이터, 모든 행 882행, 열 7
df = pd.read_csv('TSLA.csv')
# Date  Open   High	  Low   Close   Adj Close(장기적인 투자 분석에 유용)    Volume(거래량은 거래일 동안 거래된 주식의 총 수,거래량이 높을수록 가격 변동성이 높음)
# 이를 봤을 때 x 데이터에서 5, 6열이 주가를 예측할 때 중요한 역할을 할 거 같다

# 예측될 목표 값은 주가(Close)에 ‘가까운’  값이 될 것이다.
# close가 target이 된다.
# 예측을 진행할 X 데이터는 아래와 같다
X_stock = df[['AdjClose','Volume']]
y_stock = df[['Close']]

x_train, x_test, y_train, y_test = model_selection.train_test_split(X_stock, y_stock, test_size=0.3, random_state=0)


# x 데이터 Standardization
sc = MinMaxScaler(feature_range=(0,1))
sc.fit(x_train)

X_train_scaled = sc.transform(x_train)
X_test_scaled = sc.transform(x_test)
print(len(X_train_scaled))

# 예를 들어, 시퀀스 길이를 10으로 설정
sequence_length = 10
hidden_state = 5

# 입력 데이터는 (x의 열 데이터 수, 훈련 샘플 수, 시퀀스 길이)의 3차원 형태를 가지게 됩니다.
X_train = []
# Sliding Window 방식으로 ABC", "BCD", "CDE", "DEF"와 같이 겹치는 부분이 많은 연속적인 부분 문자열 생성
for i in range(len(X_train_scaled) - sequence_length + 1):
    X_train.append(X_train_scaled[i:i+sequence_length])

X_train = np.array(X_train) # (607, 10, 2) X_train.shape =>> 왜냐면 10개의 Sequence를 나누면 총 607개가 나옴 + 2개의 x데이터의 열
# 데이터의 차원 변경
X_train = np.transpose(X_train, (2, 0, 1)) # 차원의 재배열을 의미 ()안에는 index 번호를 의미한다 # (2, 607, 10)

# 타겟값 y는 (y의 열 데이터 수, 훈련 샘플 수, 시퀀스 길이)의 형태



# LSTM을 학습 시킨 결과 => 가중치W와 bias가 저장되어 있다
parameters = initialize_parameters(hidden_state, X_train.shape[1], y_train.shape[1])
da = (hidden_state, X_train.shape[1], y_train.shape[1])

iters_num = 10  # 반복횟수
for i in range(iters_num):
    # a = 다음으로 넘어갈 값, y는 예측값, c는 이전부터 저장된 값, caches는 backpropa에 사용될 값
    a, y, c, caches = lstm_forward(X_train, 0, parameters)
    print(y.shape)

















