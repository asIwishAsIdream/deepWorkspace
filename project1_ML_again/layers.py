import numpy as np
from functions import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None
        self.mask = None
        self.dropout_ration = 0.5

    def forward(self, x, train_flg=False):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        if train_flg:
            # Dropout을 고려하여 forward 수행
            self.mask = np.random.rand(*out.shape) > self.dropout_ration
            out = out * self.mask

        return out

    def backward(self, dout):
        if self.mask is not None:
            dout = dout * self.mask

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 손실함수
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블(원-핫 인코딩 형태)

    def forward(self, x, t):
        self.t = t
        self.t = self.t.astype(int)
        self.y = softmax(x)
        # self.loss = cross_entropy_error(self.y, self.t)
        self.loss = negative_log_likelihood_loss(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            # np.arange(batch_size)의 형상은 (100,)이고 self.t의 형상이 (100, 10), 이 문제를 해결하려면 self.t를 1차원 배열로 변환해야 합
            # 이 코드를 추가하면 self.t가 2차원 배열일 때, 즉 원-핫 인코딩된 레이블 배열일 때 이를 1차원 배열로 변환합니다
            if self.t.ndim != 1:
                self.t = np.argmax(self.t, axis=1)
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx