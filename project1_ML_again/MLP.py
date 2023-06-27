from layers import *
from collections import OrderedDict


class MLP:

    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list

        # 리스트이므로 벡터의 덧셈이 아니라 [784, 100,100,100, 10] 이라는 리스트가 생성됨
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        self.all_size_list_num = len(all_size_list)
        # 가중치 초기화
        self.params = {}
        for idx in range(1, self.all_size_list_num):  # hidden_layer에 맞춰서 가중치를 생성한다
            self.params['W' + str(idx)] = weight_init_std * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

        # 계층 생성 ========================
        self.layers = OrderedDict()
        for idx in range(1, self.all_size_list_num):
            # 이렇게 되면 Affine 레이어의 self.W와 self.b는 self.params에 저장된 W와 b와 동일한 넘파이 배열을 참조하게 됩니다.
            # 따라서 self.params의 W와 b를 업데이트하면, Affine 레이어의 self.W와 self.b도 같이 업데이트되게 됩니다.
            self.layers['Affine' + str(idx)] = Affine(self.params['W'+str(idx)], self.params['b'+str(idx)])
            self.layers['Relu' + str(idx)] = Relu()

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for idx in range(1, self.all_size_list_num):
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads

# 이때 어디서 W와 b 값이 갱신이 되는지 알아보기!
