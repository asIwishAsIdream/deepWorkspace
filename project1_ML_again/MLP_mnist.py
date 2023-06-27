import sys
import os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from MLP import *
import pickle
import matplotlib.pyplot as plt

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

sample=x_train[0]
plt.figure()
plt.imshow(sample.reshape(28,28), cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.show()

hidden_size_list = [100, 100, 100]

net_MLP = MLP(input_size=784, hidden_size_list=hidden_size_list, output_size=10)

# 하이퍼 파라메터
iters_num = 100000  # 반복횟수
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # print(i)
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 오차역전파법으로 기울기 계산
    grad = net_MLP.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for j in range(1, len(hidden_size_list)+1):
        net_MLP.params['W' + str(j)] -= learning_rate * grad['W' + str(j)]
        net_MLP.params['b' + str(j)] -= learning_rate * grad['b' + str(j)]

        # 학습 경과 기록
    loss = net_MLP.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭 당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = net_MLP.accuracy(x_train, t_train)
        test_acc = net_MLP.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

probability = softmax(net_MLP.predict(sample.reshape(1, 784)))
plt.subplot()
plt.bar(range(len(probability[0])), probability[0])
plt.ylim(0, 1.0)

plt.show()


with open('MLP_Prams.pkl', 'wb') as f:
    pickle.dump(net_MLP, f)
