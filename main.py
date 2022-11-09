from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
import csv



print("__________________________________________________")
with open('kc_house_data.csv', 'r') as f:
    reader = csv.reader(f)
    price = np.array([row[2] for row in reader])
    price = np.delete(price, 0)
    price = price.astype(np.float64)
with open('kc_house_data.csv', 'r') as f:
    reader = csv.reader(f)
    space = np.array([row[5] for row in reader])
    space = np.delete(space, 0)
    space = space.astype(np.float64)

    space_train, space_test, price_train, price_test = train_test_split(space, price, test_size=0.2, random_state=0)

    print(space)
    print(price)
    #print(type (space))
print("__________________________________________________")


def partial_k(y_ture, y_guess, x):
    return -2 * np.mean((np.array(y_ture) - np.array(y_guess)) * np.array(x))


# 对b求
def partial_b(y_ture, y_guess):
    return -2 * np.mean((np.array(y_ture) - np.array(y_guess)))


# 求损失函数
def l2_loss(y_ture, y_guess):
    return np.mean((np.array(y_ture) - np.array(y_guess)) ** 2)


# y_hat的函数
def y_guess(k, x, b):
    return k * x + b


trying_time = 10000
min_loss = float('inf')
best_k, best_b = None, None
learning_rate = 0.1
k = random.randint(0, 1) #随机出两个值并附给k和b
b = random.randint(-1, 1)

min_max_scaler = preprocessing.MinMaxScaler()
space_train = min_max_scaler.fit_transform(space_train.reshape(-1, 1))
price_train = min_max_scaler.fit_transform(price_train.reshape(-1, 1))
space_test = min_max_scaler.fit_transform(space_test.reshape(-1, 1))
price_test = min_max_scaler.fit_transform(price_test.reshape(-1, 1))

plt.figure()
plt.scatter(space, price, color='red', alpha=0.5)

plt.show()

# space_train, space_test, price_train, price_test

for i in range(trying_time):

    yhat = y_guess(k, space_train, b)
    L2_loss = l2_loss(price_train, yhat)
    best_k = k
    best_b = b
    min_loss = L2_loss

    k = k - partial_k(price_train, yhat, space_train) * learning_rate
    b = b - partial_b(price_train, yhat) * learning_rate
print('L2loss=', min_loss)
print('y = {} * x {}'.format(best_k, best_b))


plt.scatter(space_train, price_train, color='red')
plt.plot(space_train, y_guess(best_k, space_train, best_b), color='blue')

plt.show()

plt.scatter(space_test, price_test, color='green')
plt.plot(space_test, y_guess(best_k, space_test, best_b), color='blue')

plt.show()

# space_train, space_test, price_train, price_test