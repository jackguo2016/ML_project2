from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np
import csv

k = random.randint(0, 1)
b = random.randint(-1, 1)
runningtime = 10000
learning_rate = 0.1
min_loss = float('inf')  # max the min-loss
best_k, best_b = None, None


def straightLineEquation(k, x, b):
    return k * x + b

def kDerivative(price, P_price, space):
    part1 = np.array(price) - np.array(P_price)
    part2 = np.mean(part1 * np.array(space))
    ans = -2 * part2
    return ans


def loss_function(price, P_price):
    part1 = np.array(price) - np.array(P_price)
    part2 = part1 ** 2
    ans = np.mean(part2)
    return ans


def bDerivative(price, P_price):
    part1 = np.array(price) - np.array(P_price)
    ans = -2 * np.mean(part1)
    return ans

print("_______________________________________________________________________________")
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
print("_______________________________________________________________________________")
print('-')
print('-')
print('-')

min_max_scaler = preprocessing.MinMaxScaler()
space_train = min_max_scaler.fit_transform(space_train.reshape(-1, 1))
price_train = min_max_scaler.fit_transform(price_train.reshape(-1, 1))
space_test = min_max_scaler.fit_transform(space_test.reshape(-1, 1))
price_test = min_max_scaler.fit_transform(price_test.reshape(-1, 1))

plt.figure()
plt.scatter(space, price, color='red', alpha=0.5)

plt.show()

# space_train, space_test, price_train, price_test

for i in range(runningtime):

    yhat = straightLineEquation(k, space_train, b)
    L2_loss = loss_function(price_train, yhat)
    best_k = k
    best_b = b
    min_loss = L2_loss

    k = k - kDerivative(price_train, yhat, space_train) * learning_rate
    b = b - bDerivative(price_train, yhat) * learning_rate
print('|||End!')
print('|||The final loss=', min_loss)
print('|||Final equation will be:')
print('|||y = {} * x {}'.format(best_k, best_b))
#hi

plt.scatter(space_train, price_train, color='red')
plt.plot(space_train, straightLineEquation(best_k, space_train, best_b), color='blue')

plt.show()

plt.scatter(space_test, price_test, color='green')
plt.plot(space_test, straightLineEquation(best_k, space_test, best_b), color='blue')

plt.show()

# space_train, space_test, price_train, price_test