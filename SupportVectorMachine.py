import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

regularization_strength = 10000
learning_rate = 0.000001

data = pd.read_csv('data.csv')
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)  # remove the id and last null column
diag_map = {'M': 1.0, 'B': -1.0}  # move the data to number ready to use
data['diagnosis'] = data['diagnosis'].map(diag_map)  # split the features and final outputs
Y = data.loc[:, 'diagnosis']
X = data.iloc[:, 1:]
X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

X.insert(loc=len(X.columns), column='intercept', value=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


def svm(features, outputs):
    maxtraintime = 5000
    weights = np.zeros(features.shape[1])
    nth = 1
    prev_cost = float("inf")
    cost_threshold = 0.01
    for times in range(1, maxtraintime):
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            W, X_batch, Y_batch = weights, x, Y[ind]
            if type(Y_batch) == np.float64:
                Y_batch = np.array([Y_batch])
                X_batch = np.array([X_batch])

            distance = 1 - (Y_batch * np.dot(X_batch, W))
            dw = np.zeros(len(W))

            for ind, d in enumerate(distance):
                if max(0, d) == 0:
                    di = W
                else:
                    di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
                dw += di

            ascent = dw / len(Y_batch)
            weights = weights - (learning_rate * ascent)

        if times == 2 * nth or times == maxtraintime - 1:
            N = features.shape[0]
            distances = 1 - outputs * (np.dot(features, weights))
            distances[distances < 0] = 0
            hinge_loss = regularization_strength * (np.sum(distances) / N)
            cost = 1 / 2 * np.dot(weights, weights) + hinge_loss
            print("Train time is: ", times)
            print("Loss for this turn is: ",cost)
            if abs(prev_cost - cost) < 0.000000001:
                return weights
            prev_cost = cost
            nth += 1
    return weights


# train the model

W = svm(X_train.to_numpy(), y_train.to_numpy())

print("weights: ")
print(W)

# testing the model
print("testing the model...")
y_train_predicted = np.array([])
for i in range(X_train.shape[0]):
    yp = np.sign(np.dot(X_train.to_numpy()[i], W))
    y_train_predicted = np.append(y_train_predicted, yp)

y_test_predicted = np.array([])
for i in range(X_test.shape[0]):
    yp = np.sign(np.dot(X_test.to_numpy()[i], W))
    y_test_predicted = np.append(y_test_predicted, yp)

print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))

mx =confusion_matrix(y_test, y_test_predicted)
print(mx)


