import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score
from sklearn.utils import shuffle

regularization_strength = 10000
learning_rate = 0.000001

def svm(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01
    for epoch in range(1, max_epochs):
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

        if epoch == 2 ** nth or epoch == max_epochs - 1:
            N = features.shape[0]
            distances = 1 - outputs * (np.dot(features, weights))
            distances[distances < 0] = 0  # equivalent to max(0, distance)
            hinge_loss = regularization_strength * (np.sum(distances) / N)
            cost = 1 / 2 * np.dot(weights, weights) + hinge_loss

            print("Epoch is: {} and Cost is: {}".format(epoch, cost))#这个地方要改——————————————————————————————————————————————————————————————————————————————————————————————————————————————————！

            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights


def init():
    print("reading dataset...")
    # read data in pandas (pd) data frame
    data = pd.read_csv('data.csv')

    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    print("applying feature engineering...")
    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    data['diagnosis'] = data['diagnosis'].map(diag_map)

    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]

    # # filter features
    # remove_correlated_features(X)
    # remove_less_significant_features(X, Y)

    # normalize data for better convergence and to prevent overflow
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = svm(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

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
    print("recall on test dataset: {}".format(recall_score(y_test, y_test_predicted)))
    print("precision on test dataset: {}".format(recall_score(y_test, y_test_predicted)))


# set hyper-parameters and call init

init()