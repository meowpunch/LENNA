import numpy as np
import torch

def set_data(filename): # return type : ndarray
    f = open(filename, 'r')
    data = np.genfromtxt(f, delimiter=' ')
    # print("--------------raw data--------------\n",data) # ',' between params and inference time : saved as nan
    X_data = data[:, 0:28].reshape(data.shape[0], 7, 4) # should change index according to file
    Y_data = data[:, 29]
    # print("--------------X data--------------\n", X_data)
    # print("--------------Y data--------------\n", Y_data)

    X_data, Y_data = preprocess(X_data, Y_data) #standarize + normalize
    Y_data = torch.from_numpy(Y_data).view(data.shape[0],1)

    # separate train data and test data & convert to tensor
    train_ratio = 0.8
    ratio = int(data.shape[0] * train_ratio)
    X_train = torch.from_numpy(X_data[:ratio,:])
    X_test = torch.from_numpy(X_data[ratio:,:])
    Y_train = Y_data[:ratio,]
    Y_test = Y_data[ratio:,]

    # print("----------tensor ver.---------------")
    # print("----> X_train\n{}\n---->Y_train\n{}\n---->X_test\n{}\n---->Y_test\n{}".format(X_train, Y_train, X_test, Y_test))

    # print("---> size")
    # print(X_train.size(), X_test.size(), Y_train.size(), Y_test.size())

    return X_train, Y_train, X_test, Y_test

def preprocess(X_data, Y_data): # return type : ndarray
    # normalize : subtract min from elements and divide by (max - min)
    # print("--------------preprocess------------------------")
    for i in range(X_data.shape[2]):
        min = X_data[:, :, i].min() # min, max of column vector
        max = X_data[:, :, i].max()
        # print("min : {}, max : {}".format(min,max))
        for j in range(X_data[:, : ,i].shape[0]):
            for k in range(X_data[:, :, i].shape[1]):
                e = X_data[:, :, i][j][k]
                # print("e: ", e)
                X_data[:, :, i][j][k] = (e - min) / max
    # print("----------normalize--------------\n", "X_Data: ",X_data)

    # standarize
    for i in range(X_data.shape[2]):
        std = X_data[:, :, i].std()
        mean = X_data[:, :, i].mean()
        for j in range(X_data[:, : ,i].shape[0]):
            for k in range(X_data[:, :, i].shape[1]):
                e = X_data[:, :, i][j][k]
                # print("e: ", e)
                X_data[:, :, i][j][k] = (e - mean) / std

    # print("----------standarize--------------\n", "X_Data: ",X_data)

    # for Y
    for i in range(Y_data.shape[0]):
        min = Y_data.min()
        max = Y_data.max()
        e = Y_data[i]
        Y_data[i] = (e - min) / max

    for i in range(Y_data.shape[0]):
        std = Y_data.std()
        mean = Y_data.mean()
        e = Y_data[i]
        Y_data[i] = (e - mean) / std

    return X_data, Y_data
