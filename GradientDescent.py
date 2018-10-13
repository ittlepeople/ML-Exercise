import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time


# 逻辑回归梯度下降
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def model(X, theta):
    return sigmoid(np.dot(X, theta.T))


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))


# def gradient(X, y, theta):
#     grad = np.zeros(theta.shape)
#     error = (model(X, theta) - y).ravel()
#     for j in range(len(theta.ravel())):
#         term = np.multiply(error, X[:, j])
#         grad[0, j] = np.sum(term) / len(X)
#     return grad


# 洗牌
def shuffle_data(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, :cols-1]
    y = data[:, cols-1:]
    return X, y


def descent(data, theta, alpha, nums):
    # 梯度下降求解
    init_time = time.time()
    X, y = shuffle_data(data)
    grad = np.mat(np.zeros(theta.shape))  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值
    theta_n = int(theta.shape[1])
    for i in range(nums):
        error = (model(X, theta) - y).ravel()
        for j in range(theta_n):
            grad[0, j] = np.sum(np.multiply(error, X[:, j])) / len(X)
        theta = theta - alpha * grad
        costs.append(cost(X, y, theta))
    return theta, costs, time.time() - init_time


# 随机梯度下降
def randomgradient(data, theta, alpha, nums):
    init_time = time.time()
    grad = np.mat(np.zeros(theta.shape))  # 计算的梯度
    theta_n = int(theta.shape[1])
    m, n = shuffle_data(data)
    costs = [cost(m, n, theta)]  # 损失值
    for i in range(nums):
        X, y = shuffle_data(data)
        error = (model(X[0, :], theta) - y[0, :]).ravel()
        for j in range(theta_n):
            grad[0, j] = np.sum(np.multiply(error, X[0, j]))
        theta = theta - alpha * grad
        costs.append(cost(X, y, theta))
    return theta, costs, time.time() - init_time


# 小梯度下降
def minigradient(data, theta, alpha, nums):
    init_time = time.time()
    grad = np.mat(np.zeros(theta.shape))  # 计算的梯度
    theta_n = int(theta.shape[1])
    m, n = shuffle_data(data)
    costs = [cost(m, n, theta)]  # 损失值
    for i in range(nums):
        X, y = shuffle_data(data)
        # 选择多少条数据进行梯度下降
        error = (model(X[10, :], theta) - y[10, :]).ravel()
        for j in range(theta_n):
            grad[0, j] = np.sum(np.multiply(error, X[0, j])) / 10  # 除以多少条数据
        theta = theta - alpha * grad
        costs.append(cost(X, y, theta))
    return theta, costs, time.time() - init_time


def predict(X, theta):
    return [1 if p > 0.5 else 0 for p in model(X, theta)]


if __name__ == '__main__':
    path = 'data' + os.sep + 'LogiReg_data.txt'
    pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    # positive = pdData[pdData['Admitted'] == 1]
    # negative = pdData[pdData['Admitted'] == 0]
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.subplots()
    # ax.scatter(positive['Exam 1'], positive['Exam 2'], s=30, c='blue', marker='o', label='Admitted')
    # ax.scatter(negative['Exam 1'], negative['Exam 2'], s=30, c='red', marker='x', label='Not Admitted')
    # plt.legend()
    # plt.xlabel('Exam 1 Score')
    # plt.ylabel('Exam 2 Score')
    # plt.show()
    pdData.insert(0, 'Ones', 1)
    orig_data = np.mat(pdData)
    theta = np.mat(np.zeros([1, 3]))
    # 批量梯度下降
    # findtheta, costs, dur = descent(orig_data, theta, alpha=0.000001, nums=5000)
    # 随机梯度下降 注意学习率要低
    # findtheta, costs, dur = randomgradient(orig_data, theta, alpha=0.000001, nums=5000)
    # 小梯度下降
    findtheta, costs, dur = minigradient(orig_data, theta, alpha=0.000001, nums=5000)
    print(findtheta, dur)
    print(costs)
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(5001), costs, 'r')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()
    cols = orig_data.shape[1]  # 求有多少列
    # rows = orig_data.shape[0]  # 求有多少行
    # 设置X（训练数据）和y（目标变量）
    X = orig_data[:, :cols - 1]
    y = orig_data[:, cols - 1:]
    predictions = predict(X, findtheta)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    print('accuracy = {0}%'.format(accuracy))












