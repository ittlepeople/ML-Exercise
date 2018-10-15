import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import itertools


def printing_Kfold_scores(x_train_data, y_train_data):
    '''
    KFold(n_splits=’warn’, shuffle=False, random_state=None)  # 其返回的是个索引列
    K 折交叉验证：平均分成 K 份。由前到后开始，第一份为测试集，其余K-1 份为训练集. 训练集的索引为 0， 测试集索引为 1.
    '''
    fold = KFold(5, shuffle=False, random_state=None)  # 不洗牌
    print(fold)

    c_param_range = [0.01, 0.1, 1, 10, 100]
    results_table = pd.DataFrame(index=range(len(c_param_range)), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range

    # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
    # k-fold将给出2个列表：train_indices = indices [0]，test_indices = indices [1]
    j = 0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        recall_accs = []
        '''
        enumerate将一个可遍历对象（如列表、字符串）组成一个索引序列，
        将fold里的值迭代出来后增加一个索引。获得索引和元素值（fold对象的值，既训练集和测试集），start=1表示索引从1开始（默认为从0开始排序）
        '''
        k = 0
        for train_index, test_index in fold.split(x_train_data):
            # Call the logistic regression model with a certain C parameter
            lr = LogisticRegression(C=c_param, penalty='l1')
            '''
            # Use the training data to fit the model. In this case, we use the portion of the fold to train the model with indices[1].
            # 使用训练数据来拟合模型。在这种情况下，我们使用训练集来训练模型,索引[0]
            #  We then predict on the portion assigned as the 'test cross validation' with indices[0]
            # 然后我们使用索引[0]预测指定为“测试交叉验证”的部分
            '''
            lr.fit(x_train_data.iloc[train_index], y_train_data.iloc[train_index].values.ravel())

            # Predict values using the test indices in the training data
            y_pred_undersample = lr.predict(x_train_data.iloc[test_index].values)

            # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
            recall_acc = recall_score(y_train_data.iloc[test_index].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print('Iteration ', k, ': recall score = ', recall_acc)
            k = k + 1

        # The mean value of those recall scores is the metric we want to save and get hold of.
        # 最佳C参数
        # 千万注意results_table['Mean recall score']的类型是object，要转成float64！
        results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    # 最佳C参数
    # 千万注意results_table['Mean recall score']的类型是object，要转成float64！
    results_table['Mean recall score'] = results_table['Mean recall score'].astype('float64')
    # idxmax(): DataFrame.idxmax（axis = 0，skipna = True ）
    # 返回请求轴上第一次出现最大值的索引。NA / null值被排除在外。
    # best_c = results_table['C_Parameter'].iloc[results_table['Mean recall score'].idxmax()]
    best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

    # Finally, we can check which C parameter is the best amongst the chosen.
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')

    return best_c


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    # 显示图像，即2D常规栅格上的数据。
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # 刻度
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    path = 'data' + os.sep + 'creditcard.csv'
    data = pd.read_csv(path)
    data['normAmount'] = StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    print(data.head())
    X = data.iloc[:, data.columns != 'Class']
    y = data.iloc[:, data.columns == 'Class']

    # 异常数据量
    number_records_fraud = len(data[data.Class == 1])
    # 异常数据索引
    fraud_indices = np.array(data[data.Class == 1].index)
    # 正常样本索引
    normal_indices = np.array(data[data.Class == 0].index)

    # 向下采样-索引
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    # 样本融合 - 索引
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    # 向下采样数据
    under_sample_data = data.iloc[under_sample_indices, :]

    X_undersample = under_sample_data.iloc[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.iloc[:, under_sample_data.columns == 'Class']

    print("Percentage of normal transactions: ",
          len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
    print("Percentage of fraud transactions: ",
          len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
    print("Total number of transactions in resampled data: ", len(under_sample_data))

    # 所有数据集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print("Number transactions train dataset: ", len(X_train))
    print("Number transactions test dataset: ", len(X_test))
    print("Total number of transactions: ", len(X_train) + len(X_test))

    # 样本数据集分割
    X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)
    print("")
    print("Number transactions train dataset: ", len(X_train_undersample))
    print("Number transactions test dataset: ", len(X_test_undersample))
    print("Total number of transactions: ", len(X_train_undersample) + len(X_test_undersample))

    best_c = printing_Kfold_scores(X_train_undersample, y_train_undersample)

    '''混淆矩阵'''
    lr = LogisticRegression(C=best_c, penalty='l1')
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred_undersample = lr.predict(X_test_undersample.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample, y_pred_undersample)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()