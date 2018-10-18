import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
import re

if __name__ == '__main__':
    data = pd.read_csv('data/titanic_train.csv')
    print(data.info())
    print(data.head())
    # 填充缺失数据
    data['Age'] = data['Age'].fillna(data['Age'].median())
    # dataFrame 条件筛选
    # 更改数据形式
    data.loc[data.Sex == 'male', 'Sex'] = 0
    data.loc[data.Sex == 'female', 'Sex'] = 1
    data.Embarked = data.Embarked.fillna('S')
    data.loc[data.Embarked == 'S', 'Embarked'] = 0
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2

    '''
    ===========线性回归===================================================
    '''
    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # # 普通最小二乘线性回归
    # lg = LinearRegression()
    # kf = KFold(n_splits=3, random_state=1)
    # predictions = []
    # for train_index, test_index in kf.split(data):  # 直接把数据切分，而不是data.shape[0]
    #     # index 用位置索引 , columns 用属性索引
    #     train_predictors = data[predictors].iloc[train_index]
    #     train_target = data['Survived'].iloc[train_index]
    #     lg.fit(train_predictors, train_target)
    #     test_predictors = data[predictors].iloc[test_index]
    #     # 交叉验证集的 标签集
    #     # test_target = data['Survived'].iloc[test_index]
    #     test_prediction = lg.predict(test_predictors)  # 线性回归预测结果为数值，逻辑回归为0，1 分类
    #     predictions.append(test_prediction)  # 线性回归预测结果为数值
    # print(predictions)
    # print('长度为：', len(predictions))  # 长度为3， 3折交叉验证
    # # 沿纵向把合并起来
    # predictions = np.concatenate(predictions, axis=0)
    # # 把数值转换为分类
    # predictions[predictions > 0.5] = 1
    # predictions[predictions <= 0.5] = 0
    # # k 份交叉训练集合起来相当于整个训练集
    # accuracy = sum(predictions[predictions == data['Survived']]) / len(predictions)
    # print('训练集准确度为：', accuracy)

    '''
    =================逻辑回归==========================================================
    '''
    x = data[predictors]
    y = data['Survived']
    lg = LogisticRegression(random_state=1)
    # 通过交叉验证评估分数
    # 返回值：scores : array of float, shape=(len(list(cv)),) 每次交叉验证运行的估算器 分数 数组
    scores = cross_val_score(lg, x, y, cv=3)
    print('scores 长度：', scores)
    print(scores.mean())

    '''
    =================随机森林=========================================================
    '''
    rfc = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
    scores = cross_val_score(rfc, x, y, cv=3)
    print('随机森林：', scores.mean())

    '''
    ================比较特征的重要性===============================================================
    '''
    # 新增加2列
    data['FamilySize'] = data['SibSp'] + data['Parch']
    # 通过将匿名函数作为参数传递来apply()。
    data['NameLength'] = data['Name'].apply(lambda x: len(x))

    # A function to get the title from a name.
    def get_title(name):
        # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
        title_search = re.search(' ([A-Za-z]+)\.', name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""

    # Get all the titles and print how often each one occurs.
    titles = data["Name"].apply(get_title)
    print(pd.value_counts(titles))
    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9,
                     "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    # Verify that we converted everything.
    print(pd.value_counts(titles))
    # Add in the title column.
    data["Title"] = titles

    predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "NameLength"]

    '''
    f_classif（X，y ）: 计算所提供样品的ANOVA F值。返回值为，每个特征得分的集合
        X : {array-like, sparse matrix} shape = [n_samples, n_features]
        y : array of shape(n_samples)
        返回值：F 值: array, shape = [n_features,]
    SelectKBest（score_func = <function f_classif>，k = 10 ）：默认函数为f_classif，仅适用于分类任务，k:选择k个特征根据scores由高到低
        fit（X，y）: 在（X，y）上运行score 函数并获得适当的功能。
        返回值：object类型
    '''
    # 根据k个最高分选择功能。scores按升序排序，选择排前k名所对应的特征
    selector = SelectKBest(f_classif, k=10)
    selector.fit(data[predictors], data['Survived'])

    # 获取每个特征的原始 p值，并将p值转换为得分
    scores = -np.log10(selector.pvalues_)
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

    # 根据特征重要性选择特征.再用随机森林试试
    predictors = ["Pclass", "Sex", "Fare", "Title"]
    alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
    scores = cross_val_score(alg, x, y, cv=3)
    print('随机森林优化特征：', scores.mean())

    '''
    =================集成算法=========================================================================
    '''
    # The algorithms we want to ensemble.
    # We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
         ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
    ]
    kf = KFold(3, random_state=1)

    predictions = []
    for train, test in kf.split(data):
        train_target = data["Survived"].iloc[train]
        full_test_predictions = []
        # Make predictions for each algorithm on each fold
        for alg, predictors in algorithms:
            # Fit the algorithm on the training data.
            alg.fit(data[predictors].iloc[train], train_target)
            '''
            Select and predict on the test fold.
            The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
            .astype（float）是将数据转换为所有浮点数并避免sklearn错误所必需的
            算法的返回值是object，需要转化为float类型。用 .astype(float)
            '''
            test_predictions = alg.predict_proba(data[predictors].iloc[test].astype(float))[:, 1]
            full_test_predictions.append(test_predictions)
        # Use a simple ensembling scheme -- just average the predictions to get the final classification.
        test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
        # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
        test_predictions[test_predictions <= .5] = 0
        test_predictions[test_predictions > .5] = 1
        predictions.append(test_predictions)

    # Put all the predictions together into one array.
    # 合并，沿纵轴
    predictions = np.concatenate(predictions, axis=0)

    # Compute accuracy by comparing to the training data.
    # K 折交叉验证，把每次的验证集合并起来就是整个训练集的 标签集
    accuracy = sum(predictions[predictions == data["Survived"]]) / len(predictions)
    print(accuracy)

    '''
    =============预测======================================================
    '''
    data_test = pd.read_csv('data/titanic_test.csv')
    data_test['Fare'] = data_test['Fare'].fillna(data_test['Fare'].median())
    data_test['Age'] = data_test['Age'].fillna(data_test['Age'].median())
    data_test.loc[data_test['Sex'] == 'male', 'Sex'] = 0
    data_test.loc[data_test['Sex'] == 'female', 'Sex'] = 1
    data_test.loc[data_test['Embarked'] == 'S', 'Embarked'] = 0
    data_test.loc[data_test['Embarked'] == 'C', 'Embarked'] = 1
    data_test.loc[data_test['Embarked'] == 'Q', 'Embarked'] = 2
    titles = data_test["Name"].apply(get_title)
    # We're adding the Dona title to the mapping, because it's in the test set, but not the training set
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                     "Mme": 8, "Don": 9,
                     "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
    for k, v in title_mapping.items():
        titles[titles == k] = v
    data_test["Title"] = titles
    # Check the counts of each unique title.
    print(pd.value_counts(data_test["Title"]))

    # Now, we add the family size column.
    data_test["FamilySize"] = data_test["SibSp"] + data_test["Parch"]
    predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
    algorithms = [
        [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
        [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
    ]

    full_predictions = []
    for alg, predictors in algorithms:
        # Fit the algorithm using the full training data.
        alg.fit(data[predictors], data["Survived"])
        # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
        predictions = alg.predict_proba(data_test[predictors].astype(float))[:, 1]
        full_predictions.append(predictions)

    # The gradient boosting classifier generates better predictions, so we weight it higher.
    predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
    predictions[predictions <= .5] = 0
    predictions[predictions > .5] = 1
    print('预测结果', predictions)