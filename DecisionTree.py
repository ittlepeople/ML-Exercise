import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

if __name__ == '__main__':
    housing = fetch_california_housing()
    # print(housing.DESCR)
    #
    # #画决策树
    # dtr = tree.DecisionTreeRegressor(max_depth=2)
    # dtr.fit(housing.data[:, [6, 7]], housing.target)
    # dot_data = tree.export_graphviz(dtr, out_file=None, feature_names=housing.feature_names[6:8], filled=True, impurity=False, rounded=True)
    #
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.get_nodes()[7].set_fillcolor("#FFF2DD")  # 得到7个节点
    # Image(graph.create_png())
    # graph.write_png("dtr_white_background.png")
    # print('end..........')

    '''
    ======决策树============================================================================
    '''
    # 决策树
    data_train, data_test, target_train, target_test = train_test_split(housing.data, housing.target, test_size=0.1, random_state=42)
    dtr = tree.DecisionTreeRegressor(random_state=42)
    dtr.fit(data_train, target_train)
    tr_score = dtr.score(data_test, target_test)
    print(tr_score)

    # 随机森林
    rfr = RandomForestRegressor(random_state=42)
    rfr.fit(data_train, target_train)
    forest_score = rfr.score(data_test, target_test)
    print(forest_score)

    # Exhaustive search over specified parameter values for an estimator.
    # 彻底搜索估计器的指定参数值。
    tree_param_grid = {'min_samples_split': list((3, 6, 9)), 'n_estimators': list((10, 50, 100))}
    grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)
    grid.fit(data_train, target_train)
    print('----', grid.cv_results_, grid.best_params_, grid.best_score_)
    print('结束！')





