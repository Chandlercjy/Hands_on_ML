import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from matplotlib import style
from sklearn import linear_model, model_selection
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
style.use('ggplot')
# plt.rcParams['lines.linewidth'] = 1.4


def data_info_visulization1():
    data_train = pd.read_csv(TRAIN_FILE_PATH)
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2, 3), (0, 0))
    data_train['Survived'].value_counts().plot(kind='bar')
    plt.title('获救情况 1为获救', fontsize=10)
    plt.ylabel('人数', fontsize=10)

    plt.subplot2grid((2, 3), (0, 1))
    data_train['Pclass'].value_counts().plot(kind='bar')
    plt.title('乘客等级分布', fontsize=10)
    plt.ylabel('人数', fontsize=10)

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train['Survived'], data_train['Age'])
    plt.title('按年龄看获救情况', fontsize=10)
    plt.ylabel('年龄', fontsize=10)

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data_train['Age'][data_train['Pclass'] == 1].plot(kind='kde')
    data_train['Age'][data_train['Pclass'] == 2].plot(kind='kde')
    data_train['Age'][data_train['Pclass'] == 3].plot(kind='kde')
    plt.xlabel('年龄')
    plt.title('各等级的乘客年龄分布', fontsize=10)
    plt.ylabel('密度', fontsize=10)
    plt.legend(['头等舱', '二等舱', '三等舱'], loc='best')

    plt.subplot2grid((2, 3), (1, 2))
    data_train['Embarked'].value_counts().plot(kind='bar')
    plt.title('各登船口岸上船人数', fontsize=10)
    plt.ylabel('人数', fontsize=10)


def data_info_visulization2():
    data_train = pd.read_csv(TRAIN_FILE_PATH)
    fig = plt.figure()

    ax1 = fig.add_subplot(241)
    survived_0 = data_train['Pclass'][data_train['Survived']
                                      == 0].value_counts()
    survived_1 = data_train['Pclass'][data_train['Survived']
                                      == 1].value_counts()
    df = pd.DataFrame({'获救': survived_1, '未获救': survived_0})
    df.plot(kind='bar', ax=ax1, stacked=True)
    ax1.set_title('各乘客等级的获救情况', fontsize=10)
    ax1.set_xlabel('乘客等级')
    ax1.set_ylabel('人数')

    ax2 = fig.add_subplot(242, sharey=ax1)
    survived_m = data_train['Survived'][data_train['Sex']
                                        == 'male'].value_counts()
    survived_f = data_train['Survived'][data_train['Sex']
                                        == 'female'].value_counts()
    df = pd.DataFrame({'男性': survived_m, '女性': survived_f})
    df.plot(kind='bar', ax=ax2, stacked=True)
    ax2.set_title('按性别看获救情况', fontsize=10)
    ax2.set_xlabel('性别')
    ax2.set_ylabel('人数')

    ax3 = fig.add_subplot(243, sharey=ax1)
    survived_0 = data_train['Embarked'][data_train['Survived']
                                        == 0].value_counts()
    survived_1 = data_train['Embarked'][data_train['Survived']
                                        == 1].value_counts()
    df = pd.DataFrame({'获救': survived_1, '未获救': survived_0})
    df.plot(kind='bar', ax=ax3, stacked=True)
    ax3.set_title('各登录港口乘客的获救情况', fontsize=10)
    ax3.set_xlabel('登录港口')
    ax3.set_ylabel('人数')

    ax4 = fig.add_subplot(244, sharey=ax1)
    survived_cabin = data_train['Survived'][
        data_train['Cabin'].notnull()].value_counts()
    survived_nocabin = data_train['Survived'][
        data_train['Cabin'].isnull()].value_counts()
    df = pd.DataFrame({'有': survived_cabin, '无': survived_nocabin})
    df.plot(kind='bar', ax=ax4, stacked=True)
    ax4.set_title('按照Cabin有无看获救情况', fontsize=10)
    ax4.set_xlabel('Cabinb有无')
    ax4.set_ylabel('人数')

    ax5 = fig.add_subplot(245)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts(
    ).plot(kind='bar', label="female highclass", color='#FA2479', ax=ax5)
    ax5.set_xticklabels([u"获救", u"未获救"], rotation=0)
    ax5.legend([u"女性/高级舱"], loc='best')

    ax6 = fig.add_subplot(246, sharey=ax5)
    data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts(
    ).plot(kind='bar', label='female, low class', color='pink')
    ax6.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"女性/低级舱"], loc='best')

    ax7 = fig.add_subplot(247, sharey=ax5)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts(
    ).plot(kind='bar', label='male, high class', color='lightblue')
    ax7.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/高级舱"], loc='best')

    ax8 = fig.add_subplot(248, sharey=ax5)
    data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts(
    ).plot(kind='bar', label='male low class', color='steelblue')
    ax8.set_xticklabels([u"未获救", u"获救"], rotation=0)
    plt.legend([u"男性/低级舱"], loc='best')


def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]


# # 数据可视化
# data_info_visulization1()
# data_info_visulization2()
# plt.show()

# 数据预处理


def set_missing_ages(df):  # 用随机森林填补空缺值
    # data_train.info() # 通过这一行代码发现Age有空缺值
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0]  # 年龄，作为特征值
    X = known_age[:, 1:]  # 其他列作为特征值
    random_forest = RandomForestRegressor(
        random_state=0, n_estimators=2000)
    random_forest.fit(X, y)
    predict_age = random_forest.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predict_age

    return df, random_forest


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()), 'Cabin'] = 'No'

    return df


def process_train_data(df):

    df, random_forest = set_missing_ages(df)
    df = set_cabin_type(df)

    dummies_cabin = pd.get_dummies(df.Cabin, prefix='Cabin')
    dummies_embarked = pd.get_dummies(df.Embarked, prefix='Embarked')
    dummies_sex = pd.get_dummies(df.Sex, prefix='Sex')
    dummies_pclass = pd.get_dummies(df.Pclass, prefix='Pclass')
    new_data = pd.concat([df[['Survived']], dummies_cabin,
                          dummies_embarked, dummies_sex, dummies_pclass], axis=1)

    new_data['Age_scaled'] = preprocessing.scale(df['Age'])
    new_data['Fare_scaled'] = preprocessing.scale(df['Fare'])

    return new_data, random_forest


def process_test_data(df, processer):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    predict_age = processer.predict(unknown_age[:, 1::])
    df.loc[(df.Age.isnull()), 'Age'] = predict_age
    df = set_cabin_type(df)

    dummies_cabin = pd.get_dummies(df.Cabin, prefix='Cabin')
    dummies_embarked = pd.get_dummies(df.Embarked, prefix='Embarked')
    dummies_sex = pd.get_dummies(df.Sex, prefix='Sex')
    dummies_pclass = pd.get_dummies(df.Pclass, prefix='Pclass')
    new_data = pd.concat([dummies_cabin,
                          dummies_embarked, dummies_sex, dummies_pclass], axis=1)

    new_data['Age_scaled'] = preprocessing.scale(df['Age'])
    df['Fare'].fillna(0, inplace=True)
    new_data['Fare_scaled'] = preprocessing.scale(df['Fare'])

    return new_data


def logistic_regression():
    data_train = pd.read_csv(TRAIN_FILE_PATH)
    # 逻辑回归建模
    raw_data_train, processer = process_train_data(data_train)
    data_train = raw_data_train.as_matrix()
    y = data_train[:, 0]  # 取survived
    X = data_train[:, 1:]  # 取survived

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    # 同样对test_data预处理
    raw_data_test = pd.read_csv(TEST_FILE_PATH)
    data_test = process_test_data(raw_data_test, processer)

    # 预测
    predictions = clf.predict(data_test.as_matrix())
    predict_result = pd.DataFrame(dict(
        PassengerId=raw_data_test.PassengerId.as_matrix(),
        Survived=predictions.astype(np.int32)
    ))

    predict_coef = pd.DataFrame(
        {'columns': list(raw_data_train.columns)[1:], "coef": list(clf.coef_.T)})
    # print(predict_result)
    # predict_result.to_csv(RESULT_OUTPUT_PATH, index=False)
    print(predict_coef)

    return clf


def bagging_logistic_regression():
    data_train = pd.read_csv(TRAIN_FILE_PATH)
    # 逻辑回归建模
    raw_data_train, processer = process_train_data(data_train)
    data_train = raw_data_train.as_matrix()
    y = data_train[:, 0]  # 取survived
    X = data_train[:, 1:]  # 取survived

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8,
                                   max_features=1, bootstrap=True, bootstrap_features=False, n_jobs=-1)

    # 同样对test_data预处理
    raw_data_test = pd.read_csv(TEST_FILE_PATH)
    data_test = process_test_data(raw_data_test, processer)

    # 预测
    predictions = bagging_clf.predict(data_test.as_matrix())
    predict_result = pd.DataFrame(dict(
        PassengerId=raw_data_test.PassengerId.as_matrix(),
        Survived=predictions.astype(np.int32)
    ))
    # predict_result.to_csv(RESULT_OUTPUT_PATH, index=False)
    print(predict_result)


def cross_validation_on_logistic():
    data_train = pd.read_csv(TRAIN_FILE_PATH)
    raw_data_train, processer = process_train_data(data_train)
    data_train = raw_data_train.as_matrix()
    y = data_train[:, 0]  # 取survived
    X = data_train[:, 1:]  # 取survived

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    score = model_selection.cross_val_score(clf, X, y, cv=5)
    print(score)


def show_bad_cross_validation_on_logistic():
    origin_data_train = pd.read_csv(TRAIN_FILE_PATH)
    raw_data_train, processer = process_train_data(origin_data_train)
    split_train, split_cv = model_selection.train_test_split(
        raw_data_train, test_size=0.3, random_state=0)

    # 生成模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(split_train.as_matrix()[:, 1:], split_train.as_matrix()[:, 0])

    # 对cross validation数据进行预测
    predictions = clf.predict(split_cv.as_matrix()[:, 1:])

    bad_cases_index = split_cv[split_cv.as_matrix()[:, 0] != predictions].index
    bad_cases = origin_data_train.loc[bad_cases_index].sort_index()
    print(bad_cases)


def plot_learning_curve_func(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                             train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    train_sizes: 每次取数据集的百分比
    """
    style.use('ggplot')
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)

        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean,
                 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean,
                 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) +
                (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]
            ) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff


def plot_learning_curve():
    data_train = pd.read_csv(TRAIN_FILE_PATH)
    # 逻辑回归建模
    raw_data_train, processer = process_train_data(data_train)
    data_train = raw_data_train.as_matrix()
    y = data_train[:, 0]  # 取survived
    X = data_train[:, 1:]  # 取survived

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    plot_learning_curve_func(clf, "学习曲线", X, y)


TRAIN_FILE_PATH = 'data/Titanic/train.csv'
TEST_FILE_PATH = 'data/Titanic/test.csv'
RESULT_OUTPUT_PATH = 'chandler_titanic.csv'


# data_train.describe()
# data_train.info()
# logistic_regression()
# cross_validation_on_logistic()
# show_bad_cross_validation_on_logistic()
# plot_learning_curve()
bagging_logistic_regression()
