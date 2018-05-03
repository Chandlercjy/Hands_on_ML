import pprint
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def info_entropy(features):
    sample_size = len(features)
    result = Counter(features)

    assert sample_size  # 分母不能为0
    ent = 0.0

    for value in result.values():
        p = value/sample_size
        ent += -p*np.log2(p)

    return ent


def conditional_entropy(features, label_set):
    feature_dict = defaultdict(list)

    for feature, result in zip(features, label_set):
        feature_dict[feature].append(result)

    ent = 0.0
    sample_size = len(label_set)

    for value in feature_dict.values():
        p = len(value)/sample_size * info_entropy(value)
        ent += p

    return ent


def intrinsic_value(features, label_set):
    feature_dict = defaultdict(list)

    for feature, result in zip(features, label_set):
        feature_dict[feature].append(result)

    iv = 0.0
    sample_size = len(label_set)

    for value in feature_dict.values():
        p = len(value)/sample_size
        iv += -p * np.log2(p)

    return iv


def info_gain(features, label_set):
    total_ent = info_entropy(label_set)
    conditional_ent = conditional_entropy(features, label_set)

    return total_ent - conditional_ent


def choose_max_gain(data_set, label_set, labels):
    result_dict = {}

    for i, value in enumerate(labels):
        features = data_set[:, i]
        result_dict[value] = info_gain(features, label_set)
    max_entropy = max(result_dict.values())

    for label in result_dict:
        if result_dict[label] == max_entropy:
            return label


def choose_max_gain_ratio_by_inspire(data_set, label_set, labels):
    entropy_dict = {}
    gain_ratio_dict = {}

    for i, value in enumerate(labels):
        features = data_set[:, i]
        entropy_dict[value] = info_gain(features, label_set)
    mean_entropy = np.mean(list(entropy_dict.values()))

    for key, value in entropy_dict.items():
        if value >= mean_entropy:
            gain_ratio_dict[key] = value
    max_gain_ratio = max(gain_ratio_dict.values())

    for label in gain_ratio_dict:
        if gain_ratio_dict[label] == max_gain_ratio:
            return label


def is_same_class(label_set):
    if len(np.unique(label_set)) == 1:
        return True

    return False


def is_empty(data_object):
    return len(data_object) == 0


def is_repeat(data_set):
    m, n = np.shape(data_set)

    for i in range(n):
        if len(np.unique(data_set[:, i])) != 1:
            return False

    return True


def get_most_common(value):
    return list(Counter(value).keys())[0]


class DecisionTree:

    def __init__(self, decide_func=choose_max_gain):
        self.tree = None
        self.labels = None
        self.result = None
        self.decide_func = decide_func

    def generate_tree(self, data_set, label_set, labels):
        data_set = np.array(data_set)

        if is_same_class(label_set):  # 当前结点包含的样本全属于同一类别，无需划分
            return list(np.unique(label_set))

        if is_empty(label_set) or is_repeat(data_set):  # 当前属性集为空，或是所有样本再所有属性上取值相同，无法划分
            return get_most_common(label_set)

        best_feature_label = self.decide_func(data_set, label_set, labels)
        best_feature_index = list(labels).index(best_feature_label)

        best_feature_col_unique = np.unique(data_set[:, best_feature_index])
        labels_temp = labels[labels != best_feature_label]

        tree = {best_feature_label: {}}

        for item in best_feature_col_unique:
            data_set_temp = []
            label_set_temp = []

            for i in range(len(data_set)):
                if item == data_set[i, best_feature_index]:  # 按照新feature将样本再分类
                    raw_new_data = data_set[i, :]
                    new_data = np.delete(
                        raw_new_data, best_feature_index, axis=0)
                    data_set_temp.append(new_data)
                    label_set_temp.append(label_set[i])

            if is_empty(data_set_temp):  # 或者当前结点包含的样本集合为空,不能划分
                return get_most_common(label_set)
            else:
                tree[best_feature_label][item] = self.generate_tree(
                    data_set_temp, label_set_temp, labels_temp)

        return tree

    def fit(self, data_set, label_set, labels):
        self.tree = self.generate_tree(data_set, label_set, labels)
        self.labels = labels

    def predict(self, data_test):
        self._predict(data_test, self.tree, self.labels)

        return self.result

    def _predict(self, data_test, tree, labels):

        if not isinstance(tree, dict):
            self.result = tree

            return

        for key, value in tree.items():
            index = list(labels).index(key)
            new_tree = value[data_test[index]]
            self._predict(data_test, new_tree, labels)


if __name__ == "__main__":

    df = pd.read_csv('data/watermelon_1.csv', index_col=0)
    result_col = '好瓜'
    data = df.loc[:, df.columns != result_col]
    res = df[result_col]

    DATA_SET = data.values
    LABEL_SET = res.values
    LABELS = data.columns.values

    test = DecisionTree(choose_max_gain_ratio_by_inspire)
    # test = DecisionTree(choose_max_gain)
    test.fit(DATA_SET, LABEL_SET, LABELS)
    test.predict(DATA_SET[1])
    print(test.result)
    pprint.pprint(test.tree)
