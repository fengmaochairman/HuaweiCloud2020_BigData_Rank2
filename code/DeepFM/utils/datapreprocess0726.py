import os
import pandas as pd
import random
import collections
from sklearn.preprocessing import MinMaxScaler
import time

# sparse feature
sparse_features = ['start_year', 'start_month', 'start_weekofyear', 'is_covids']

COLUMNS = ['longitude', 'latitude', 'speed', 'direction',  'start_lon', 'start_lat', 'end_lon',
           'end_lat', 'lon_diff', 'lat_diff',  'lon_diff_start', 'lat_diff_start',
           'dis_diff_start', 'lon_diff_end', 'lat_diff_end', 'dis_diff_end',  'start_year', 'start_month',
           'start_weekofyear', 'is_covids', 'trace_Dir_mean', 'trace_Dir_max',
           'trace_Dir_min', 'trace_Dir_median', 'trace_Dir_std', 'trace_aver_speed_mean', 'trace_aver_speed_max',
           'trace_aver_speed_min', 'trace_aver_speed_median', 'trace_aver_speed_std', 'trace_dis_mean',
           'trace_dis_max', 'trace_dis_min', 'trace_dis_median', 'trace_dis_std', 'trace_end_anchor_time_mean',
           'trace_end_anchor_time_max', 'trace_end_anchor_time_min', 'trace_end_anchor_time_median',
           'trace_end_anchor_time_std', 'trace_label_mean', 'trace_label_max', 'trace_label_min',
           'trace_label_median', 'trace_label_std', 'label']

class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """
    def __init__(self, num_feature):
        self.dicts = []
        self.feature_df = pd.DataFrame()
        self.feature_name, self.feature_count = [], []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    # 对每列出现的category特征（过滤出现频率小于cutoff的特征，统一赋值为0）进行数字编码
    def build(self, df, categorial_features, cutoff = 0):
        for i in range(0, self.num_feature):
            self.feature_df = df.iloc[:, categorial_features[i]].value_counts()
            self.feature_name = self.feature_df.index.tolist()
            self.feature_count = self.feature_df.values.tolist()
            for j in range(len(self.feature_name)):
                self.dicts[i][self.feature_name[j]] = int(self.feature_count[j])

        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key = lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    # 类别特征编码，缺失补0
    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = 0
            # res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    # 返回各列category特征的数量
    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]

def get_feature_type(columns):
    columns.remove('label')
    dense_features = [c for c in columns if c not in sparse_features]
    features = dense_features + sparse_features

    return dense_features, features

def norm_continuous_feature(df, scaler):
    # 连续特征归一化
    df_dense = df[dense_features]
    df_dense = scaler.fit_transform(df_dense)
    df_dense = pd.DataFrame(df_dense, columns = dense_features)

    # 其他特征
    df = df.drop(dense_features, axis = 1)
    new_df = pd.concat([df_dense, df], axis = 1)

    return new_df

def preprocess(train_df, test_df):
    # 测试集、训练集Null值检查
    train_null = train_df[train_df.isnull().values == True]
    test_null = test_df[test_df.isnull().values == True]
    print('train_null:{}\ttest_null:{}'.format(len(train_null), len(test_null)))

    # 连续特征归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df = norm_continuous_feature(train_df, scaler)
    test_df = norm_continuous_feature(test_df, scaler)

    # 类别特征创建字典
    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(train_df, categorial_features, cutoff = 0)
    print('build category dict:{} seconds'.format(time.process_time() - start))

    dict_sizes = dicts.dicts_sizes()  # 获取每列category特征的数目
    print('dict_sizes:\n{}'.format(dict_sizes))

    # 类别特征特征数间隔
    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
    # print(categorial_feature_offset)

    # 各Field特征个数保存
    with open(os.path.join('../data/feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [1] * len(continuous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))
    random.seed(0)

    # 训练集特征处理、保存
    train_df_sparse = []
    for i in range(len(train_df)):
        if (i % 50000 == 0):
            print('i:{}'.format(i))

        # 类别特征编码
        categorial_vals = []
        for j in range(0, len(categorial_features)):
            # val = dicts.gen(j, train_df.iloc[i, categorial_features[j]]) + categorial_feature_offset[j]
            val = dicts.gen(j, train_df.iloc[i, categorial_features[j]])
            categorial_vals.append(val)
        train_df_sparse.append(categorial_vals)
    train_df_sparse = pd.DataFrame(train_df_sparse, columns = sparse_features)

    train_data = train_df.drop(sparse_features + ['label'], axis = 1)
    train_data = pd.concat([train_data, train_df_sparse, train_df['label']], axis = 1)
    train_data.to_csv('../data/train_data.csv', index = False)
    print('train encode:{} seconds'.format(time.process_time() - start))

    # 测试集特征处理、保存
    test_df_sparse = []
    for i in range(len(test_df)):
        # 类别特征编码
        categorial_vals = []
        for j in range(0, len(categorial_features)):
            # val = dicts.gen(j, test_df.iloc[i, categorial_features[j]]) + categorial_feature_offset[j]
            val = dicts.gen(j, test_df.iloc[i, categorial_features[j]])
            categorial_vals.append(val)
        test_df_sparse.append(categorial_vals)
    test_df_sparse = pd.DataFrame(test_df_sparse, columns = sparse_features)

    test_data = test_df.drop(sparse_features, axis = 1)
    test_data = pd.concat([test_data, test_df_sparse], axis = 1)
    test_data.to_csv('../data/test_data.csv', index=False)


if __name__ == "__main__":
    start = time.process_time()

    # 训练集
    train_df = pd.read_csv('../data/raw/train_df.csv')
    train_df = train_df.dropna(how = 'any', axis = 0)
    train_df = train_df[COLUMNS]
    # train_df = train_df.drop(['loadingOrder', 'timestamp', 'TRANSPORT_TRACE', 'vesselMMSI', 'tmax'], axis = 1)
    print(len(train_df))

    # label, sparse, dense特征分块
    dense_features, features = get_feature_type(train_df.columns.tolist())
    train_df = train_df[features + ['label']]

    # dense, sparse特征位置
    continuous_features = range(0, len(dense_features))
    categorial_features = range(len(dense_features), len(features))

    # 测试集
    test_df = pd.read_csv('../data/raw/test_df.csv')
    test_df = test_df[features]
    print(len(test_df))

    # 特征处理
    preprocess(train_df, test_df)

    print('running time: {} seconds'.format(time.process_time() - start))










