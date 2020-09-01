import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.DeepFM import DeepFM
from data.dataset import CriteoDataset
import time
import warnings
warnings.filterwarnings("ignore")

def result_process(preds):
    test_data = pd.read_csv('data/raw/test_df.csv')
    test_data['travel_t'] = preds
    test_data['run_time'] = (pd.to_datetime(test_data['tmax']) - pd.to_datetime(
    test_data['timestamp'])).dt.total_seconds() / 3600
    test_data['arr_time'] = test_data['travel_t'] - test_data['run_time']

    result = test_data.groupby(['loadingOrder'], sort=False)['arr_time'].agg({'label': np.mean}).reset_index()
    test_data = test_data.merge(result, on='loadingOrder')
    test_data = test_data.drop_duplicates(['loadingOrder']).reset_index(drop=True)

    return test_data[['loadingOrder', 'TRANSPORT_TRACE', 'onboardDate', 'label']]


# sparse feature
sparse_features = ['start_year', 'start_month', 'start_weekofyear', 'is_covids']

# 训练集读取
train_raw = pd.read_csv('data/train_data.csv')
train_raw = train_raw.dropna(how='any', axis=0)
train_raw = train_raw.sample(frac = 1)  # 打乱数据

test_data = pd.read_csv('data/test_data.csv')
test_data['dummy_label'] = 0
print('train_data:{}\ttest_data:{}'.format(len(train_raw), len(test_data)))

# 测试集、训练集Null值检查
train_null = train_raw[train_raw.isnull().values == True]
test_null = test_data[test_data.isnull().values == True]
print('train_null:{}\ttest_null:{}'.format(len(train_null), len(test_null)))

train_data = train_raw.iloc[: int(len(train_raw) * 0.9), :] # 训练集
valid_data = train_raw.iloc[int(len(train_raw) * 0.9):, :]  # 测试集

# 数据集构建
batch_num = 5000
train_data = CriteoDataset(train_data, train = True)
loader_train = DataLoader(train_data, batch_size = batch_num)

# 验证集
val_data = CriteoDataset(valid_data, train = True)
loader_val = DataLoader(val_data, batch_size = 2000)

# 测试集
test_data = CriteoDataset(test_data, train = True)
loader_test = DataLoader(test_data)

# 读取每个特征的size
feature_sizes = np.loadtxt('./data/feature_sizes.txt', delimiter=',')
feature_sizes = [int(x) for x in feature_sizes]
print(feature_sizes)

# dense, sparse特征数量
n_sparse = len(sparse_features)
n_dense = len(train_raw.columns.tolist()) - n_sparse - 1

start = time.process_time()

# 模型训练
model = DeepFM(feature_sizes, n_dense, n_sparse,  use_cuda=False)
optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 0.0)
model.fit(loader_train, loader_val, optimizer, epochs = 50, verbose = True)

# 模型预测
y_pred = model.predict(loader_test)
print(y_pred)
results = result_process(np.array(y_pred))
print(results)
results.to_csv('../../code result/label/label0803.csv', index=False)

end = time.process_time()
print('Running time: %d seconds' % (end - start))



