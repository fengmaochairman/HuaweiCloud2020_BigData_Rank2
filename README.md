赛题介绍——货物到港时间预测（ETA）
比赛提供2019至2020年航船历史运单GPS数据，数据主要包括运单号，承运商，采样点时间、经纬度坐标、速度、方向，所在船只及订单路由。目标是预测已知路由的测试集运单截取的各采样点到达目的港口时间（ETA）。

1 环境配置

. Python 3.6.4

. Pandas 0.22.0

. lightgbm 2.3.1

. Pytorch

2 数据下载

华为云赛事列表 https://competition.huaweicloud.com/information/1000037843/introduction

3 模型运行
. lightGBM模型

运行 code/lgbm/lgb_gps.ipynb

. DeepFM模型

运行 code/DeepFM/main.py
