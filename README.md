<!--
 * @Autor: xujiahuan
 * @Date: 2020-03-07 22:22:52
 * @LastEditors: xujiahuan
 * @LastEditTime: 2020-05-16 23:13:57
 -->
## 数据集
MSRA公开数据集

## 算法模型部分
采用HMMH、CRF、BiLSTM、BiLSTM+CRF、BERT

## 运行方式
HMM:

`python HMM.py`

CRF:

`python CRF.py`

BiLSTM:

`python BiLSTM.py`

BiLSTM_CRF:

`python BiLSTM_CRF.py`

BERT:

`python Bert.py`

web启动方式:

`cd web`

`python main.py`

## 开发部分
采用flask+vue搭建web可视化界面

## 评测指标
以完整的实体为单位，计算f1得分

## 各算法f1得分
HMM:0.5096

CRF:0.8569

RNN:0.5518

BiLSTM:0.7251

BiLSMT+CRF:0.7458

Bert:0.9224