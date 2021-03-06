'''
@Autor: xujiahuan
@Date: 2020-04-21 20:57:09
@LastEditors: xujiahuan
@LastEditTime: 2020-05-16 23:52:17
'''
# 设置lstm训练参数


class TrainingConfig(object):
    batch_size = 32
    # 学习速率
    lr = 0.001
    epoches = 30
    print_step = 5


class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数


class RNNConfig():
    emb_size = 128  # 词向量维数
    hidden_size = 128  # 隐向量维数
