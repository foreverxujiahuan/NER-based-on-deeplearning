'''
@Autor: xujiahuan
@Date: 2020-05-03 22:56:35
@LastEditors: xujiahuan
@LastEditTime: 2020-05-17 15:23:44
'''
from data import build_corpus
from models.rnn import RNN_Model
from metrics import Metrics
from utils import extend_maps
from utils import save_model
import time
from sklearn.externals import joblib
import json

# 制作数据
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus(train_path)
dev_word_lists, dev_tag_lists = build_corpus(dev_path, make_vocab=False)
test_word_lists, test_tag_lists = build_corpus(test_path, make_vocab=False)
bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)


def rnn_pred(train_word_lists, train_tag_lists, dev_word_lists,
             dev_tag_lists, test_word_lists, test_tag_lists):
    # start = time.time()
    # vocab_size = len(bilstm_word2id)
    # out_size = len(bilstm_tag2id)
    path = "ckpts/rnn.pkl"
    model = joblib.load(path)
    # model = RNN_Model(vocab_size, out_size)
    # model.train(train_word_lists, train_tag_lists,
    #             dev_word_lists, dev_tag_lists, bilstm_word2id, bilstm_tag2id)
    # save_model(model, "./ckpts/rnn.pkl")
    # print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    test_word_lists = [['许', '嘉', '欢', '来', '自', '郑', '州']]
    test_tag_lists = []
    length = len(test_word_lists[0])
    temp = ['O' for i in range(length)]
    test_tag_lists.append(temp)
    # test_word_lists = test_word_lists[0:1]
    # test_tag_lists = test_tag_lists[0:1]
    print(test_word_lists)
    print(test_tag_lists)
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


def rnn_pred2(test_word_lists, test_tag_lists):
    # start = time.time()
    # vocab_size = len(bilstm_word2id)
    # out_size = len(bilstm_tag2id)
    path = "ckpts/rnn.pkl"
    model = joblib.load(path)
    # model = RNN_Model(vocab_size, out_size)
    # model.train(train_word_lists, train_tag_lists,
    #             dev_word_lists, dev_tag_lists, bilstm_word2id, bilstm_tag2id)
    # save_model(model, "./ckpts/rnn.pkl")
    # print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    
    # test_word_lists = test_word_lists[0:1]
    # test_tag_lists = test_tag_lists[0:1]
    print(test_word_lists)
    print(test_tag_lists)
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


# rnn_pred = rnn_pred(train_word_lists, train_tag_lists, dev_word_lists,
#                     dev_tag_lists, test_word_lists, test_tag_lists)
# metrics = Metrics(rnn_pred, test_tag_lists)
# f1 = metrics.get_f1()
# print("RNN的f1得分为%.4f" % f1)
test_word_lists = [['许', '嘉', '欢', '来', '自', '郑', '州']]
test_tag_lists = []
length = len(test_word_lists[0])
temp = ['O' for i in range(length)]
test_tag_lists.append(temp)
rnn_pred2 = rnn_pred2(test_word_lists, test_tag_lists)
print(rnn_pred2)
