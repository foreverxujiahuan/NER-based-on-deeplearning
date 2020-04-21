'''
@Autor: xujiahuan
@Date: 2020-04-21 20:41:13
@LastEditors: xujiahuan
@LastEditTime: 2020-04-21 21:26:33
'''
from data import build_corpus
from models.bilstm_crf import BILSTM_Model
from metrics import Metrics
from utils import extend_maps, prepocess_data_for_lstmcrf
import time

# 制作数据
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus(train_path)
dev_word_lists, dev_tag_lists = build_corpus(dev_path, make_vocab=False)
test_word_lists, test_tag_lists = build_corpus(test_path, make_vocab=False)

bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
# print(bilstm_tag2id)


def bilstm_pred(train_word_lists, train_tag_lists, dev_word_lists,
                dev_tag_lists, test_word_lists, test_tag_lists):
    start = time.time()
    vocab_size = len(bilstm_word2id)
    out_size = len(bilstm_tag2id)
    model = BILSTM_Model(vocab_size, out_size, crf=False)
    model.train(train_word_lists, train_tag_lists,
                dev_word_lists, dev_tag_lists, bilstm_word2id, bilstm_tag2id)
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


bilstm_pred = bilstm_pred(train_word_lists, train_tag_lists, dev_word_lists,
                          dev_tag_lists, test_word_lists, test_tag_lists)
metrics = Metrics(bilstm_pred, test_tag_lists)
f1 = metrics.get_f1()
print("BiLSTM的f1得分为%.4f" % f1)
