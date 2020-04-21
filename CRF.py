'''
@Autor: xujiahuan
@Date: 2020-04-21 20:12:02
@LastEditors: xujiahuan
@LastEditTime: 2020-04-21 20:19:10
'''
from data import build_corpus
from models.crf import CRFModel
from metrics import Metrics

# 制作数据
# 制作数据
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
train_word_lists, train_tag_lists = build_corpus(train_path, make_vocab=False)
dev_word_lists, dev_tag_lists = build_corpus(dev_path, make_vocab=False)
test_word_lists, test_tag_lists = build_corpus(test_path, make_vocab=False)


def crf_pred(train_word_lists, train_tag_lists, test_word_lists,
             test_tag_lists):
    model = CRFModel()
    model.train(train_word_lists, train_tag_lists)
    pred = model.test(test_word_lists)
    return pred


print("正在训练CRF...")
crf_pred = crf_pred(train_word_lists, train_tag_lists, test_word_lists,
                    test_tag_lists)
print("训练完毕...")
print("正在评估CRF...")
metrics = Metrics(crf_pred, test_tag_lists)
f1 = metrics.get_f1()
print("HMM的f1得分为%.4f" % f1)
