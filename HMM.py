'''
@Autor: xujiahuan
@Date: 2020-04-21 12:37:36
@LastEditors: xujiahuan
@LastEditTime: 2020-04-21 18:53:33
'''
from data import build_corpus
from models.hmm import HMM
from metrics import Metrics

# 制作数据
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus(train_path)
dev_word_lists, dev_tag_lists = build_corpus(dev_path, make_vocab=False)
test_word_lists, test_tag_lists = build_corpus(test_path, make_vocab=False)


def hmm_pred(train_word_lists, train_tag_lists, test_word_lists,
             test_tag_lists, word2id, tag2id):
    model = HMM(len(tag2id), len(word2id))
    model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    pred = model.test(test_word_lists, word2id, tag2id)
    return pred


print("正在训练HMM...")
hmm_pred = hmm_pred(train_word_lists, train_tag_lists, test_word_lists,
                    test_tag_lists, word2id, tag2id)
print("训练完毕...")
print("正在评估HMM...")
metrics = Metrics(hmm_pred, test_tag_lists)
f1 = metrics.get_f1()
print("HMM的f1得分为%.4f" % f1)
