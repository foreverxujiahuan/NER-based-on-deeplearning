'''
@Autor: xujiahuan
@Date: 2020-04-23 13:53:21
@LastEditors: xujiahuan
@LastEditTime: 2020-04-23 14:55:52
'''
from data import build_corpus
from metrics import Metrics
import requests

# 制作数据
train_path = 'data/train.txt'
dev_path = 'data/dev.txt'
test_path = 'data/test.txt'
train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus(train_path)
dev_word_lists, dev_tag_lists = build_corpus(dev_path, make_vocab=False)
test_word_lists, test_tag_lists = build_corpus(test_path, make_vocab=False)


def pred(s):
    temp = ''
    for t in s:
        temp = temp + t
    url_params = {'s': temp}
    r = requests.post('http://127.0.0.1:6000', params=url_params)
    res = r.json()
    return res


flag = 0
bert_preds = []
for word_list in test_word_lists:
    bert_pred = pred(word_list)
    bert_preds.append(bert_pred)
    flag = flag + 1
    if flag % 100 == 0:
        print(flag)


print("正在评估bert:")
metrics = Metrics(bert_preds, test_tag_lists)
f1 = metrics.get_f1()
print("bert的f1得分为%.4f" % f1)
