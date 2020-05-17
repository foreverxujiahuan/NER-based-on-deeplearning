'''
@Autor: xujiahuan
@Date: 2020-05-06 11:00:17
@LastEditors: xujiahuan
@LastEditTime: 2020-05-17 16:51:08
'''
from sklearn.externals import joblib
import json
from utils import extend_maps

test_word_lists = [['上', '海', '浦', '东']]
with open('tag2id.json') as f:
    tag2id = json.load(f)
with open('word2id.json', encoding='utf-8') as f:
    word2id = json.load(f)
bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=True)


path = "ckpts/lstm_crf.pkl"
model = joblib.load(path)
test_tag_lists = []
length = len(test_word_lists[0])
temp = ['O' for i in range(length)]
test_tag_lists.append(temp)
pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
print(pred_tag_lists)
