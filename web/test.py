'''
@Autor: xujiahuan
@Date: 2020-05-06 11:00:17
@LastEditors: xujiahuan
@LastEditTime: 2020-05-16 21:45:32
'''
from sklearn.externals import joblib

test_word_lists = [['许', '嘉', '欢', '来', '自', '郑', '州']]

# with open('tag2id.json') as f:
#     tag2id = json.load(f)
# with open('word2id.json', encoding='utf-8') as f:
#     word2id = json.load(f)


path = "../ckpts/crf.pkl"
model = joblib.load(path)
pred = model.test(test_word_lists)
print(pred)
