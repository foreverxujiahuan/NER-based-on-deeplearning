'''
@Autor: xujiahuan
@Date: 2020-05-06 11:00:17
@LastEditors: xujiahuan
@LastEditTime: 2020-05-16 21:24:20
'''
from sklearn.externals import joblib
import json

test_word_lists = [['许', '嘉', '欢', '来', '自', '郑', '州']]


path = "web/ckpts/crf.pkl"
model = joblib.load(path)
pred = model.test(test_word_lists)
print(pred)
