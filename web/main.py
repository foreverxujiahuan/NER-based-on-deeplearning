'''
@Autor: xujiahuan
@Date: 2020-05-12 21:41:12
@LastEditors: xujiahuan
@LastEditTime: 2020-05-17 15:44:28
'''
from flask import Flask, render_template, request
from sklearn.externals import joblib
import json
from utils import extend_maps


app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    text = request.form.get("text")
    model = request.form.get("model")
    pred = ner(text, model)
    if len(pred) != 0:
        pred = pred[0]
    return render_template("index.html", pred=pred, text=text)


def ner(text, model):
    pred = []
    if model == "crf":
        pred = crf(text)
    if model == 'hmm':
        pred = hmm(text)
    if model == 'rnn':
        pred = rnn(text)
    if model == 'lstm':
        pred = lstm(text)
    if model == 'lstm_crf':
        pred = lstm_crf(text)
    return pred


def lstm_crf(text):
    path = "../ckpts/lstm_crf.pkl"
    model = joblib.load(path)
    test_tag_lists = []
    length = len(text)
    temp = ['O' for i in range(length)]
    test_tag_lists.append(temp)
    with open('tag2id.json') as f:
        tag2id = json.load(f)
    with open('word2id.json', encoding='utf-8') as f:
        word2id = json.load(f)
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    test_word_lists = []
    text_temp = []
    for i in range(len(text)):
        text_temp.append(text[i])
    test_word_lists.append(text_temp)
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


def lstm(text):
    path = "../ckpts/lstm.pkl"
    model = joblib.load(path)
    test_tag_lists = []
    length = len(text)
    temp = ['O' for i in range(length)]
    test_tag_lists.append(temp)
    with open('tag2id.json') as f:
        tag2id = json.load(f)
    with open('word2id.json', encoding='utf-8') as f:
        word2id = json.load(f)
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    test_word_lists = []
    text_temp = []
    for i in range(len(text)):
        text_temp.append(text[i])
    test_word_lists.append(text_temp)
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


def rnn(text):
    path = "../ckpts/rnn.pkl"
    model = joblib.load(path)
    test_tag_lists = []
    length = len(text)
    temp = ['O' for i in range(length)]
    test_tag_lists.append(temp)
    with open('tag2id.json') as f:
        tag2id = json.load(f)
    with open('word2id.json', encoding='utf-8') as f:
        word2id = json.load(f)
    bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
    test_word_lists = []
    text_temp = []
    for i in range(len(text)):
        text_temp.append(text[i])
    test_word_lists.append(text_temp)
    pred_tag_lists, test_tag_lists = model.test(
        test_word_lists, test_tag_lists, word2id, tag2id)
    return pred_tag_lists


def crf(text):
    path = "ckpts/crf.pkl"
    model = joblib.load(path)
    test_word_lists = []
    temp = []
    leng = len(text)
    for i in range(leng):
        temp.append(text[i])
    test_word_lists.append(temp)
    pred = model.test(test_word_lists)
    return pred


def hmm(text):
    path = "ckpts/hmm.pkl"
    with open('tag2id.json') as f:
        tag2id = json.load(f)
    with open('word2id.json', encoding='utf-8') as f:
        word2id = json.load(f)
    model = joblib.load(path)
    test_word_lists = []
    temp = []
    leng = len(text)
    for i in range(leng):
        temp.append(text[i])
    test_word_lists.append(temp)
    pred = model.test(test_word_lists, word2id, tag2id)
    return pred


@app.route("/xjh")
def xjh():
    return "你好,我叫许嘉欢"


@app.route("/login", methods=['POST', 'GET'])
def login():
    username = request.form.get("username")
    password = request.form.get("pwd")
    print(username)
    print(password)
    if username == "xjh":
        return "成功"
    else:
        return "失败"


if __name__ == '__main__':
    app.run()
