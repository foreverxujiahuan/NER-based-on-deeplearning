'''
@Autor: xujiahuan
@Date: 2020-05-12 21:41:12
@LastEditors: xujiahuan
@LastEditTime: 2020-05-16 22:58:24
'''
from flask import Flask, render_template, request
from sklearn.externals import joblib
import json


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
        path = "ckpts/crf.pkl"
        model = joblib.load(path)
        test_word_lists = []
        temp = []
        leng = len(text)
        for i in range(leng):
            temp.append(text[i])
        test_word_lists.append(temp)
        pred = model.test(test_word_lists)
    if model == 'hmm':
        path = "ckpts/hmm.pkl"
        with open('../tag2id.json') as f:
            tag2id = json.load(f)
        with open('../word2id.json', encoding='utf-8') as f:
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
