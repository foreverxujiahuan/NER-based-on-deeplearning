'''
@Autor: xujiahuan
@Date: 2020-05-12 21:41:12
@LastEditors: xujiahuan
@LastEditTime: 2020-05-12 23:42:15
'''
from flask import Flask, render_template


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/xjh")
def xjh():
    return "你好,我叫许嘉欢"


if __name__ == '__main__':
    app.run()
