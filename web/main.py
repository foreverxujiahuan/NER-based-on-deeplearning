'''
@Autor: xujiahuan
@Date: 2020-05-12 21:41:12
@LastEditors: xujiahuan
@LastEditTime: 2020-05-14 13:49:16
'''
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


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
