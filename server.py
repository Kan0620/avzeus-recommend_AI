# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, make_response, redirect
import requests
import urllib.parse
from AVzeus import test_rec


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/inputted-data/<string:data>')
def inputted_data(data):
    # data: ユーザーが好きな順の女優5人のid配列
    list_data = data.split(',')
    # list_data: dataをint型の配列に変換したもの
    list_data = [int(i) for i in list_data]

    # ここからかんたその処理
    output = test_rec(list_data)
    # ここまでかんたその処理

    recommended_data = {
        "id": output[0],
        "state": output[1],
        "epsiron": output[2],
    }
    # Backendにリダイレクト
    
    redirect_url = 'http://localhost:8000/outputted-data?' + \
        'data='+str(recommended_data)
    return redirect(redirect_url)


@app.route('/sample')
def sample():
    return 'This is sampla page'


# export FLASK_APP=server.py
# flask run
# で起動
# localhost:5000/sample でサンプルページ表示
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
