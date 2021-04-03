# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, make_response, redirect
import urllib.parse
from AVzeus import test_rec


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/inputted-data/<string:data>', methods=['GET'])
def inputted_data(data):
    # data: ユーザーが好きな順の女優5人のid配列
    list_data = data.split(',')
    # list_data: dataをint型の配列に変換したもの
    list_data = [int(i) for i in list_data]

    # AI の処理
    outputs = test_rec(list_data)
    # ここまでAIの処理

    # outputをクエリに渡すために加工
    actresses_ids = ','.join(str(i) for i in outputs[0])
    states = ','.join(str(i) for i in outputs[1])
    epsilons = ','.join(str(i) for i in outputs[2])

    # クエリをエンコード
    query = 'actresses_ids='+actresses_ids+'&states='+states+'&epsilons='+epsilons
    # Backendにリダイレクト
    redirect_url = 'http://localhost:8000/outputted-data?' + query
    return redirect(redirect_url)


# export FLASK_APP=server.py + flask run または python3 server.py で起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
