# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, make_response, redirect
from AVzeus import recommend


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/recommendation', methods=['GET'])
def recommendation():
    data = request.args.get("selected_wemen_ids")
    list_data = data.split(',')
    # list_data: dataをint型の配列に変換したもの
    list_data = [int(i) for i in list_data]

    # AI の処理
    try:
        outputs = recommend(list_data)
    except:
        outputs = [[0], [0], [0]]
        print("AIアルゴリズムのエラー")
    # ここまでAIの処理

    # outputs を recommended_actresses に格納
    recommended_actresses = {
        "recommended_actresses_ids": outputs[0],
        "states": outputs[1],
        "epsilons": outputs[2]
    }
    response = make_response(jsonify(recommended_actresses))
    response.headers["Access-Control-Allow-Origin"] = "*"
    # Backend サーバーに JSONを返す
    return response


@app.route('/learning', methods=['POST'])
def learn():
    # postされたデータ受け取り
    data = request.get_json()
    learning_data = []
    # 配列に挿入
    for d in data:
        l = [d['states'], d['epsilons'], d['id']]
        learning_data.append(t)
    # ここからAIの処理
    try:
        msg = {"msg": "Learning was successed"}
    except:
        msg = {"msg": "Learning was failed"}
    # ここまでAIの処理　

    # レスポンス返す
    response = make_response(msg)
    response.headers["Content-Type"] = "application/json"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

    # export FLASK_APP=server.py + flask run または python3 server.py で起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
