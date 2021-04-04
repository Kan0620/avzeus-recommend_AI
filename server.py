# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, make_response, redirect
from AVzeus import recommend


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/recommendation', methods=['GET'])
def inputted_data():
    data = request.args.get("selected_wemen_ids")
    list_data = data.split(',')
    # list_data: dataをint型の配列に変換したもの
    list_data = [int(i) for i in list_data]

    # AI の処理
    outputs = recommend(list_data)
    # ここまでAIの処理

    # outputs を recommended_actresses に格納
    recommended_actresses = {
        "actresses_ids": outputs[0],
        "states": outputs[1],
        "epsilons": outputs[2]
    }
    # Backend サーバーに JSONを返す
    return jsonify(recommended_actresses)


# export FLASK_APP=server.py + flask run または python3 server.py で起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
