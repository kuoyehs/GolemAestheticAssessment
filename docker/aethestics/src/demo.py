from flask import Flask, Blueprint
from flask import request
from flask import Response
from flask import render_template

import json

bp = Blueprint('auth', __name__)


def Response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/echarts/caption1')
def echarts1():
    datas = {
        "data": [
            {"num": 0.00847206823527},
            {"num": 0.00956442393362},
            {"num": 0.05627568066120},
            {"num": 0.16258512437343},
            {"num": 0.31094545125961},
            {"num": 0.24895244836807},
            {"num": 0.11819330602884},
            {"num": 0.05404437333345},
            {"num": 0.01977592147886},
            {"num": 0.01119113713502},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption2')
def echarts2():
    datas = {
        "data": [
            {"num": 0.0021513532847},
            {"num": 0.0003247390268},
            {"num": 0.0125540206208},
            {"num": 0.0415405891835},
            {"num": 0.1553033441305},
            {"num": 0.2880765199661},
            {"num": 0.2239273935556},
            {"num": 0.1473982185125},
            {"num": 0.0924648791551},
            {"num": 0.0362588576972},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption3')
def echarts3():
    datas = {
        "data": [
            {"num": 0.0021513532847},
            {"num": 0.0003247390268},
            {"num": 0.0125540206208},
            {"num": 0.0415405891835},
            {"num": 0.1553033441305},
            {"num": 0.2880765199661},
            {"num": 0.2239273935556},
            {"num": 0.1473982185125},
            {"num": 0.0924648791551},
            {"num": 0.0362588576972},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption4')
def echarts4():
    datas = {
        "data": [
            {"num": 0.01507144235074},
            {"num": 0.02259951271116},
            {"num": 0.05490336194634},
            {"num": 0.13497579097747},
            {"num": 0.26123875379562},
            {"num": 0.23518401384353},
            {"num": 0.13096800446510},
            {"num": 0.07285019010305},
            {"num": 0.04460401087999},
            {"num": 0.02760487981140},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption5')
def echarts5():
    datas = {
        "data": [
            {"num": 0.010680878534913},
            {"num": 0.016916198655962},
            {"num": 0.064250253140926},
            {"num": 0.176680624485015},
            {"num": 0.300408601760864},
            {"num": 0.231857389211654},
            {"num": 0.112066902220249},
            {"num": 0.051209580153226},
            {"num": 0.022688966244459},
            {"num": 0.013240592554211},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption6')
def echarts6():
    datas = {
        "data": [
            {"num": 0.00412417482584},
            {"num": 0.00272141303867},
            {"num": 0.02072184346616},
            {"num": 0.07525141537189},
            {"num": 0.23338444530963},
            {"num": 0.29408660531044},
            {"num": 0.18907172977924},
            {"num": 0.10622318089008},
            {"num": 0.05135880783200},
            {"num": 0.02305644005537},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption7')
def echarts7():
    datas = {
        "data": [
            {"num": 0.0079222731292},
            {"num": 0.0029799295589},
            {"num": 0.0411808751523},
            {"num": 0.1190156340599},
            {"num": 0.2719620764255},
            {"num": 0.2712507545948},
            {"num": 0.1499943286180},
            {"num": 0.0759594812989},
            {"num": 0.0385106876492},
            {"num": 0.0212239474058},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption8')
def echarts8():
    datas = {
        "data": [
            {"num": 0.035882577300073},
            {"num": 0.045833203941581},
            {"num": 0.131803274154663},
            {"num": 0.236108750104903},
            {"num": 0.277652859687806},
            {"num": 0.154610440135000},
            {"num": 0.071062922477720},
            {"num": 0.029055535793300},
            {"num": 0.009942200966173},
            {"num": 0.00804826710373},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption9')
def echarts9():
    datas = {
        "data": [
            {"num": 0.010630455799400},
            {"num": 0.010424138046801},
            {"num": 0.054196875542402},
            {"num": 0.1525678932666770},
            {"num": 0.300705075263977},
            {"num": 0.261208385229110},
            {"num": 0.123351000249385},
            {"num": 0.053786676377058},
            {"num": 0.021131426095962},
            {"num": 0.011998089961707},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption10')
def echarts10():
    datas = {
        "data": [
            {"num": 30},
            {"num": 20},
            {"num": 23},
            {"num": 100},
            {"num": 102},
            {"num": 90},
            {"num": 70},
            {"num": 20},
            {"num": 13},
            {"num": 3},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption11')
def echarts11():
    datas = {
        "data": [
            {"num": 13},
            {"num": 21},
            {"num": 23},
            {"num": 33},
            {"num": 66},
            {"num": 90},
            {"num": 70},
            {"num": 60},
            {"num": 13},
            {"num": 3},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp


@bp.route('/echarts/caption12')
def echarts12():
    datas = {
        "data": [
            {"num": 30},
            {"num": 60},
            {"num": 73},
            {"num": 100},
            {"num": 60},
            {"num": 50},
            {"num": 60},
            {"num": 30},
            {"num": 23},
            {"num": 13},
        ]
    }
    content = json.dumps(datas)
    resp = Response_headers(content)
    return resp
