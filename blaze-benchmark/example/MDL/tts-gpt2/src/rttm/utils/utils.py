#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/16
# update: 2020/9/16
"""
utils.py.py:
"""

import json
import urllib.parse
import requests


def call_ost_server(max_length, min_length, count, scenes, title, one_keyword):
    base_url = "http://scs.chuangyi.taobao.com"
    params = {"appid": "101007008", "pid": "alimama_nlp", "name": "scs", "callback": "cc", "max_length": str(max_length), "min_length": str(min_length), "count": str(count), "scenes": scenes, "title": str(title), "ob_ext": one_keyword}
    quote_str = urllib.parse.quote(json.dumps(params))
    urlencode_result = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    url = base_url + "/?" + urlencode_result
    return requests.get(url)


def call_alinlp_ner_server(text):
    base_url = "https://nlp-api.alibaba-inc.com/api/getAquilaNlpResult"
    params = {"uuid": "f54b1d9cbc25472e8ec4d6f2440f7e39",
              "token": "abmlhhggaklajbjo",
              "lexer_id": "ECOM",
              "text": text}
    urlencode_result = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    url = base_url + "/?" + urlencode_result
    return requests.get(url)


if __name__ == "__main__":
    ret = call_ost_server(8, "online_short_title", "你今天真好看", "")
    print(ret)
