#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: ored
# date: 2020/9/30
# update: 2020/9/30
"""
ost_two_line.py.py:
"""

import json
from framework.component import Component
from rttm.utils.utils import call_alinlp_ner_server


class OSTTwoLine(Component):
    """
    require: ["text"]
    provide: ["text_two_line"]
    """

    @classmethod
    def name(cls):
        return "ost_two_line"

    def __init__(self, conf_dict):
        # super.__init__(conf_dict)
        pass

    def process(self, message):
        if isinstance(message, list):
            for m in message:
                text = m.get("text")
                pre_length = m.get("pre_length")
                text_two_line_list = []
                for t in text:
                    alinlp_ner_result = call_alinlp_ner_server(t)
                    result = json.loads(alinlp_ner_result.text)["result"]
                    pre = 0
                    pre_list = []
                    aft_list = []
                    for r in result:
                        word = r["semantic_words"][0]["word"]
                        if pre + len(word) <= int(pre_length):
                            pre_list.append(word)
                            pre += len(word)
                        else:
                            aft_list.append(word)
                    text_two_line = "".join(pre_list) + "\001" + "".join(aft_list)
                    text_two_line_list.append(text_two_line)
                m.set("text_two_line", text_two_line_list)
        else:
            text = message.get("text")
            pre_length = message.get("pre_length")
            alinlp_ner_result = call_alinlp_ner_server(text)
            result = json.loads(alinlp_ner_result.text)["result"]
            pre = 0
            pre_list = []
            aft_list = []
            for r in result:
                word = r["semantic_words"][0]["word"]
                if pre + len(word) <= int(pre_length):
                    pre_list.append(word)
                    pre += len(word)
                else:
                    aft_list.append(word)
            text_two_line = "".join(pre_list) + "\001" + "".join(aft_list)
            message.set("text_two_line", text_two_line)
